import torch 
import torch.nn as nn
import math


# # here we are taking d_k = d_v

# def SPDA(Q, K, V, mask =None):

#         d_k = K.size(-1)
#         assert Q.size(-1) == K.size(-1)
#         attention_weights = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)   # scaled by root(dk)
#         # print(f"Shape of attention scores after unsqueezing for head dim: {attention_weights.shape}")   # shape is like (B,seq_len, seq_len)

#         if mask is not None:   #mask if only additive_key_padding_mask is (B,1,k=seq_len)  and for additive_attn_mask is (1,q=seq_len,k=seq_len)  so for masked MHA the broadcasted shape is (B,seq_len,seq_len)
#             attention_weights = attention_weights + mask # shape will be (B,q=seq_len,k=seq_len)  
#         # print(f"attention scores after broadcasting(due to additon is : {attention_weights.shape})")
#         attention_weights = torch.softmax(attention_weights, dim=-1)    # For each query, attention weights over all keys must sum to 1.
#         attention = torch.matmul(attention_weights, V)
#         # attention = attention.squeeze(1)
#         return attention, attention_weights


def better_SPDA(Q,K,V, mask = None):

    d_k = K.size(-1)
    assert Q.size(-1) == K.size(-1)
    attention_weights = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
    # expected shape is (B,n_heads,seq_len,seq_len)
    # print(f"Shape of the attenion weights : {attention_weights.shape}")

    # mask shape 4D  key_padding_mask is (B,1,1,seq_len) for encoder MHA
    # but attn_mask = (1,1,seq_len,seq_len) will be broadcasted to (B,1,seq_len,seq_len) and key_padding_mask will be broadcasted to (B,1,seq_len,seq_len) similarily 

    if mask is not None :
        # print(f"shape of my additive mask is : {mask.shape}") 
        attention_weights = attention_weights + mask   # mask broadcasted shape will eb (B,n_heads,seq_len,seq_len) 
    
    attention_weights = torch.softmax(attention_weights, dim =-1 )   # for each query, attention weights over all keys must sum to 1 
    attention = torch.matmul(attention_weights, V)   # shape would be (B,n_heads,seq_len,head_dim)
    # print(f"Shape of attention after matmul with V: {attention.shape}")
    return attention, attention_weights   



# supports self-attention (where Q=K=V=X) and cross-attention too!
# class AttentionBlock(nn.Module):

#     def __init__(self, input_dim, output_dim):

#         super().__init__()

#         # linear transformation on input
#         # print(f"debug 1 inside AB")
#         self.W_q = torch.nn.Linear(input_dim, output_dim)
#         # print(f"debug 2 inside AB")
#         self.W_k = torch.nn.Linear(input_dim, output_dim)
#         # print(f"debug 3 inside AB")
#         self.W_v = torch.nn.Linear(input_dim, output_dim)

#     def forward(self, Q, K, V, mask = None):
#         # project Q, K, V  (input_dim->output_dim)
#         Q = self.W_q(Q)
#         K = self.W_k(K)
#         V = self.W_v(V)
        
#         # print(f"shape of q,k,v is : {Q.shape}")

#         attn, weights = SPDA(Q,K,V, mask=mask)

#         return attn, weights

'''
Notes: Here the input_dim for both single and multi head attention would be d_model but in MHA the output_dim would be d_model/n_heads but in single head it would be d_model

'''
# import numpy as np
# class MHA(nn.Module):
#     def __init__(self, d_model, n_heads):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

#         self.embed_dim = d_model
#         self.n_heads = n_heads
#         head_embed_dim = self.embed_dim // n_heads
#         blocks = []
#         # creating N seperate layers, each layer handles one head expected 3D tensor --> (batch,seq_len,head_embed_dim) and not 4D(with head) 
#         for i in range(self.n_heads):
#             one_head = AttentionBlock(input_dim=d_model, output_dim=head_embed_dim)   
#             blocks.append(one_head)                 
#         self.head_blocks = torch.nn.ModuleList(blocks)
#         self.projection = torch.nn.Linear(d_model, d_model) 

#     def forward(self,Q,K,V, mask = None):
#         attns_list = []
#         for head in self.head_blocks:
#             attn, _ = head.forward(Q, K, V, mask=mask)
#             # print(f"Shape of attn (inside the MHA forward pass ) : {attn.shape}")
#             attns_list.append(attn)

#         attns = torch.cat(attns_list, dim=-1)
#         # attns = attns.squeeze(1) # fix the unsqueeze we did in SPDA 
#         # print(f"debug 4, shape of attns in MHA forward pass after concat is : {attns.shape}")
#         linear_proj = self.projection(attns)
#         return linear_proj



class better_MHA(nn.Module):
    def __init__(self,d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.embed_dim = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)

        self.projection = torch.nn.Linear(d_model, d_model)

        
    def forward(self,Q,K,V, mask = None):

        B,seq_len_q,_ = Q.size()
        _,seq_len_k,_ = K.size()  # very important to get different seq_lens (T_q for Q and T_k for K) for cross attention to work properly as Q comes from decoer having max_len-1 while K and V from encoder have seq_len as max_len

        projected_Q = self.W_q(Q).view(B, seq_len_q, self.n_heads, self.head_dim).transpose(dim0=2,dim1=1)
        projected_K = self.W_k(K).view(B, seq_len_k, self.n_heads, self.head_dim).transpose(dim0=2,dim1=1)
        projected_V = self.W_v(V).view(B, seq_len_k, self.n_heads, self.head_dim).transpose(dim0=2,dim1=1)

        # viewing splits the last dimension(d_model) to (n_heads*head_embed_dim) shape from (B,T,D) to (B,T,n_heads,head_embed_dim) and then transpose to (B,n_heads,T,head_embed_dim)

        attn, _ = better_SPDA(Q=projected_Q,K=projected_K,V=projected_V,mask = mask)
        
        # attn has shape (B,n_heads,seq_len_q,head_dim)  

        # after transpose it should be (B,seq_len_q, n_heads,head_dim) and then concat
        attn = attn.transpose(2,1).contiguous().view(B,seq_len_q,self.embed_dim)   # after transpose dim not contiguous so need contiguous()
        # print(f"shape of attn after concat is {attn.shape}")
        linear_proj = self.projection(attn)
        return linear_proj





class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff = 128, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads= n_heads
        self.mha = better_MHA(d_model, n_heads)
        # 1st Layer Norm after MHA
        self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=d_model)
        # 2nd Layer Norm after FFN
        self.layer_norm2 = torch.nn.LayerNorm(normalized_shape=d_model)

        # dropout for MHA sublayer
        self.dropout1 = torch.nn.Dropout(p=dropout)
        # dropout for FFN sublayer
        self.dropout2 = torch.nn.Dropout(p=dropout)

        self.posisiton_wise_ff_net = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=d_ff, out_features=d_model)
        )

    def forward(self , x, key_padding_mask = None):
        # (B, seq_len) --> (B,1,seq_len)
        if key_padding_mask is not None:
            additive_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).float() * -1e9     # convert the 2D mask (B,seq_len) to 4D (B,1,1,seq_len) to be easily broadcastable
        else:
            additive_mask = None
        attn_output = self.mha.forward(x,x,x, mask = additive_mask)  # encoder attention and mask the <PAD> tokens
        # print("inside the forward func of the EncoderLayer")
        x = x + self.dropout1(attn_output)
        # print(f"1 shape : {x.shape}")
        x = self.layer_norm1(x)
        # print(f"2 shape : {x.shape}")

        output_ff = self.posisiton_wise_ff_net(x)
        # print(f"3 shape : {x.shape}")

        x = x + self.dropout2(output_ff)
        x = self.layer_norm2(x)

        return x
    

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff = 128, dropout = 0.1):
        super().__init__()
        self.mha1 = better_MHA(d_model, n_heads)   # first MHA block (decoder attention)
        self.mha2 = better_MHA(d_model, n_heads)   # second MHA block (cross attention)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=d_model)  # after MHA1  normalized over d_model 
        self.norm2 = torch.nn.LayerNorm(normalized_shape=d_model)  # after MHA2
        self.norm3 = torch.nn.LayerNorm(normalized_shape=d_model)  # after projection 
        self.dropout1 = torch.nn.Dropout(p=dropout)   # after MHA1
        self.dropout2 = torch.nn.Dropout(p=dropout)   # after MHA2 
        self.dropout3  =  torch.nn.Dropout(p=dropout) # After projection
        self.posisiton_wise_ff_net = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=d_ff, out_features=d_model)
        )
        # self.linear = torch.nn.Linear(d_model,d_model)

    def forward(self,x, encoder_out, tgt_key_padding_mask, src_key_padding_mask, attn_mask):
        # print("Inside the forward pass of the decoder block")
        # print(f"Shape of key_mask is {key_padding_mask.shape} and shape of attn_mask is {attn_mask.shape}")
        tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B,seq_len_q) --> (B,1,1,seq_len_q)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)   # (seq_len_q,seq_len_q)  --> (1,1,seq_len_q,seq_len_q) 
        combined_mask = tgt_key_padding_mask | attn_mask 
        additive_mask = combined_mask.float() * -1e9
        # print(f"Shape of the additive mask : {additive_mask.shape}")
        # MHA block 1 (decoder self attention, needs to use casual mask)
        attn_output = self.mha1.forward(x,x,x, mask = additive_mask) 
        x = self.norm1(x+self.dropout1(attn_output))
        # MHA block 2 (cross attention, needs just to use key_mask)
        src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(1).float() * -1e9

        #CROSS ATTENTION 
        # pass the src_key_padding_mask for getting T_k required for cross attention in the decoder key_padding_mask dim to be (B,1,1,T_k) and not T_q, that's why it is called "key" padding mask
        # we do this to match and be broadcastable to the attention shape in cross attention which is (B,n_heads,T_q,T_k)
        attn_output2 = self.mha2.forward(Q=x, K=encoder_out, V=encoder_out, mask = src_key_padding_mask) 

        x = self.norm2(x+self.dropout2(attn_output2))
        projection = self.posisiton_wise_ff_net(x)
        x = self.norm3(x+self.dropout3(projection))
        return x



class PosistionalEncoding(nn.Module):
    def __init__(self,d_model, dropout=0.1,max_len = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        posistion = torch.arange(max_len).unsqueeze(1)  # shape : [max_len,1]
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0)/d_model))   # evaluates same to 1/10000^(2i/d_model)  shape : [d_model/2]
        pe = torch.zeros(1,max_len,d_model)
        pe[0,:,0::2] = torch.sin(posistion*div_term)   # shape after broadcasting : [max_len, d_model/2]
        pe[0,:,1::2] = torch.cos(posistion*div_term)   # shape after broadcasting : [max_len, d_model/2]
        self.register_buffer("pe", pe)  #
        
    def forward(self,x):   # shape of x [batch_size, seq_len, d_model]   
        x = x + self.pe[:,:x.size(1)]     # first slicing then broadcasting to [batch_size, seq_len, d_model]  (only require seq_len out of max_len)
        return self.dropout(x)



class Transformer(torch.nn.Module):
    def __init__(self,
                 N,
                #  batch_size,
                #  seq_len,
                 d_model,
                 d_ff,
                 n_heads,
                 dropout_rate,
                 max_len,
                 src_vocab_size,
                 tgt_vocab_size):
        super().__init__()
        self.N = N
        # self.batch_size = batch_size
        # self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.pos_encoding = PosistionalEncoding(d_model=d_model,dropout=dropout_rate, max_len=max_len)
        self.src_embed = torch.nn.Embedding(src_vocab_size,d_model)
        self.tgt_embed = torch.nn.Embedding(tgt_vocab_size,d_model)

        e_blocks = []
        for _ in range(N):
            one_encoder = EncoderLayer(d_model = d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout_rate) 
            e_blocks.append(one_encoder)
        self.encoder_blocks = torch.nn.ModuleList(e_blocks)
        
        d_blocks = []
        for _ in range(N):
            one_decoder = DecoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout_rate)
            d_blocks.append(one_decoder)
        self.decoder_blocks = torch.nn.ModuleList(d_blocks)


        self.last_decoder_ff = torch.nn.Linear(in_features=d_model, out_features=tgt_vocab_size)
        # self.softmax = torch.nn.Softmax(dim=1)  # softmax accross the seq_len 

    
    def encode(self, src, src_mask):   # pass in (B,seq_len) src_mask
        x = self.src_embed(src) 
        x = self.pos_encoding.forward(x)
        for block in self.encoder_blocks:
            x = block.forward(x, src_mask)

        encoder_out = x
        return encoder_out  #  
    
    def decode(self, tgt, encoder_out, tgt_key_padding_mask, src_key_padding_mask, attn_mask):  # pass in (seq_len_q,seq_len_q) shape attn_mask and (B,seq_len_q) pad_mask and (B,seq_len_k) from encoder 
        x = self.tgt_embed(tgt)
        x = self.pos_encoding.forward(x)
        for block in self.decoder_blocks:
            x = block.forward(x=x,encoder_out=encoder_out,src_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask,attn_mask=attn_mask)
        return x

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, attn_mask):
        encoder_out = self.encode(src=src, src_mask=src_key_padding_mask)
        # print(f"ENCODER BLOCKS DONE !!!")

        decoder_out = self.decode(tgt=tgt, encoder_out=encoder_out, src_key_padding_mask= src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, attn_mask=attn_mask)
        #need to also pass the src_key_padding_mask for getting T_k required for cross attention in the decoder key_padding_mask dim to be (B,1,1,T_k) and not T_q, that's why it is called "key" padding mask4

        out = self.last_decoder_ff(decoder_out)
        # out_prob = self.softmax(linear_out)
        # CrossEntropy loss applies softmax automatically!

        return out


def main():
    d_model = 64
    n_heads = 4
    seq_len = 10
    batch_size = 2
    d_ff = 128


    src = torch.randn(batch_size, seq_len, d_model)
    print(f"Input Shape: {src.shape}")

    # taking 2 sentences (batch_size = 2) 
    # 1 is 10 long 2 is 6 long

    # 1 is for mask and 0 is for no_mask 

    # shape before is (batch,seq_len)
    key_mask_padding =  torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # length 10
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],   # length 6 + padding
])
    key_mask_padding = key_mask_padding.unsqueeze(1)

    key_additive_mask = key_mask_padding.float() * -1e9
    
    # shape of mask : (batch, 1, k) --(broadcasted)--> (batch, q, k)

    print(f"Mask Shape: {key_additive_mask.shape}") 

    encoder_layer = EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    encoder_out = encoder_layer.forward(src, mask=key_additive_mask)

    print("Encoder forward pass successful!")
    print(f"Encoder Output Shape: {encoder_out.shape}")

    decoder_layer = DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    #shifted right
    target = torch.randn(batch_size,seq_len, d_model)
    # not needed now, i'll use it once I make the padding 
    decoder_input = src

    # triu 1's the upper triangle in a matix
    attention_mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1) # we want to mask out the future tokens and not the current token to attend to
    attention_mask = attention_mask.unsqueeze(0)   # from (seq_len,seq_len) to (1, seq_len, seq_len) to be able to be broadcastable with the key_mask_padding (B,1,seq_len)
    print(f"Shape of attention mask after unsqueeze: {attention_mask.shape}")
    decoder_out = decoder_layer.forward(decoder_input, encoder_out, key_mask_padding.bool(), attention_mask.bool())

    print("Decoder forward pass successful!")
    print(f"Decoder Output Shape: {decoder_out.shape}")


if __name__ == "__main__":
    main()
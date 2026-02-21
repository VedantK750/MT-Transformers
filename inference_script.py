from translationDataset import TranslationDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Transformer
import sentencepiece as spm
import torch


en_file = "test.en"
de_file = "test.de"

sp = spm.SentencePieceProcessor()
sp.load("mt_bpe.model")

N = 6
MAX_LEN = 256
d_model = 128
d_ff = 2048
n_heads = 8
dropout_rate = 0.1
src_vocab_size = sp.get_piece_size()
tgt_vocab_size = sp.get_piece_size()



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(N,d_model,d_ff,n_heads,dropout_rate,MAX_LEN,src_vocab_size,tgt_vocab_size).to(DEVICE)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))


test_dataset = TranslationDataset(en_file=en_file, de_file=de_file, tokenizer=sp)

sample = test_dataset[50]
src = sample["src"].unsqueeze(0).to(DEVICE)  

def greedy_decoding(model, src , pad_id, bos_id, eos_id):
    model.eval()
    with torch.no_grad():
        encoder_out= model.encode(src, src_mask = None)
        ys = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)
        # [1,1] because decoder expects [batch, seq_len]
        # dtype long required for nn.Embedding
        # placed on same device as model
        src_key_padding_mask = torch.zeros_like(src).bool().to(DEVICE)
        
        for _ in range(MAX_LEN):
            # print(f"Shape of ys : {ys.shape}")
            tgt_key_padding_mask = torch.zeros_like(ys).bool().to(DEVICE)
            seq_len = ys.size(1)
            attn_mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).bool().to(DEVICE)
            
            decoder_out = model.decode(tgt = ys, encoder_out = encoder_out, src_key_padding_mask = src_key_padding_mask, tgt_key_padding_mask = tgt_key_padding_mask, attn_mask=attn_mask)
            logits = model.last_decoder_ff(decoder_out[:, -1, :])  # we only need the last logit
            print(f"Shape of last logits is : {logits.shape}")   # [B,vocab_size]
            
            top_k_values, top_k_indices = torch.topk(logits, k=10, dim=-1)
            
            print(f"\nTime step {ys.size(1)-1}:")
            for i in range(5):
                token_id = top_k_indices[0, i].item()
                token_piece = sp.id_to_piece(token_id)
                score = top_k_values[0, i].item()
                print(f"{i+1}. {token_piece}  (logit={score:.4f})")
                
            next_token = torch.argmax(logits, dim=-1)  # dim=-1 because [batch_size,seq_len]
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)  # dim=1 → time/sequence dimension  because [batch_size, sequence_length]
            
            if next_token.item() == eos_id:
                break
    return ys

output_ids = greedy_decoding(
    model,
    src,
    pad_id=sp.pad_id(),
    bos_id=sp.bos_id(),
    eos_id=sp.eos_id()    
)

print(output_ids.shape)
pred_tokens = output_ids.squeeze(0).tolist()
pred_text = sp.decode(pred_tokens)

print(f"shape of sample['src'] is : {sample['src'].shape} ")

src_text = sp.decode(sample["src"].tolist())
tgt_text = sp.decode(sample["tgt"].tolist())

print("SOURCE:", src_text)
print("TARGET:", tgt_text)
print("PRED  :", pred_text)
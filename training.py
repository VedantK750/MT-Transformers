import torch
from translationDataset import TranslationDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import torch.optim as optim
from model import Transformer
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


en_file = "/home/vedant/machine_translation_using_transformers_from_scratch/train.en"
de_file = "/home/vedant/machine_translation_using_transformers_from_scratch/train.de"

en_val = "val.en"
de_val = "val.de"

sp = spm.SentencePieceProcessor()
sp.load("mt_bpe.model")


BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_EPOCHS = 50
DEVICE = "cpu"
b1 = 0.9
b2 = 0.98
epsilon = 1e-09
intial_lr = 1.0
warmup_steps = 1000

#####(Model Hyperparameters)##############
N = 6
MAX_LEN = 256
d_model = 128
d_ff = 2048
n_heads = 8
dropout_rate = 0.1
src_vocab_size = sp.get_piece_size()
tgt_vocab_size = sp.get_piece_size()
##########################################


def custom_collate(batch,pad_id):
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    # Shift target
    decoder_input = tgt_padded[:, :-1]
    decoder_target = tgt_padded[:, 1:]

    return {
        "src": src_padded,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target
    }


pad_id = sp.pad_id()

train_dataset = TranslationDataset(en_file=en_file, de_file=de_file, tokenizer=sp)
val_dataset = TranslationDataset(en_file=en_val, de_file = de_val, tokenizer=sp)

training_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: custom_collate(batch, pad_id), 
                        num_workers=NUM_WORKERS, pin_memory=True)

validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: custom_collate(batch, pad_id), 
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)


model = Transformer(N,d_model,d_ff,n_heads,dropout_rate,MAX_LEN,src_vocab_size,tgt_vocab_size)

# step is just a number 
# and calling scheduler.step() just incrememnts this step counter 
def lr_lambda(step):
    step = max(step,1)
    return (d_model** -0.5) * min(
        step ** -0.5, 
        step * warmup_steps ** -1.5
    )


optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(b1,b2), eps=epsilon) 
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

def train_epoch(epoch):
    total_loss = 0.0

    progress_bar = tqdm(training_loader, desc=f"Epoch {epoch+1}")

    # interactive plot while training 
    plt.ion()                      
    fig, ax = plt.subplots()
    losses = []

    for step, batch in enumerate(progress_bar):
        src = batch["src"].to(DEVICE)
        dec_in = batch["decoder_input"].to(DEVICE)
        dec_out = batch["decoder_target"].to(DEVICE)

        src_key_padding_mask = (src == pad_id)
        tgt_key_padding_mask = (dec_in == pad_id)
        seq_len = dec_in.size(1)
        attn_mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).bool()
        attn_mask = attn_mask.to(src.device)  # push attn_mask to DEVICE

        optimizer.zero_grad()
        outputs = model.forward(src,dec_in,
                                src_key_padding_mask,
                                tgt_key_padding_mask,
                                attn_mask)
        
        vocab_size = outputs.size(-1)
        # print(f"Shape of outputs : {outputs.shape} and dec_out shape : {dec_out.shape}")
        loss = criterion(outputs.contiguous().view(-1,vocab_size), dec_out.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        losses.append(loss_value)

        progress_bar.set_postfix(loss=loss.item())  # this takes in a dictionary to be displayed 
        if step % 20 == 0:
            ax.clear()
            ax.plot(losses)
            ax.set_xlabel("Batch Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"Epoch {epoch+1} Running Loss")
            plt.pause(0.01)
        
        plt.ioff()                
        plt.show()

    return total_loss/len(training_loader)  # length of the training loader is the number of batches in 1 epoch 


def validater():
    total_loss = 0.0
    for i,batch in enumerate(validation_loader):
        src = batch["src"].to(DEVICE)
        dec_in = batch["decoder_input"].to(DEVICE)
        dec_out = batch["decoder_target"].to(DEVICE)

        src_key_padding_mask = (src == pad_id)
        tgt_key_padding_mask = (dec_in == pad_id)
        seq_len = dec_in.size(1)
        attn_mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).bool()
        attn_mask = attn_mask.to(src.device)  # push attn_mask to DEVICE

        outputs = model.forward(src,dec_in,
                                src_key_padding_mask,
                                tgt_key_padding_mask,
                                attn_mask)
        vocab_size = outputs.size(-1)
        loss = criterion(outputs.contiguous().view(-1,vocab_size), dec_out.contiguous().view(-1))
        total_loss += loss.item()

    return total_loss/len(validation_loader)

def bleu_score_val():
    predictions = []
    references = []

    pass

best_val_loss = float("inf")
for epoch in range(NUM_EPOCHS):
    # train phase
    model.train()
    avg_loss = train_epoch(epoch)
    print(f"Epoch {epoch+1} avg TRAIN loss: {avg_loss:.4f}")

    # val phase
    model.eval()
    with torch.no_grad():
        # NOTE: even during validation we are doing teacher forcing for calculating the VAL LOSS
        val_loss = validater()
    print(f"Epoch {epoch+1} avg VAL loss: {val_loss:.4f}")
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved new best model.")







    




    





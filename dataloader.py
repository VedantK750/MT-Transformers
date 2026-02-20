from datasets import load_dataset
from transformers import AutoTokenizer
from model import Transformer
import torch


MAX_LENGTH = 256

# ds = load_dataset("bentrevett/multi30k")

train_dataset = load_dataset("bentrevett/multi30k", split="train")
# valid_dataset = load_dataset("bentrevett/multi30k", split="validation")
# test_dataset  = load_dataset("bentrevett/multi30k", split="test")

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def tokenization(example):
    src = tokenizer(
        example["en"],
        truncation=True,
        padding="max_length",
    )
    tgt = tokenizer(
        example["de"],
        truncation=True,
        padding="max_length",
    )

    return {
        # "en" : example["en"],
        # "de" : example["de"],
        "src_input_ids": src["input_ids"],   
        "src_attention_mask": src["attention_mask"],   #will become key_padding_mask after inverting 0 and 1's (as 1 in HF means valid)
        "tgt_input_ids": tgt["input_ids"],
        "tgt_attention_mask": tgt["attention_mask"],    # similarily become key_padding_mask
    }

# the casual mask is then passed during training using torch.triu to the model directly



tokenizer.model_max_length = MAX_LENGTH


# tokenizing the train dataset (from words to token_ids)
train_dataset = train_dataset.map(tokenization, batched=True)

# valid_dataset = valid_dataset.map(tokenization, batched=True)
# test_dataset = test_dataset.map(tokenization, batched=True)


train_dataset.set_format(
    type="torch",
    columns=[
        # "en",
        # "de"
        "src_input_ids",
        "src_attention_mask",
        "tgt_input_ids",
        "tgt_attention_mask"
    ]
)


# Hyperparameters
N = 4
d_model = 512
d_ff = 2048
heads = 8

drop_rate = 0.1

src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = tokenizer.vocab_size

print(f"vocab_size : {tokenizer.vocab_size}")

model = Transformer(N=N,d_model=d_model,d_ff=d_ff, n_heads=heads, dropout_rate=drop_rate, max_len=MAX_LENGTH, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)

src_example = train_dataset[0]["src_input_ids"].unsqueeze(0)   #(1,src_seq_len)
tgt_example = train_dataset[0]["tgt_input_ids"][:-1].unsqueeze(0)   #(1,tgt_seq_len)

#NOTE: After embedding in the model forward pass it would be (1,src_seq_len,d_model) simillarily for the tgt 

print(f"shape of the src_example is : {src_example.shape}")

# print(f'the english sentence is : {train_dataset[0]["en"]}')

print(tokenizer.model_max_length)


tgt_size = tgt_example.size(0)  
attn_mask = torch.triu(torch.ones(tgt_size,tgt_size), diagonal=1).bool()   # building the casual mask for the example 

out = model.forward(src=src_example, tgt = tgt_example, src_key_padding_mask= (train_dataset[0]["src_attention_mask"] == 0).unsqueeze(0).bool(), tgt_key_padding_mask=(train_dataset[0]["tgt_attention_mask"][:-1] == 0).unsqueeze(0).bool(), attn_mask = attn_mask)
print(out)

# # text = str(train_dataset[0])
# # enc = tokenizer(text)
# # # print(enc["input_ids"])

# # dec = tokenizer.decode(enc["input_ids"])
# # print(dec)

# example = train_dataset[0]
# print(example.keys())
# # ids = enc[]

# # print(train_dataset)


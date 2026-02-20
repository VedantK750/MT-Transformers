from transformers import AutoTokenizer
from torch.utils.data import Dataset, dataloader
import torch
from torch.nn.utils.rnn import pad_sequence

MAX_LENGTH = 256


class TranslationDataset(Dataset):
    def __init__(self,en_file,de_file, tokenizer):
        with open(en_file) as f:
            self.en_sentences = f.readlines()
        with open(de_file) as f:
            self.de_sentences = f.readlines()
        assert len(self.en_sentences) == len(self.de_sentences)
        
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()

    def __len__(self):
        return len(self.en_sentences)
    
    def __getitem__(self,idx):
        src = self.en_sentences[idx].strip()
        tgt = self.de_sentences[idx].strip()

        src_tokenized = self.tokenizer.encode(src,out_type=int)
        tgt_tokenized = self.tokenizer.encode(tgt,out_type=int)

        tgt_tokens = [self.sos_id] + tgt_tokenized + [self.eos_id]

        return {
        "src": torch.tensor(src_tokenized),
        "tgt": torch.tensor(tgt_tokens)
        }


# shifting must happen after padding

# padding batch wise is better than a global max_len (saves memory) and implement padding and right shift logic in the collate function NOT in the __getitem__ 

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


def main(): 
    import sentencepiece as spm
    from torch.utils.data import DataLoader

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("mt_bpe.model")

    # Create dataset
    dataset = TranslationDataset(
        en_file="train.en",
        de_file="train.de",
        tokenizer=sp
    )

    # Basic checks
    print("Dataset size:", len(dataset))

    sample = dataset[1]
    print("Sample keys:", sample.keys())
    print("Source tokens:", sample["src"])
    print("Target tokens:", sample["tgt"])

    # Optional: test batching
    pad_id = sp.pad_id()

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: custom_collate(batch, pad_id),
        shuffle=True
    )

    batch = next(iter(loader))
    print(batch)
    print("Batch src shape:", batch["src"].shape)
    print("Batch tgt shape:", batch["decoder_target"].shape)
    print("Batch tgt input into decoder:", batch["decoder_input"].shape)


if __name__ == "__main__":
    main()
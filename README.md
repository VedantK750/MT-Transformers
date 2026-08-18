# MT-Transformers

A Transformer model for English → German machine translation, built from scratch in PyTorch (no `nn.Transformer` — custom multi-head attention, encoder/decoder stacks, and masking).

## Features

- Custom Transformer encoder-decoder implementation (`model.py`)
- Byte-Pair Encoding tokenizer trained with SentencePiece (`bpe_trainer.py`)
- Training loop with the original "Attention Is All You Need" learning-rate schedule (warmup + inverse-sqrt decay) and early stopping (`training.py`, `early_stop.py`)
- Greedy-decoding inference script (`inference_script.py`)

## Project Structure

```
model.py               # Transformer architecture (attention, encoder, decoder)
training.py             # Training loop, optimizer, LR schedule, early stopping
inference_script.py     # Greedy decoding / inference on the test set
dataloader.py            # HF-based tokenization experiment (Multi30k, BERT tokenizer)
translationDataset.py   # PyTorch Dataset for parallel en/de text
bpe_trainer.py            # Trains the SentencePiece BPE tokenizer
early_stop.py            # Early stopping callback
merge_txt.py / txt_file_creator.py / bpe_infer_test.py   # Data prep / testing utilities
mt_bpe.model / mt_bpe.vocab   # Trained SentencePiece tokenizer artifacts
test.en / test.de       # Test split (parallel English/German sentences)
```

## Setup

```bash
pip install torch sentencepiece tqdm matplotlib datasets transformers
```

## Usage

**Train the BPE tokenizer** (expects a `train_de_en_bpe` file with combined EN/DE text):
```bash
python bpe_trainer.py
```

**Train the model** (expects `train.en`, `train.de`, `val.en`, `val.de`):
```bash
python training.py
```

**Run inference** on the test set with the trained checkpoint (`best_model.pt`):
```bash
python inference_script.py
```

## Notes

- Model hyperparameters (layers, `d_model`, heads, dropout, etc.) are set at the top of `training.py` / `inference_script.py`.
- Training data files (`train.en`, `train.de`, `val.en`, `val.de`) and the model checkpoint are git-ignored — supply your own data or download Multi30k.

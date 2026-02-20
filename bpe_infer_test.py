import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.load("mt_bpe.model")

ids = sp.encode("Hi how are you doing?", out_type=int)

print(ids+ [sp.eos_id()])
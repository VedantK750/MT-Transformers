import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.load("mt_bpe.model")

ids1 = sp.encode("Boston Terrier", out_type=str)
ids2 = sp.encode("Rep-T-Shirt", out_type= str)
print(ids1)
print(ids2)
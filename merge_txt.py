

with open("train_de_en_bpe", 'w') as f:
    with open("train.en", 'r') as f1:
        f.write(f1.read())

    with open("train.de", "r") as f2:
        f.write(f2.read())

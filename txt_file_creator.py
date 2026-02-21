from datasets import load_dataset

dataset = load_dataset("bentrevett/multi30k",split="test")

with open("test.en", "w") as f_en, open("test.de", "w") as f_de:
    for ex in dataset:
        f_en.write(ex["en"] + "\n")
        f_de.write(ex["de"] + "\n")
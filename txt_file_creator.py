from datasets import load_dataset

dataset = load_dataset("bentrevett/multi30k",split="validation")

with open("val.en", "w") as f_en, open("val.de", "w") as f_de:
    for ex in dataset:
        f_en.write(ex["en"] + "\n")
        f_de.write(ex["de"] + "\n")
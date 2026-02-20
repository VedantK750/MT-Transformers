import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input = "train_de_en_bpe",
    model_prefix = 'mt_bpe',
    vocab_size = 10000,
    model_type = 'bpe',

    pad_id=0, pad_piece='<pad>',
    bos_id=1, bos_piece='<sos>',
    eos_id=2, eos_piece='<eos>',
    unk_id=3, unk_piece='<unk>',

    character_coverage=1.0            # fine for EN/DE
)


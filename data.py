import sentencepiece as spm
from torch.utils.data import DataLoader

def pad_sequence(sequence, max_len, pad_idx):
    return sequence[:max_len] + [pad_idx] * max(0, (max_len - len(sequence)))

def create_padding_mask(sequence, pad_idx):
    return [1 if token == pad_idx else 0 for token in sequence]

def prepare_data_split(split, sp, max_len, pad_idx):
    return split.map(lambda example: {
        'sentence': example['sentence'],
        'indices': pad_sequence(
            sp.encode(f"<BOS> {example['sentence']} <EOS>", out_type=int),
            max_len, pad_idx
        ),
        'mask': create_padding_mask(
            pad_sequence(sp.encode(f"<BOS> {example['sentence']} <EOS>", out_type=int), max_len, pad_idx),
            pad_idx
        )
    })

def collate_fn(batch):
    indices = torch.tensor([item['indices'] for item in batch])
    masks = torch.tensor([item['mask'] for item in batch])
    sentences = [item['sentence'] for item in batch]
    return {'indices': indices, 'masks': masks, 'sentences': sentences}

def prepare_dataloaders(train_data, val_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

def train_sentencepiece_model(data, vocab_size):
    sentences = [example['sentence'] for example in data]
    normalized_data = [sentence.lower().strip() for sentence in sentences]
    with open("train.txt", "w") as f:
        f.write("\n".join(normalized_data))
    spm.SentencePieceTrainer.train(
        input="train.txt",
        model_prefix="spm",
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )

def load_sentencepiece_model(model_path="spm.model"):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

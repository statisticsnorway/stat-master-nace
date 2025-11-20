from torch.utils.data import Dataset

class NaceDataset(Dataset):
    def __init__(self, df, input_cols, label_col, tokenizer):
        self.texts = df[input_cols].agg(" ".join, axis=1).tolist()
        self.labels = df[label_col].tolist()
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=128
        )
        tokenized["labels"] = self.labels[idx]
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.texts)

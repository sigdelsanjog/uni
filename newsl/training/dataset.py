import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class NewsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["newscontents"])  # Ensure it's a string
        tokenized = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(tokenized, dtype=torch.long)

def collate_fn(batch):
    """Pads sequences in the batch"""
    batch = [torch.tensor(sample, dtype=torch.long) for sample in batch]
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch, padded_batch  # Source and target are the same

def get_dataloader(file_path, tokenizer, batch_size=32, max_length=128):
    dataset = NewsDataset(file_path, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

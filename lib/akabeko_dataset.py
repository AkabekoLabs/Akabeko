import os
import torch
from torch.utils.data import Dataset

class AkabekoDataset(Dataset):
    def __init__(self, file_path: str, max_length: int = 512):
        self.max_length = max_length

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenized data not found: {file_path}")

        # PyTorch ≥2.4 推奨：weights_only=True でロード高速化
        tokens = torch.load(file_path, weights_only=True)

        if isinstance(tokens, (list, torch.Tensor)):
            self.tokens = tokens
        else:
            raise TypeError(f"Unexpected type for loaded tokens: {type(tokens)}")

    def __len__(self) -> int:
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.max_length
        end = start + self.max_length
        return torch.tensor(self.tokens[start:end], dtype=torch.long)

    def get_token_count(self) -> int:
        return len(self.tokens)

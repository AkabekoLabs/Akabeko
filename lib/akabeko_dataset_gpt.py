from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

class AkabekoDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # list of list of messages
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]  # [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True)

        if len(input_ids) == 0:
            return None

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # → ChatMLマスキングするならここ後で改造
        }

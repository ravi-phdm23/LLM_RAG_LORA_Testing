"""Helper classes and utilities."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AlpacaDataset(Dataset):
    """PyTorch dataset for Alpaca-style records."""

    def __init__(self, data_dir: str, model_name: str = "decapoda-research/llama-7b-hf", max_length: int = 512):
        self.data: List[Dict] = self._load_data(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    @staticmethod
    def _load_data(data_dir: str) -> List[Dict]:
        records: List[Dict] = []
        for file_name in os.listdir(data_dir):
            if not file_name.endswith(".json"):
                continue
            path = os.path.join(data_dir, file_name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    records.extend(data)
            except Exception:
                continue
        return records

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        record = self.data[idx]
        prompt = f"### Instruction:\n{record['instruction']}\n\n### Response:\n{record.get('output', '')}"
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


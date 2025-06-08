"""Training and inference utilities for LoRA fine-tuning.

This module provides reusable functions to load JSON training data, fine tune a
causal language model with LoRA adapters and run inference with the adapted
model.  The training process roughly follows the original Jupyter notebook
`final_fine_tuning.ipynb`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    model_id: str
    data_dir: str
    output_dir: str
    batch_size: int = 1
    grad_acc_steps: int = 2
    epochs: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def login_huggingface(token: str | None) -> None:
    """Login to HuggingFace Hub if a token is provided."""

    if not token:
        print("ℹ️ Hugging Face token not provided, skipping login")
        return

    try:
        login(token=token)
        print("✅ Logged in to Hugging Face Hub")
    except Exception as exc:
        print(f"❌ Failed to login: {exc}")


def load_dataset_from_json(folder_path: str) -> Dataset:
    """Load all JSON files under ``folder_path`` into a ``datasets.Dataset``."""

    records: List[dict] = []
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue
        path = os.path.join(folder_path, file_name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                records.extend(data)
            else:
                print(f"⚠️ Skipped {file_name}: not a list of records")
        except Exception as exc:
            print(f"⚠️ Failed to load {file_name}: {exc}")

    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    """Format a single example in Alpaca style."""

    text = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    ).format(**example)
    return {"text": text}


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenize dataset and add labels aligned to ``input_ids``."""

    dataset = dataset.map(format_prompt)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    tokenized = dataset.map(tokenize, batched=True)
    return tokenized.map(lambda x: {"labels": x["input_ids"]})


def create_model(config: TrainingConfig) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load base model and wrap it with LoRA adapters."""

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return tokenizer, model


def train(config: TrainingConfig) -> None:
    """Run the training loop."""

    tokenizer, model = create_model(config)
    raw_dataset = load_dataset_from_json(config.data_dir)
    tokenized = tokenize_dataset(raw_dataset, tokenizer)

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_acc_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        save_steps=500,
        logging_steps=250,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(config.output_dir)


def truncate_at_stop(text: str, stop_token: str = "### Instruction:") -> str:
    """Truncate generated text at the stop token."""

    return text.split(stop_token)[0].strip()


def compare_base_and_lora(
    prompts: Iterable[str],
    base_model_id: str,
    lora_adapter_path: str,
    stop_token: str = "### Instruction:",
) -> None:
    """Generate text for prompts using a base and LoRA adapted model."""

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path, is_trainable=False)
    lora_model.eval()
    lora_pipe = pipeline("text-generation", model=lora_model, tokenizer=tokenizer)

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n📌 Prompt {idx}:\n{prompt.strip()}\n")

        base_output = base_pipe(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        base_result = truncate_at_stop(base_output.replace(prompt, ""), stop_token)
        print("🔹 Base Model Response:\n", base_result, "\n")

        lora_output = lora_pipe(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        lora_result = truncate_at_stop(lora_output.replace(prompt, ""), stop_token)
        print("🔸 LoRA Model Response:\n", lora_result, "\n")
        print("—" * 100)


def main() -> None:
    """Entry point for training."""

    hf_token = os.getenv("HF_TOKEN")  # Optional token
    login_huggingface(hf_token)

    config = TrainingConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_dir="./training_data",
        output_dir="./lora_output",
    )

    train(config)


if __name__ == "__main__":
    main()

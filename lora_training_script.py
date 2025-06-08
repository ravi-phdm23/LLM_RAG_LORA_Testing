import json
import os
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch


def load_data(data_dir: str) -> Dataset:
    """Load all JSON files under ``data_dir`` and return a Dataset."""
    records = []
    for name in os.listdir(data_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(data_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                records.extend(data)
    return Dataset.from_list(records)


def format_alpaca(example: dict) -> dict:
    text = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    ).format(**example)
    return {"text": text}


def main() -> None:
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    data_dir = "./training_data"
    output_dir = "./lora_adapters"

    dataset = load_data(data_dir)
    dataset = dataset.map(format_alpaca)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        save_steps=500,
        logging_steps=250,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

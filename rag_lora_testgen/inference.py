"""Inference helpers for LoRA models."""

from __future__ import annotations

from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .retriever import load_faiss_index, search_index


def load_lora_pipeline(base_model_id: str, adapter_path: str):
    """Load a base model with LoRA adapters and return a generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def build_prompt(query: str, index, chunks: List[str], emb_model, top_k: int = 3) -> str:
    """Build a prompt with retrieved context and the user instruction."""
    results = search_index(query, index, chunks, emb_model, top_k)
    context = "\n".join(text for text, _ in results)
    return f"### Instruction:\n{query}\n\n### Context:\n{context}\n\n### Response:\n"


def generate(
    instruction: str,
    adapter_path: str,
    base_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    index_path: str = "faiss.index",
    metadata_path: str = "faiss.json",
    max_new_tokens: int = 200,
) -> str:
    """Generate a response using the LoRA adapted model and retrieved context."""
    index, chunks, emb_model = load_faiss_index(index_path, metadata_path)
    pipe = load_lora_pipeline(base_model_id, adapter_path)
    prompt = build_prompt(instruction, index, chunks, emb_model)

    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0][
        "generated_text"
    ]
    return output.replace(prompt, "").split("### Instruction:")[0].strip()


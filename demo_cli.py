import argparse
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from pdf_faiss_index import load_faiss_index, search_index


def load_lora_pipeline(base_model_id: str, adapter_path: str):
    """Return a text generation pipeline for the LoRA adapted model."""
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


def build_prompt(
    query: str,
    use_rag: bool,
    index: Optional[object] = None,
    chunks: Optional[List[str]] = None,
    emb_model: Optional[object] = None,
    top_k: int = 3,
) -> str:
    """Create a prompt with optional retrieved context."""

    if use_rag and index is not None and chunks is not None and emb_model is not None:
        results = search_index(query, index, chunks, emb_model, top_k)
        context = "\n".join(text for text, _ in results)
        return (
            f"### Instruction:\n{query}\n\n"
            f"### Context:\n{context}\n\n"
            "### Response:\n"
        )

    return f"### Instruction:\n{query}\n\n### Response:\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate answers with a fine-tuned LoRA model"
    )
    parser.add_argument("instruction", help="Instruction for the model")
    parser.add_argument("--use-rag", action="store_true", help="Retrieve Basel context")
    parser.add_argument("--adapter", default="./lora_output", help="Path to LoRA adapter")
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model identifier",
    )
    parser.add_argument("--index", default="faiss.index", help="Path to FAISS index")
    parser.add_argument("--meta", default="faiss.json", help="Path to metadata file")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Maximum tokens to generate")

    args = parser.parse_args()

    pipe = load_lora_pipeline(args.base_model, args.adapter)

    index = chunks = emb_model = None
    if args.use_rag:
        index, chunks, emb_model = load_faiss_index(args.index, args.meta)

    prompt = build_prompt(args.instruction, args.use_rag, index, chunks, emb_model)

    output = pipe(prompt, max_new_tokens=args.max_new_tokens, do_sample=False)[0]["generated_text"]
    result = output.replace(prompt, "").split("### Instruction:")[0].strip()
    print(result)


if __name__ == "__main__":
    main()

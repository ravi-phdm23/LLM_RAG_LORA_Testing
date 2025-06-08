# LLM_RAG_LORA_Testing

This repository contains utilities for experimenting with LoRA fine tuning and retrieval augmented generation (RAG) using the Basel regulations PDF.

## Demo CLI

`demo_cli.py` loads the fine‑tuned model and optionally retrieves context from `Basel_summary.pdf` via a FAISS index.

Build the index once using `pdf_faiss_index.py`:

```bash
python pdf_faiss_index.py --build
```

Run the CLI with or without retrieval:

```bash
# With RAG
python demo_cli.py "What are the Basel III capital requirements?" --use-rag

# Without RAG
python demo_cli.py "Summarise Basel III capital requirements"
```

Options are available to specify the LoRA adapter path, base model ID and index files.

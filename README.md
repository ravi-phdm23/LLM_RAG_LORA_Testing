# LLM_RAG_LORA_Testing

This project explores fine‑tuning a small language model with LoRA adapters and enhancing generation with retrieval augmented generation (RAG).  The Basel III summary PDF serves as the knowledge source.

## Model

Scripts such as `lora_training_script.py` fine‑tune **TinyLlama/TinyLlama-1.1B-Chat-v1.0** using Alpaca‑formatted JSON data from `training_data/`.

## Fine‑tuning with LoRA

1. Prepare your dataset inside `training_data/`.
2. Run the training script to produce adapters:
   ```bash
   python lora_training_script.py
   ```
   The script loads the base model, applies LoRA modules and saves the result under `lora_adapters/`.

## Retrieval Augmented Generation

`pdf_faiss_index.py` builds a FAISS index over the provided Basel summary. Retrieved text is appended to the instruction before generation to improve factuality.

Build the index once:
```bash
python pdf_faiss_index.py --build
```

## Running the Demo

Use `demo_cli.py` to interact with the fine‑tuned model. Pass `--use-rag` to include retrieved context.

```bash
# With RAG
python demo_cli.py "What are the Basel III capital requirements?" --use-rag

# Without RAG
python demo_cli.py "What are the Basel III capital requirements?"
```

Typical output with RAG looks like:

```
Basel III requires banks to maintain at least 4.5% Common Equity Tier 1 capital. Including the conservation buffer, this rises to around 7% with additional countercyclical buffers when required.
```

Adapter path, base model ID and index files can be customised via command line options.

## End-to-End Pipeline

Run all steps (training, indexing and inference) in one go with `run_pipeline.sh`:

```bash
./run_pipeline.sh "What are the Basel III capital requirements?"
```

The script first trains LoRA adapters, builds the FAISS index and finally runs `demo_cli.py` with RAG enabled.

#!/bin/bash

set -e

# Train LoRA adapters
python lora_training_script.py

# Build FAISS index from the Basel summary PDF
python pdf_faiss_index.py --build

# Run the demo with retrieval augmented generation
python demo_cli.py "$1" --use-rag

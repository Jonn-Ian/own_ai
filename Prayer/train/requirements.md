# Storage Formats
    - Model weights (parameters):
    - Saved in framework‑specific binary checkpoint files:
    - PyTorch: .pt or pytorch_model.bin
    - TensorFlow: .ckpt
    - ONNX: .onnx for cross‑framework portability
    - These formats are optimized for fast GPU/CPU loading.
    - Tokenizer vocabulary:
    - Stored separately in vocab.json (token → ID mapping) and sometimes merges.txt (for BPE rules).
    - Configuration:
    - A small config.json file describing architecture (layers, hidden size, attention heads).

# Training a Small GPT
    Training is the most resource‑intensive part.
    - Model size:
    A “small” GPT usually means ~100M–500M parameters (compared to billions in large GPTs).
    - Storage for dataset:
    - Text corpus: ~10–50 GB is enough for a small prototype.
    - Preprocessed tokenized data: slightly smaller, but still tens of GB.
    - RAM (system memory):
    - 32 GB recommended (16 GB minimum).
    - Needed for data loading, preprocessing, and batching.
    - GPU:
    - At least one NVIDIA GPU with 12–24 GB VRAM (e.g., RTX 3090, A6000, or similar).
    - Training speed scales with more GPUs, but for small models, a single high‑end GPU is enough.
    - If you only have 8 GB VRAM (e.g., RTX 3060), you’ll need smaller batch sizes and gradient checkpointing.


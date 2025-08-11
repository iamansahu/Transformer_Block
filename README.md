# Transformer Block Implementation

## Overview
This project implements a **Transformer block** from scratch in PyTorch, inspired by the architecture used in GPT models.  
It includes **multi-head self-attention**, **layer normalization**, **residual connections**, and a **feed-forward network**, closely following the design principles of the original Transformer paper *"Attention is All You Need"*.

The goal is to provide a clean, educational implementation that shows the internals of how a Transformer layer processes token embeddings.

---

## Features
- **Custom Layer Normalization** (no reliance on `nn.LayerNorm` to show internal calculations)
- **Multi-Head Self-Attention** with:
  - Learnable Q, K, V projection matrices
  - Scaled dot-product attention
  - Causal masking to prevent information leakage from future tokens
- **Feed Forward Network (FFN)** with **GELU** activation
- **Residual Connections** & **Dropout** for training stability
- **Configurable Hyperparameters** (e.g., vocab size, embedding dimension, number of heads, dropout)


---

##  Configuration
Example configuration for a **GPT-124M style block**:
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Max sequence length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of Transformer blocks
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Whether to add bias to Q/K/V projections
}

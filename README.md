# moe
Mixture of Experts implementation for LLMs

This repo is designed to be a simple and easy to understand implementation of the Mixture of Experts (MoE) architecture for LLMs.

Implementation is based on DeepSeek's v3 MoE architecture.

## Usage

```python
from moe import MoE

moe = MoE(
    vocab_size=30000,
    hidden_size=768,
    num_experts=12,
    num_tokens=1024,
)
```

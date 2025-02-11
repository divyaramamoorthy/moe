# MoE (Mixture of Experts)

A PyTorch implementation of the Mixture of Experts (MoE) architecture for Large Language Models, based on DeepSeek's v3 MoE architecture.

## Overview

This repository provides a PyTorch implementation of the Mixture of Experts (MoE) architecture. The implementation includes:

- Core MoE components (Expert networks and Gating mechanism)
- Load balancing with auxiliary loss
- Visualization tools for expert specialization
- Synthetic data generation for testing

## Architecture Details

The implementation includes three main components:

1. **Expert Networks**: Gated Linear Units (GLU) with SiLU activation
2. **Gating Mechanism**: Top-k routing with auxiliary loss for load balancing
3. **Classifier Head**: PyTorch Lightning wrapper for classification tasks

## Visualization Tools

The package includes tools for visualizing:
- Data distributions using PCA and t-SNE
- Expert specialization patterns
- Training metrics
- Load balancing effectiveness

## Installation

```bash
# Clone and install
git clone https://github.com/divyaramamoorthy/moe.git
cd moe
conda env create -f environment.yml
conda activate pytorch
pip install -e .
```

## Usage

```python
from moe import MoE

# Basic PyTorch MoE model
moe = MoE(
    dim=768,               # Input/output dimension for the model
    n_experts=12,          # Total number of expert networks
    n_activated_experts=2, # Number of experts that process each input
    moe_inter_dim=3072    # Hidden dimension size within each expert
)

# Classification PyTorch Lightning wrapper
from moe import MoEClassifier
classifier = MoEClassifier(
    dim=768,               # Input/output dimension for the model
    n_experts=12,          # Total number of expert networks
    n_activated_experts=2, # Number of experts that process each input
    moe_inter_dim=3072,   # Hidden dimension size within each expert
    learning_rate=1e-3    # Learning rate for optimizer
)
```

## License

MIT License - see LICENSE file for details.
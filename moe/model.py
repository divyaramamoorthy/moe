import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.  
    It looks like Deepseek modeled this off of Meta's Llama code: https://github.com/meta-llama/llama/blob/main/llama/model.py
    It is a gated linear unit (GLU) that uses a SiLu (instead of sigmoid) activation function.
    Orig GLU paper: https://arxiv.org/pdf/2002.05202

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer. 

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Gate(nn.Module):
    def __init__(self, dim: int, n_experts: int, topk_experts: int):
        """
        Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

        Args:
            dim (int): Input dimension
            n_experts (int): Number of experts
            topk_experts (int): Number of experts activated for each input
            temperature (float): Temperature parameter for softmax (default: 1.0)
        """
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.topk_experts = topk_experts
        self.linear = nn.Linear(dim, n_experts)
        # Initialize weights to small random values around 1/n_experts
        # nn.init.normal_(self.linear.weight, mean=1/self.n_experts, std=0.02)
        nn.init.normal_(self.linear.weight, mean=0.0, std=1)
        nn.init.zeros_(self.linear.bias)
        assert self.n_experts > 1, 'Number of experts must be greater than 1'

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.linear(x)
        scores = F.softmax(scores, dim=-1)
        indices = torch.topk(scores, self.topk_experts, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        return weights, indices


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_experts (int): Total number of experts in the model.
        n_activated_experts (int): Number of experts activated for each input.
        experts (nn.ModuleList): List of expert modules.
    """
    def __init__(self, dim, n_experts, n_activated_experts, moe_inter_dim):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim
        self.experts = nn.ModuleList([Expert(dim, moe_inter_dim) for _ in range(n_experts)])
        self.gate = Gate(dim=dim, n_experts=n_experts, topk_experts=n_activated_experts)
        assert n_activated_experts <= n_experts, 'Number of activated experts must be less than or equal to the number of experts'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        # Counts the number of tokens sent to each expert
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).tolist()
        for i in range(self.n_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            # Sum the outputs of experts for each token, scaled by the expert's weight
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        return y
    

class MoEClassifier(L.LightningModule):
    """
    PyTorch Lightning module for training the MoE model, with auxiliary-loss load balancing
    
    Args:
        dim (int): Input dimension
        n_experts (int): Number of experts
        n_activated_experts (int): Number of experts to activate per input
        moe_inter_dim (int): Internal dimension for expert networks
        learning_rate (float): Learning rate for optimization
    """
    def __init__(self, 
                 dim: int = 10, 
                 n_experts: int = 3, 
                 n_activated_experts: int = 2, 
                 moe_inter_dim: int = 20, 
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.moe = MoE(dim=dim, 
                      n_experts=n_experts, 
                      n_activated_experts=n_activated_experts,
                      moe_inter_dim=moe_inter_dim)
        self.classifier_head = nn.Sequential(
            nn.Linear(dim, n_experts),
            nn.LogSoftmax(dim=1)
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.classifier_head(self.moe(x))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)

        # Add load balancing loss
        _, indices = self.moe.gate(x)
        # Calculate fraction of inputs going to each expert
        expert_counts = torch.bincount(indices.flatten(), minlength=self.moe.n_experts)
        expert_fractions = expert_counts.float() / indices.numel()
        # Ideal fraction is uniform distribution
        ideal_fraction = 1.0 / self.moe.n_experts
        # L2 loss between actual and ideal fractions
        balance_loss = torch.mean((expert_fractions - ideal_fraction) ** 2)
        
        total_loss = loss + 0.01 * balance_loss  # Scale factor for balance loss
        self.log('train_loss', loss)
        self.log('balance_loss', balance_loss)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
    
    def get_expert_assignments(self, dataloader):
        """Collect expert assignments for all data in the dataloader
        
        Args:
            dataloader (DataLoader): DataLoader containing the dataset
            
        Returns:
            tuple: (input_data, expert_indices) as numpy arrays
        """
        self.eval()
        assignments = []
        with torch.no_grad():
            for x, _ in dataloader:
                # Get weights and indices from gate
                _, indices = self.moe.gate(x)
                assignments.append(indices.numpy())
        return np.concatenate(assignments, axis=0)
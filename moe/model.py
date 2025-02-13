import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from lightning.pytorch import LightningModule
from torch.optim import Adam


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.  
    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        dropout (float): Dropout rate.
    """
    def __init__(self, dim: int, inter_dim: int, dropout: float = 0.1):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer. 

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.dropout(self.w2(F.relu(self.w1(x))))

class Gate(nn.Module):
    def __init__(self, dim: int, n_experts: int, topk_experts: int, route_scale: float = 1.0):
        """
        Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

        Args:
            dim (int): Input dimension
            n_experts (int): Number of experts
            topk_experts (int): Number of experts activated for each input, 
            # bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.

        """
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.topk_experts = topk_experts
        self.weight = nn.Parameter(torch.randn(n_experts, dim))
        # self.bias = nn.Parameter(torch.zeros(n_experts))
        self.route_scale = route_scale
        assert self.n_experts > 1, 'Number of experts must be greater than 1'

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores =  F.linear(x, self.weight)
        scores = scores.sigmoid()
        original_scores = scores
        # # add bias as load balancing - should only be used to influence indices
        # biased_scores = scores + self.bias
        # indices = torch.topk(biased_scores, self.topk_experts, dim=-1)[1]
        indices = torch.topk(scores, self.topk_experts, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
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
    def __init__(self, dim, n_experts, n_activated_experts, moe_inter_dim, route_scale: float = 1.0, 
                #  bias_update_speed: float = 0.01
                 ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim
        # self.bias_update_speed = bias_update_speed
        self.experts = nn.ModuleList([Expert(dim, moe_inter_dim) for _ in range(n_experts)])
        self.gate = Gate(dim=dim, n_experts=n_experts, topk_experts=n_activated_experts, route_scale=route_scale)
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
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts)
        for i in range(self.n_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            # Sum the outputs of experts for each token, scaled by the expert's weight
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        # # Update the bias of the gate
        # ideal_load = (x.shape[0]*self.n_activated_experts) / self.n_experts
        # if self.bias_update_speed is not None:
        #     with torch.no_grad():  
        #         self.gate.bias.data += (ideal_load - counts) * self.bias_update_speed
        return y
    

class MoEClassifier(LightningModule):
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
                 learning_rate: float = 1e-3, 
                 route_scale: float = 1.0,
                #  bias_update_speed: float = 0.01,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.moe = MoE(dim=dim, 
                      n_experts=n_experts, 
                      n_activated_experts=n_activated_experts,
                      moe_inter_dim=moe_inter_dim,
                      route_scale=route_scale,
                    #   bias_update_speed=bias_update_speed
                      )
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
        self.log('train_loss', loss)
        return loss
    
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
                assignments.append(indices)
        return torch.cat(assignments, dim=0)
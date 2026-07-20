"""
Module 5: Cross-Modal Contrastive Alignment (COSMOS Logic)

Core Purpose:
    Align transcriptomic (Z_RNA) and proteomic (Z_ADT) embeddings into a shared
    coordinate space through symmetric InfoNCE contrastive learning.

Key Architecture:
    • Symmetric InfoNCE Loss: Pulls same-spot RNA/ADT pairs together, pushes
      different-spot pairs apart (bidirectional: RNA→ADT and ADT→RNA)
    • Cosine Similarity: Computes pairwise similarity between modalities
    • Temperature Scaling: Softens/sharpens probability distributions
    • Shared Embedding Space: Both modalities exist in identical (d=512) space

Inputs:
    • Z_RNA ∈ R^(3484 × 512) - Spatially-informed RNA embeddings from Module 4
    • Z_ADT ∈ R^(3484 × 512) - Spatially-informed ADT embeddings from Module 4

Outputs:
    • Aligned Z_RNA, Z_ADT - Same tensors, now synchronized via loss
    • L_cl - Contrastive loss scalar for total training objective

Mathematical Foundation:
    The contrastive loss follows the Information Noise-Contrastive Estimation
    (InfoNCE) framework, which maximizes the mutual information between modalities:

    L_cl = -1/(2N) * Σ_i [
        log(exp(sim(Z_RNA_i, Z_ADT_i)/τ) / Σ_j exp(sim(Z_RNA_i, Z_ADT_j)/τ))
      + log(exp(sim(Z_ADT_i, Z_RNA_i)/τ) / Σ_j exp(sim(Z_ADT_i, Z_RNA_j)/τ))
    ]

    Where:
    - sim(u,v) = u·v^T / (||u|| ||v||) - Cosine similarity
    - τ - Temperature parameter (default: 0.1)
    - N - Batch size (number of spots)
    - Symmetry ensures bidirectional alignment

References:
    • COSMOS: "...InfoNCE loss with symmetric formulation"
    • module_explanation.md: Complete mathematical specification
    • flow.md: Algorithm, inputs, outputs, mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Contrastive Loss for Cross-Modal Alignment.

    Computes bidirectional contrastive loss between two modalities (RNA and ADT)
    to maximize their mutual information in a shared embedding space.

    Mathematical Formulation:
        L_cl = -1/(2N) * Σ_i [
            log(exp(sim(Z_RNA_i, Z_ADT_i)/τ) / Σ_j exp(sim(Z_RNA_i, Z_ADT_j)/τ))
          + log(exp(sim(Z_ADT_i, Z_RNA_i)/τ) / Σ_j exp(sim(Z_ADT_i, Z_RNA_j)/τ))
        ]

    Args:
        temperature (float): Softens the similarity scores. Default: 0.1.
            Lower τ → sharper probability distribution
            Higher τ → softer probability distribution
        reduction (str): Reduction method ('mean' or 'sum'). Default: 'mean'.

    Attributes:
        temperature: Temperature parameter τ
        reduction: How to aggregate loss across batch

    Forward Input:
        z_rna: Tensor of shape (N, d) - RNA embeddings from Module 4
        z_adt: Tensor of shape (N, d) - ADT embeddings from Module 4

    Returns:
        loss: Scalar tensor - Contrastive loss value L_cl

    Example:
        >>> criterion = InfoNCELoss(temperature=0.1)
        >>> z_rna = torch.randn(3484, 512)
        >>> z_adt = torch.randn(3484, 512)
        >>> loss = criterion(z_rna, z_adt)
        >>> print(loss.item())  # Scalar loss value
    """

    def __init__(self, temperature=0.1, reduction='mean'):
        """Initialize InfoNCE loss with temperature parameter."""
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z_rna, z_adt):
        """
        Compute symmetric InfoNCE loss between RNA and ADT embeddings.

        Args:
            z_rna (torch.Tensor): Shape (N, d) - Spatially-informed RNA embeddings
            z_adt (torch.Tensor): Shape (N, d) - Spatially-informed ADT embeddings

        Returns:
            loss (torch.Tensor): Scalar contrastive loss
        """
        # Normalize embeddings for cosine similarity
        # L2 normalization: u → u / ||u||
        z_rna_norm = F.normalize(z_rna, p=2, dim=1)  # (N, d)
        z_adt_norm = F.normalize(z_adt, p=2, dim=1)  # (N, d)

        # Compute pairwise similarity matrix: sim = Z @ Z^T
        # sim_matrix[i,j] = cosine_similarity(z_rna_i, z_adt_j)
        sim_matrix_rna_to_adt = torch.mm(z_rna_norm, z_adt_norm.T)  # (N, N)
        sim_matrix_adt_to_rna = torch.mm(z_adt_norm, z_rna_norm.T)  # (N, N)

        # Scale by temperature: sim / τ
        # This controls the sharpness of the softmax distribution
        sim_matrix_rna_to_adt = sim_matrix_rna_to_adt / self.temperature  # (N, N)
        sim_matrix_adt_to_rna = sim_matrix_adt_to_rna / self.temperature  # (N, N)

        # Create positive labels: diagonal entries (same-spot pairs)
        # labels[i] = i means the positive pair for spot i is at position i
        batch_size = z_rna.shape[0]
        labels = torch.arange(batch_size, device=z_rna.device)  # (N,)

        # Compute InfoNCE loss for RNA → ADT direction
        # log(exp(sim_ii) / Σ_j exp(sim_ij))
        loss_rna_to_adt = F.cross_entropy(sim_matrix_rna_to_adt, labels)

        # Compute InfoNCE loss for ADT → RNA direction
        # log(exp(sim_ii) / Σ_j exp(sim_ij))
        loss_adt_to_rna = F.cross_entropy(sim_matrix_adt_to_rna, labels)

        # Symmetric loss: average of both directions
        # L_cl = 1/2 * (L_RNA→ADT + L_ADT→RNA)
        loss = 0.5 * (loss_rna_to_adt + loss_adt_to_rna)

        return loss

    def __repr__(self):
        """String representation of the loss module."""
        return (
            f"{self.__class__.__name__}(temperature={self.temperature}, "
            f"reduction='{self.reduction}')"
        )


class ContrastiveAlignmentModule(nn.Module):
    """
    Cross-Modal Contrastive Alignment (COSMOS Logic).

    Aligns transcriptomic and proteomic embeddings by maximizing their mutual
    information through symmetric InfoNCE loss. Produces synchronized embeddings
    in a shared space that can be directly fused in Module 6.

    Core Mechanism:
        1. Takes Z_RNA and Z_ADT from Module 4 (both in 512-dim space)
        2. Computes pairwise cosine similarities between all spots
        3. Applies symmetric InfoNCE loss:
           - RNA→ADT: Pulls each spot's RNA close to its own ADT, away from others
           - ADT→RNA: Pulls each spot's ADT close to its own RNA, away from others
        4. Returns contrastive loss for backpropagation through Module 4
        5. Embeddings themselves are NOT modified (alignment via loss)

    Architecture:
        • InfoNCE Loss Computation (no learnable parameters)
        • Similarity Matrix Generation
        • Symmetric bidirectional loss averaging
        • Temperature-scaled softmax for probability calibration

    Inputs:
        z_rna: RNA embeddings (3484, 512) from Module 4
        z_adt: ADT embeddings (3484, 512) from Module 4

    Outputs:
        loss_cl: Scalar contrastive loss for total training objective

    Attributes:
        loss_fn (InfoNCELoss): Contrastive loss function
        temperature (float): Temperature scaling parameter

    Example:
        >>> module = ContrastiveAlignmentModule(temperature=0.1)
        >>> z_rna = torch.randn(3484, 512)
        >>> z_adt = torch.randn(3484, 512)
        >>> loss = module(z_rna, z_adt)
        >>> print(loss.item())  # Scalar loss value
    """

    def __init__(self, temperature=0.1):
        """
        Initialize Contrastive Alignment Module.

        Args:
            temperature (float): Temperature parameter τ for scaling similarities.
                Default: 0.1 (standard for contrastive learning)
        """
        super(ContrastiveAlignmentModule, self).__init__()
        self.temperature = temperature
        self.loss_fn = InfoNCELoss(temperature=temperature)

    def forward(self, z_rna, z_adt, return_similarity=False):
        """
        Compute contrastive alignment loss for RNA and ADT embeddings.

        Args:
            z_rna (torch.Tensor): RNA embeddings, shape (N, d=512) from Module 4
            z_adt (torch.Tensor): ADT embeddings, shape (N, d=512) from Module 4
            return_similarity (bool): If True, also return similarity matrices.
                Default: False

        Returns:
            loss_cl (torch.Tensor): Scalar contrastive loss
            or (loss_cl, sim_rna_to_adt, sim_adt_to_rna) if return_similarity=True

        Forward Pass Logic:
            1. Normalize embeddings for cosine similarity (handles scale invariance)
            2. Compute pairwise similarity: RNA @ ADT^T
            3. Apply InfoNCE loss with positive (diagonal) and negative pairs
            4. Return symmetric loss: 1/2 * (L_RNA→ADT + L_ADT→RNA)
        """
        # Compute contrastive loss
        # This internally normalizes embeddings and applies InfoNCE
        loss_cl = self.loss_fn(z_rna, z_adt)

        if return_similarity:
            # Optionally return similarity matrices for inspection
            # Normalize for cosine similarity
            z_rna_norm = F.normalize(z_rna, p=2, dim=1)
            z_adt_norm = F.normalize(z_adt, p=2, dim=1)
            sim_rna_to_adt = torch.mm(z_rna_norm, z_adt_norm.T)
            sim_adt_to_rna = torch.mm(z_adt_norm, z_rna_norm.T)
            return loss_cl, sim_rna_to_adt, sim_adt_to_rna

        return loss_cl

    def compute_alignment_quality(self, z_rna, z_adt):
        """
        Compute alignment quality metrics (for monitoring/debugging).

        Metrics:
            - positive_similarity: Average cosine similarity of same-spot pairs
            - negative_similarity: Average cosine similarity of different-spot pairs
            - alignment_ratio: positive_sim / (negative_sim + ε)

        Args:
            z_rna (torch.Tensor): RNA embeddings (N, d)
            z_adt (torch.Tensor): ADT embeddings (N, d)

        Returns:
            metrics (dict): Dictionary with quality metrics
        """
        # Normalize and compute similarity matrix
        z_rna_norm = F.normalize(z_rna, p=2, dim=1)
        z_adt_norm = F.normalize(z_adt, p=2, dim=1)
        sim_rna_to_adt = torch.mm(z_rna_norm, z_adt_norm.T)

        # Get positive mask (diagonal) and negative mask (off-diagonal)
        batch_size = z_rna.shape[0]
        device = z_rna.device
        pos_mask = torch.eye(batch_size, device=device).bool()
        neg_mask = ~pos_mask

        # Compute average similarities
        positive_sim = sim_rna_to_adt[pos_mask].mean().item()
        negative_sim = sim_rna_to_adt[neg_mask].mean().item()
        alignment_ratio = positive_sim / (negative_sim + 1e-8)

        metrics = {
            'positive_similarity': positive_sim,
            'negative_similarity': negative_sim,
            'alignment_ratio': alignment_ratio,
        }

        return metrics

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}(temperature={self.temperature})\n"
            f"  └─ InfoNCELoss: {self.loss_fn}"
        )


def create_contrastive_module(temperature=0.1, device='cpu'):
    """
    Factory function to create and configure ContrastiveAlignmentModule.

    Simplifies initialization with KAC-Net defaults and device management.

    Args:
        temperature (float): Temperature parameter τ. Default: 0.1
        device (str or torch.device): Device placement ('cpu' or 'cuda'). Default: 'cpu'

    Returns:
        module (ContrastiveAlignmentModule): Initialized and device-placed module

    Example:
        >>> module = create_contrastive_module(temperature=0.1, device='cuda')
        >>> print(module)
    """
    module = ContrastiveAlignmentModule(temperature=temperature)
    module = module.to(device)
    return module


# ============================================================================
# Integration with Total Training Loss
# ============================================================================

def compute_total_loss(loss_cl, loss_recon, loss_spat, lambda_cl=1.0, lambda_recon=1.0, lambda_spat=1.0):
    """
    Compute total training loss across all modules.

    Formula:
        L_total = λ_1 * L_cl + λ_2 * L_recon + λ_3 * L_spat

    Where:
        - L_cl: Contrastive alignment loss (Module 5)
        - L_recon: Reconstruction loss (Module 7)
        - L_spat: Spatial regularization loss (Module 7)
        - λ_1, λ_2, λ_3: Balancing hyperparameters

    Args:
        loss_cl (torch.Tensor): Contrastive loss from Module 5
        loss_recon (torch.Tensor): Reconstruction loss from Module 7
        loss_spat (torch.Tensor): Spatial loss from Module 7
        lambda_cl (float): Weight for contrastive loss. Default: 1.0
        lambda_recon (float): Weight for reconstruction loss. Default: 1.0
        lambda_spat (float): Weight for spatial loss. Default: 1.0

    Returns:
        loss_total (torch.Tensor): Weighted sum of all losses
    """
    loss_total = lambda_cl * loss_cl + lambda_recon * loss_recon + lambda_spat * loss_spat
    return loss_total

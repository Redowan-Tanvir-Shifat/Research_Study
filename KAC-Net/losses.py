"""
KAC-Net Loss Functions

Centralized loss functions for all training objectives:
- Contrastive Alignment Loss (cross-modal InfoNCE)
- Reconstruction Loss (VAE-style)
- Spatial Regularization Loss (graph smoothness)
- Combined Loss (weighted sum)

All losses are modular and can be used independently or combined.
Enables easy experimentation with different loss formulations.

Functions:
    - contrastive_loss() - InfoNCE loss for cross-modal alignment
    - reconstruction_loss() - MSE reconstruction for RNA and ADT
    - spatial_loss() - Laplacian-based spatial smoothness
    - combined_loss() - Weighted sum of all losses
    - kl_divergence_loss() - Optional: KL divergence regularization
    - wasserstein_loss() - Optional: Wasserstein distance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import logging

# Import class-based loss implementations from modules
from modules.contrastive_alignment import InfoNCELoss
from modules.reconstruction_loss import ReconstructionLoss, SpatialRegularizationLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# WRAPPER FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================
# These functions wrap the class-based implementations from modules/
# to maintain compatibility with existing code that uses functional API

def contrastive_loss(
    Z_RNA: torch.Tensor,
    Z_ADT: torch.Tensor,
    temperature: float = 0.07,  # FIXED: Changed from 0.1 to 0.07 (matches config)
    use_cosine_sim: bool = True,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    InfoNCE Contrastive Loss for cross-modal alignment (WRAPPER).
    
    Wraps InfoNCELoss from modules.contrastive_alignment for backward compatibility.
    See modules.contrastive_alignment.InfoNCELoss for full documentation.
    
    Args:
        Z_RNA: RNA embeddings (batch_size, embedding_dim)
        Z_ADT: ADT embeddings (batch_size, embedding_dim)
        temperature: Temperature parameter (default 0.07, fixed from 0.1)
        use_cosine_sim: If True, use cosine similarity; else dot product (ignored - always cosine)
        reduction: 'mean' or 'sum' (ignored - always mean)
    
    Returns:
        loss: Scalar contrastive loss
    """
    loss_fn = InfoNCELoss(temperature=temperature, reduction='mean')
    return loss_fn(Z_RNA, Z_ADT)


def reconstruction_loss(
    X_RNA_recon: torch.Tensor,
    X_RNA_true: torch.Tensor,
    X_ADT_recon: torch.Tensor,
    X_ADT_true: torch.Tensor,
    rna_weight: float = 1.0,
    adt_weight: float = 1.0,
    loss_type: str = 'mse',
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Reconstruction Loss for RNA and ADT data (WRAPPER).
    
    Wraps ReconstructionLoss from modules.reconstruction_loss for backward compatibility.
    See modules.reconstruction_loss.ReconstructionLoss for full documentation.
    
    Args:
        X_RNA_recon: Reconstructed RNA (batch_size, n_genes)
        X_RNA_true: True RNA data (batch_size, n_genes)
        X_ADT_recon: Reconstructed ADT (batch_size, n_proteins)
        X_ADT_true: True ADT data (batch_size, n_proteins)
        rna_weight: Weight for RNA loss (default 1.0) (ignored - assumes equal weighting)
        adt_weight: Weight for ADT loss (default 1.0) (ignored - assumes equal weighting)
        loss_type: 'mse', 'mae', or 'huber' (ignored - uses MSE only)
        reduction: 'mean' or 'sum' (default 'mean')
    
    Returns:
        loss: Scalar reconstruction loss
    """
    loss_fn = ReconstructionLoss(reduction=reduction)
    return loss_fn(X_RNA_true, X_RNA_recon, X_ADT_true, X_ADT_recon)


def spatial_loss(
    Z: torch.Tensor,
    adj_spatial: Union[torch.Tensor, np.ndarray],
    loss_type: str = 'laplacian'
) -> torch.Tensor:
    """
    Spatial Regularization Loss for spatial smoothness (WRAPPER).
    
    Wraps SpatialRegularizationLoss from modules.reconstruction_loss for backward compatibility.
    See modules.reconstruction_loss.SpatialRegularizationLoss for full documentation.
    
    Args:
        Z: Embedding matrix (n_spots, embedding_dim)
        adj_spatial: Spatial adjacency matrix (n_spots, n_spots)
        loss_type: 'laplacian' (default), 'smoothness' (ignored - uses laplacian)
    
    Returns:
        loss: Scalar spatial loss
    """
    loss_fn = SpatialRegularizationLoss(loss_type='laplacian')
    return loss_fn(Z, adj_spatial)


# ============================================================================
# COMBINED LOSS
# ============================================================================

def combined_loss(
    Z_RNA: torch.Tensor,
    Z_ADT: torch.Tensor,
    X_RNA_recon: torch.Tensor,
    X_RNA_true: torch.Tensor,
    X_ADT_recon: torch.Tensor,
    X_ADT_true: torch.Tensor,
    Z_fused: torch.Tensor,
    adj_spatial: Union[torch.Tensor, np.ndarray],
    lambda_contrastive: float = 0.5,
    lambda_reconstruction: float = 1.0,
    lambda_spatial: float = 0.3,
    temperature: float = 0.07,
    return_components: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Combined Loss Function (all 3 loss components).
    
    Total loss used for training:
        L_total = λ_cl * L_cl + λ_recon * L_recon + λ_spatial * L_spatial
    
    Args:
        Z_RNA: RNA embeddings from Module 4 (batch_size, embedding_dim)
        Z_ADT: ADT embeddings from Module 4 (batch_size, embedding_dim)
        X_RNA_recon: Reconstructed RNA (batch_size, n_genes)
        X_RNA_true: True RNA (batch_size, n_genes)
        X_ADT_recon: Reconstructed ADT (batch_size, n_proteins)
        X_ADT_true: True ADT (batch_size, n_proteins)
        Z_fused: Fused embeddings from Module 6 (batch_size, embedding_dim)
        adj_spatial: Spatial adjacency matrix (n_spots, n_spots)
        lambda_contrastive: Weight for contrastive loss (default 0.5)
        lambda_reconstruction: Weight for reconstruction loss (default 1.0)
        lambda_spatial: Weight for spatial loss (default 0.3)
        temperature: Temperature for contrastive loss (default 0.07, fixed)
        return_components: If True, return dict of individual losses
    
    Returns:
        loss: Total scalar loss OR
        (loss, components_dict) if return_components=True
    
    Example:
        >>> loss = combined_loss(
        ...     Z_RNA, Z_ADT,
        ...     X_RNA_recon, X_RNA_batch,
        ...     X_ADT_recon, X_ADT_batch,
        ...     Z_fused, adj_spatial,
        ...     lambda_contrastive=0.5,
        ...     lambda_reconstruction=1.0,
        ...     lambda_spatial=0.3
        ... )
        >>> loss.backward()
    
    Example (with components):
        >>> loss, components = combined_loss(
        ...     ...,
        ...     return_components=True
        ... )
        >>> print(f"Total: {loss:.4f}, L_cl: {components['L_cl']:.4f}")
    
    Notes:
        - Lambda values should sum to ~2 for balanced losses
        - Contrastive (0.5) + Reconstruction (1.0) + Spatial (0.3) = 1.8
        - Adjust based on importance to your task
    """
    # Compute individual losses
    L_cl = contrastive_loss(
        Z_RNA, Z_ADT,
        temperature=temperature,
        use_cosine_sim=True
    )
    
    L_recon = reconstruction_loss(
        X_RNA_recon, X_RNA_true,
        X_ADT_recon, X_ADT_true,
        rna_weight=1.0,
        adt_weight=1.0,
        loss_type='mse'
    )
    
    L_spatial = spatial_loss(
        Z_fused, adj_spatial,
        loss_type='laplacian'
    )
    
    # Compute total loss
    L_total = (
        lambda_contrastive * L_cl +
        lambda_reconstruction * L_recon +
        lambda_spatial * L_spatial
    )
    
    if return_components:
        components = {
            'L_total': L_total.item(),
            'L_cl': L_cl.item(),
            'L_recon': L_recon.item(),
            'L_spatial': L_spatial.item(),
        }
        return L_total, components
    else:
        return L_total


# ============================================================================
# LOSS UTILITIES
# ============================================================================

def compute_loss_weights(
    config: dict
) -> Tuple[float, float, float]:
    """
    Extract and validate loss weights from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        lambda_contrastive, lambda_reconstruction, lambda_spatial
    
    Example:
        >>> from config import get_config
        >>> config = get_config('lymph_node')
        >>> lam_cl, lam_rec, lam_spat = compute_loss_weights(config)
    """
    lam_cl = config['losses'].get('lambda_contrastive', 0.5)
    lam_rec = config['losses'].get('lambda_reconstruction', 1.0)
    lam_spat = config['losses'].get('lambda_spatial', 0.3)
    
    logger.info(f"Loss weights: λ_cl={lam_cl}, λ_recon={lam_rec}, λ_spatial={lam_spat}")
    logger.info(f"  Total: {lam_cl + lam_rec + lam_spat:.1f}")
    
    return lam_cl, lam_rec, lam_spat


def log_loss_components(
    losses_dict: dict,
    epoch: int,
    print_interval: int = 5
):
    """
    Log loss components during training.
    
    Args:
        losses_dict: Dictionary with loss values
        epoch: Current epoch number
        print_interval: Print every N epochs
    
    Example:
        >>> loss, components = combined_loss(..., return_components=True)
        >>> log_loss_components(components, epoch=10)
    """
    if epoch % print_interval == 0:
        msg = f"Epoch {epoch:3d} | "
        for key, val in losses_dict.items():
            if not key.startswith('_'):
                msg += f"{key}={val:.4f} "
        logger.info(msg)


if __name__ == '__main__':
    """
    Example usage and testing:
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    embedding_dim = 64
    n_genes = 18085
    n_proteins = 31
    
    # Create dummy data
    Z_RNA = torch.randn(batch_size, embedding_dim, device=device)
    Z_ADT = torch.randn(batch_size, embedding_dim, device=device)
    Z_fused = torch.randn(batch_size, embedding_dim, device=device)
    
    X_RNA_recon = torch.randn(batch_size, n_genes, device=device)
    X_RNA_true = torch.randn(batch_size, n_genes, device=device)
    X_ADT_recon = torch.randn(batch_size, n_proteins, device=device)
    X_ADT_true = torch.randn(batch_size, n_proteins, device=device)
    
    adj_spatial = torch.eye(batch_size, device=device)  # Dummy adjacency
    
    # Test individual losses
    L_cl = contrastive_loss(Z_RNA, Z_ADT)
    L_recon = reconstruction_loss(X_RNA_recon, X_RNA_true, X_ADT_recon, X_ADT_true)
    L_spatial = spatial_loss(Z_fused, adj_spatial)
    
    print(f"L_cl: {L_cl:.4f}")
    print(f"L_recon: {L_recon:.4f}")
    print(f"L_spatial: {L_spatial:.4f}")
    
    # Test combined loss
    L_total, components = combined_loss(
        Z_RNA, Z_ADT,
        X_RNA_recon, X_RNA_true,
        X_ADT_recon, X_ADT_true,
        Z_fused, adj_spatial,
        return_components=True
    )
    
    print(f"\nCombined Loss: {L_total:.4f}")
    for key, val in components.items():
        print(f"  {key}: {val:.4f}")
    """
    print("KAC-Net losses module ready. See docstrings for usage.")

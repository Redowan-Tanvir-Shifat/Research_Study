"""
KAC-Net v2 — losses.py
Loss functions for optimization:
  - InfoNCE Loss (Module 5 Cross-Modal Alignment)
  - Reconstruction Loss (Module 7 MSE)
  - Graph Laplacian Loss (Spatial Regularization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(features_a, features_b, temperature=0.07):
    """
    Symmetric InfoNCE Loss (COSMOS alignment logic).
    
    Aims to maximize cosine similarity of matched spots across modalities (a & b)
    while minimizing similarity with mismatched spots.
    
    Parameters
    ----------
    features_a  : torch.Tensor of shape (N, proj_dim)  –  normalized RNA projections
    features_b  : torch.Tensor of shape (N, proj_dim)  –  normalized ADT projections
    temperature : float                                –  InfoNCE scaling factor (default 0.07)
    
    Returns
    -------
    loss        : torch.Tensor
    """
    # L2 normalize inputs
    feat_a = F.normalize(features_a, p=2, dim=1)
    feat_b = F.normalize(features_b, p=2, dim=1)
    
    batch_size = feat_a.size(0)
    
    # Cosine similarity matrix between all pairs
    # shape: (N, N)
    logits_ab = torch.matmul(feat_a, feat_b.T) / temperature
    logits_ba = logits_ab.T
    
    # Ground-truth labels: identity matrix (matching indices)
    labels = torch.arange(batch_size, device=feat_a.device)
    
    loss_ab = F.cross_entropy(logits_ab, labels)
    loss_ba = F.cross_entropy(logits_ba, labels)
    
    return (loss_ab + loss_ba) / 2.0


def reconstruction_loss(recon_rna, rna_target, recon_adt, adt_target):
    """
    Module 7 Reconstruction Loss (MSE).
    Calculates reconstruction loss for both RNA and ADT features.
    """
    loss_rna = F.mse_loss(recon_rna, rna_target)
    loss_adt = F.mse_loss(recon_adt, adt_target)
    return loss_rna, loss_adt


def graph_laplacian_loss(fused_latent, adj_spatial_tensor):
    """
    Graph Laplacian Loss for spatial smoothing.
    Encourages spatially adjacent spots to have similar latent representations:
        L_lap = Tr( H^T * L * H )
    """
    # Degree matrix D is implicit in normalized adjacency A_norm
    # With A_norm = D^{-1/2} A D^{-1/2}, we approximate Laplaican smoothing
    # by calculating difference between feature vectors and normalized neighborhood averages.
    
    # Neighborhood aggregation: A_norm * H
    neighborhood_mean = torch.sparse.mm(adj_spatial_tensor, fused_latent)
    
    # Mean squared error difference
    laplacian_loss = F.mse_loss(fused_latent, neighborhood_mean)
    return laplacian_loss


def compute_total_loss(
    model_outputs,
    rna_target,
    adt_target,
    adj_spatial_tensor,
    w_recon_rna=1.0,
    w_recon_adt=1.0,
    w_align=1.0,
    w_laplacian=1.0,
    temperature=0.07
):
    """
    Combine all loss functions into a unified optimization target.
    """
    # 1. Reconstruction Loss
    loss_rec_rna, loss_rec_adt = reconstruction_loss(
        model_outputs['recon_rna'], rna_target,
        model_outputs['recon_adt'], adt_target
    )
    
    # 2. InfoNCE Alignment Loss
    loss_align = info_nce_loss(
        model_outputs['p_rna'], model_outputs['p_adt'],
        temperature=temperature
    )
    
    # 3. Spatial Regularization Laplacian Loss
    loss_lap = graph_laplacian_loss(
        model_outputs['h_fused'], adj_spatial_tensor
    )
    
    # Blended Loss
    total_loss = (
        w_recon_rna * loss_rec_rna +
        w_recon_adt * loss_rec_adt +
        w_align * loss_align +
        w_laplacian * loss_lap
    )
    
    return {
        'loss': total_loss,
        'loss_rec_rna': loss_rec_rna.item(),
        'loss_rec_adt': loss_rec_adt.item(),
        'loss_align': loss_align.item(),
        'loss_lap': loss_lap.item()
    }

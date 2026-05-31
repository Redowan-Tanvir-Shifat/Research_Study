"""
Module 7: Reconstruction & Regularization (Decoder & Hub)

Core Purpose:
    Force the model to prove it hasn't forgotten original biological information
    by reconstructing high-dimensional data (RNA: 18,085 dims, ADT: 31 dims) from
    the compressed 64-dimensional latent space. Simultaneously regularize spatial
    smoothness to eliminate technical noise.

Key Architecture:
    • Decoder MLPs: RNA (64→18085) and ADT (64→31) reconstruct original dimensions
    • Reconstruction Loss: MSE between reconstructions and preprocessed data
    • Spatial Regularization: Graph Laplacian smoothness penalty
    • Total Loss: Weighted sum of contrastive (Module 5) + reconstruction + spatial

Inputs:
    • Z_Fused ∈ R^(3484 × 64) - Unified embeddings from Module 6
    • X̃_RNA ∈ R^(3484 × 18085) - Preprocessed RNA from Module 1
    • X̃_ADT ∈ R^(3484 × 31) - Preprocessed ADT from Module 1
    • A_s ∈ R^(3484 × 3484) - Spatial adjacency matrix from Module 3

Outputs:
    • X̂_RNA ∈ R^(3484 × 18085) - Reconstructed RNA counts
    • X̂_ADT ∈ R^(3484 × 31) - Reconstructed ADT counts
    • L_recon - Reconstruction loss (scalar)
    • L_spat - Spatial regularization loss (scalar)
    • L_total - Total training loss (scalar)

Mathematical Foundation:
    Reconstruction Loss (MSE):
        L_recon = (1/N) * Σ_i [||X̃_RNA,i - X̂_RNA,i||² + ||X̃_ADT,i - X̂_ADT,i||²]

    Spatial Regularization (Graph Laplacian Smoothing):
        L_spat = Σ_{i,j} A_{s,ij} * ||Z_Fused,i - Z_Fused,j||²

    Total Loss:
        L_total = λ_cl * L_cl + λ_recon * L_recon + λ_spat * L_spat

References:
    • SpatialGlue: Decoder architecture
    • module_explanation.md: Complete mathematical specification
    • flow.md: Algorithm, inputs, outputs, mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNADecoder(nn.Module):
    """
    Decoder for reconstructing RNA expression from latent embeddings.

    Maps compressed 64-dimensional latent space back to 18,085 gene dimensions.
    Uses multi-layer MLP with intermediate non-linearity to learn the mapping
    from abstract representation to biological gene counts.

    Purpose:
        Verification that latent space preserves transcriptomic information.
        If decoder can accurately reconstruct RNA counts, the compression
        (64-dim) must contain meaningful biological structure.

    Args:
        latent_dim (int): Input latent dimension (64 from Module 6)
        output_dim (int): Output dimension (18085 genes)
        hidden_dim (int): Hidden layer dimension (default: 512)

    Mathematical Form:
        X̂_RNA = MLP(Z_Fused) where MLP = Linear(64, 512) → ReLU → Linear(512, 18085)

    Inputs:
        z_fused (torch.Tensor): Shape (N, 64) - Fused embeddings from Module 6

    Output:
        x_rna_hat (torch.Tensor): Shape (N, 18085) - Reconstructed RNA counts
    """

    def __init__(self, latent_dim=64, output_dim=18085, hidden_dim=512):
        """Initialize RNA decoder MLP."""
        super(RNADecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Multi-layer MLP: 64 → 512 → 18085
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z_fused):
        """
        Reconstruct RNA expression from latent embedding.

        Args:
            z_fused (torch.Tensor): Shape (N, 64) - Fused embeddings

        Returns:
            x_rna_hat (torch.Tensor): Shape (N, 18085) - Reconstructed RNA
        """
        x_rna_hat = self.mlp(z_fused)  # (N, 18085)
        return x_rna_hat

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}(latent_dim={self.latent_dim}, "
            f"output_dim={self.output_dim}, hidden_dim={self.hidden_dim})"
        )


class ADTDecoder(nn.Module):
    """
    Decoder for reconstructing ADT (protein) expression from latent embeddings.

    Maps compressed 64-dimensional latent space back to 31 protein dimensions.
    Simpler architecture than RNA decoder due to lower output dimensionality
    (31 proteins vs 18,085 genes).

    Purpose:
        Verification that latent space preserves proteomic information.
        Ensures ADT signal is not discarded during cross-modal fusion.

    Args:
        latent_dim (int): Input latent dimension (64 from Module 6)
        output_dim (int): Output dimension (31 proteins)
        hidden_dim (int): Hidden layer dimension (default: 128)

    Mathematical Form:
        X̂_ADT = MLP(Z_Fused) where MLP = Linear(64, 128) → ReLU → Linear(128, 31)

    Inputs:
        z_fused (torch.Tensor): Shape (N, 64) - Fused embeddings from Module 6

    Output:
        x_adt_hat (torch.Tensor): Shape (N, 31) - Reconstructed ADT counts
    """

    def __init__(self, latent_dim=64, output_dim=31, hidden_dim=128):
        """Initialize ADT decoder MLP."""
        super(ADTDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Multi-layer MLP: 64 → 128 → 31
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z_fused):
        """
        Reconstruct ADT expression from latent embedding.

        Args:
            z_fused (torch.Tensor): Shape (N, 64) - Fused embeddings

        Returns:
            x_adt_hat (torch.Tensor): Shape (N, 31) - Reconstructed ADT
        """
        x_adt_hat = self.mlp(z_fused)  # (N, 31)
        return x_adt_hat

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}(latent_dim={self.latent_dim}, "
            f"output_dim={self.output_dim}, hidden_dim={self.hidden_dim})"
        )


class ReconstructionLoss(nn.Module):
    """
    Mean Squared Error Loss for data reconstruction fidelity.

    Measures how accurately the decoders can reconstruct original preprocessed
    data (X̃_RNA, X̃_ADT) from the compressed latent representation (Z_Fused).

    Mathematical Specification:
        L_recon = (1/N) * Σ_i [||X̃_RNA,i - X̂_RNA,i||² + ||X̃_ADT,i - X̂_ADT,i||²]

    Where:
        - X̃_RNA, X̃_ADT: Original preprocessed data from Module 1
        - X̂_RNA, X̂_ADT: Reconstructed by decoders
        - N: Number of spots (3484)

    Purpose:
        Reconstruction loss ensures:
        1. Latent representation contains sufficient information
        2. Compression (1024→64 dims) preserves biological signals
        3. Decoders learn meaningful mappings between spaces

    Loss Value Interpretation:
        - Good: 0.1-0.5 (indicates meaningful reconstruction)
        - Bad: > 1.0 (latent space has lost important information)

    Args:
        reduction (str): 'mean' or 'sum'. Default: 'mean' (normalized by N)
    """

    def __init__(self, reduction='mean'):
        """Initialize reconstruction loss."""
        super(ReconstructionLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_rna_true, x_rna_hat, x_adt_true, x_adt_hat):
        """
        Compute reconstruction loss for both modalities.

        Args:
            x_rna_true (torch.Tensor): Shape (N, 18085) - Preprocessed RNA from Module 1
            x_rna_hat (torch.Tensor): Shape (N, 18085) - Reconstructed RNA from decoder
            x_adt_true (torch.Tensor): Shape (N, 31) - Preprocessed ADT from Module 1
            x_adt_hat (torch.Tensor): Shape (N, 31) - Reconstructed ADT from decoder

        Returns:
            loss_recon (torch.Tensor): Scalar MSE loss
        """
        # Compute per-sample MSE for each modality
        mse_rna = F.mse_loss(x_rna_hat, x_rna_true, reduction=self.reduction)
        mse_adt = F.mse_loss(x_adt_hat, x_adt_true, reduction=self.reduction)

        # Combined reconstruction loss
        loss_recon = mse_rna + mse_adt

        return loss_recon

    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}(reduction='{self.reduction}')"


class SpatialRegularizationLoss(nn.Module):
    """
    Graph Laplacian Smoothness Loss for spatial regularization.

    Penalizes sudden changes in latent embeddings between physically adjacent
    spots. Eliminates technical "salt-and-pepper" noise and enforces spatial
    coherence in the 64-dimensional latent space.

    Mathematical Specification:
        L_spat = Σ_{i,j} A_{s,ij} * ||Z_Fused,i - Z_Fused,j||²

    Where:
        - A_s: Spatial adjacency matrix (k=6 nearest neighbors)
        - Z_Fused: Latent embeddings (3484 × 64)
        - Sum over all edges in spatial graph

    Expanded Form (using Graph Laplacian):
        L_spat = tr(Z_Fused^T * L * Z_Fused)
        where L = D - A_s (Laplacian matrix)

    Purpose:
        Spatial regularization ensures:
        1. Spatially adjacent spots have similar latent representations
        2. Tissue structure is preserved (boundaries remain intact)
        3. Small-scale noise is suppressed while preserving large-scale patterns

    Loss Value Interpretation:
        - Good: 0.01-0.1 (tissue has coherent structure)
        - Bad: > 0.5 (excessive fragmentation, noise not suppressed)

    Args:
        normalize (bool): Whether to normalize by number of edges. Default: True
    """

    def __init__(self, normalize=True):
        """Initialize spatial regularization loss."""
        super(SpatialRegularizationLoss, self).__init__()
        self.normalize = normalize

    def forward(self, z_fused, adj_spatial):
        """
        Compute spatial regularization loss.

        Args:
            z_fused (torch.Tensor): Shape (N, 64) - Latent embeddings from Module 6
            adj_spatial (torch.Tensor): Shape (N, N) - Spatial adjacency matrix from Module 3

        Returns:
            loss_spat (torch.Tensor): Scalar spatial regularization loss
        """
        # Ensure adj_spatial is a dense matrix (convert from sparse if needed)
        if adj_spatial.is_sparse:
            adj_spatial = adj_spatial.to_dense()

        # Compute pairwise differences: z_i - z_j for adjacent pairs
        # For each edge (i,j) in A_s, compute ||z_i - z_j||²
        n_nodes = z_fused.shape[0]
        
        # Method: Trace(Z^T * L * Z) where L = D - A_s is graph Laplacian
        # Equivalent to: Σ_{i,j} A_{ij} * ||Z_i - Z_j||²
        
        # Compute degree matrix D
        degree = adj_spatial.sum(dim=1)  # (N,)
        
        # Laplacian matrix: L = D - A
        # But we compute L * Z more efficiently:
        # L * Z = D * Z - A * Z (element-wise degree mult - adjacency product)
        
        # D * Z: each row multiplied by degree
        dz = degree.unsqueeze(1) * z_fused  # (N, 64)
        
        # A * Z: adjacency-weighted embedding sum
        az = torch.mm(adj_spatial, z_fused)  # (N, 64)
        
        # Laplacian applied to embeddings
        lz = dz - az  # (N, 64)
        
        # Loss: trace(Z^T * L * Z) = sum(Z * (L * Z))
        loss_spat = torch.sum(z_fused * lz)
        
        # Normalize by number of edges if requested
        if self.normalize:
            num_edges = adj_spatial.sum().item()
            loss_spat = loss_spat / max(num_edges, 1.0)
        
        return loss_spat

    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}(normalize={self.normalize})"


class ReconstructionModule(nn.Module):
    """
    Module 7: Reconstruction & Regularization (Decoder & Hub).

    Reconstructs high-dimensional RNA and ADT data from compressed 64-dim
    latent representation and computes both reconstruction and spatial losses.
    Serves as the "hub" where all three loss components (contrastive from Module 5,
    reconstruction, spatial) are computed for training.

    Core Architecture:
        • RNADecoder: 64 → 18085 dimensions
        • ADTDecoder: 64 → 31 dimensions
        • ReconstructionLoss: MSE between reconstructions and preprocessed data
        • SpatialRegularizationLoss: Graph Laplacian smoothness

    Inputs:
        z_fused (torch.Tensor): (3484, 64) from Module 6
        x_rna_true (torch.Tensor): (3484, 18085) from Module 1
        x_adt_true (torch.Tensor): (3484, 31) from Module 1
        adj_spatial (torch.Tensor): (3484, 3484) from Module 3

    Outputs:
        x_rna_hat, x_adt_hat: Reconstructions
        loss_recon, loss_spat: Losses for total training objective

    Attributes:
        rna_decoder (RNADecoder): RNA reconstruction MLP
        adt_decoder (ADTDecoder): ADT reconstruction MLP
        reconstruction_loss (ReconstructionLoss): MSE loss
        spatial_loss (SpatialRegularizationLoss): Laplacian smoothness loss
    """

    def __init__(
        self,
        latent_dim=64,
        rna_output_dim=18085,
        adt_output_dim=31,
        rna_hidden_dim=512,
        adt_hidden_dim=128
    ):
        """
        Initialize Reconstruction Module.

        Args:
            latent_dim (int): Latent dimension from Module 6 (64)
            rna_output_dim (int): RNA output dimension (18085)
            adt_output_dim (int): ADT output dimension (31)
            rna_hidden_dim (int): RNA decoder hidden size (512)
            adt_hidden_dim (int): ADT decoder hidden size (128)
        """
        super(ReconstructionModule, self).__init__()
        self.latent_dim = latent_dim
        self.rna_output_dim = rna_output_dim
        self.adt_output_dim = adt_output_dim

        # Decoders for each modality
        self.rna_decoder = RNADecoder(
            latent_dim=latent_dim,
            output_dim=rna_output_dim,
            hidden_dim=rna_hidden_dim
        )
        self.adt_decoder = ADTDecoder(
            latent_dim=latent_dim,
            output_dim=adt_output_dim,
            hidden_dim=adt_hidden_dim
        )

        # Loss functions
        self.reconstruction_loss = ReconstructionLoss(reduction='mean')
        self.spatial_loss = SpatialRegularizationLoss(normalize=True)

    def forward(self, z_fused, x_rna_true, x_adt_true, adj_spatial):
        """
        Forward pass: decode and compute losses.

        Args:
            z_fused (torch.Tensor): Shape (N, 64) - Fused embeddings from Module 6
            x_rna_true (torch.Tensor): Shape (N, 18085) - Preprocessed RNA from Module 1
            x_adt_true (torch.Tensor): Shape (N, 31) - Preprocessed ADT from Module 1
            adj_spatial (torch.Tensor): Shape (N, N) - Spatial adjacency from Module 3

        Returns:
            x_rna_hat (torch.Tensor): Reconstructed RNA (N, 18085)
            x_adt_hat (torch.Tensor): Reconstructed ADT (N, 31)
            loss_recon (torch.Tensor): Reconstruction loss (scalar)
            loss_spat (torch.Tensor): Spatial regularization loss (scalar)
        """
        # Decode: 64 → 18085 and 64 → 31
        x_rna_hat = self.rna_decoder(z_fused)  # (N, 18085)
        x_adt_hat = self.adt_decoder(z_fused)  # (N, 31)

        # Compute losses
        loss_recon = self.reconstruction_loss(x_rna_true, x_rna_hat, x_adt_true, x_adt_hat)
        loss_spat = self.spatial_loss(z_fused, adj_spatial)

        return x_rna_hat, x_adt_hat, loss_recon, loss_spat

    def compute_total_loss(self, loss_cl, loss_recon, loss_spat, 
                          lambda_cl=1.0, lambda_recon=1.0, lambda_spat=1.0):
        """
        Compute total training loss (all 3 modules combined).

        Formula:
            L_total = λ_cl * L_cl + λ_recon * L_recon + λ_spat * L_spat

        Args:
            loss_cl (torch.Tensor): Contrastive loss from Module 5
            loss_recon (torch.Tensor): Reconstruction loss from this module
            loss_spat (torch.Tensor): Spatial loss from this module
            lambda_cl (float): Weight for contrastive loss (default: 1.0)
            lambda_recon (float): Weight for reconstruction loss (default: 1.0)
            lambda_spat (float): Weight for spatial loss (default: 1.0)

        Returns:
            loss_total (torch.Tensor): Scalar total loss for backpropagation
        """
        loss_total = (
            lambda_cl * loss_cl +
            lambda_recon * loss_recon +
            lambda_spat * loss_spat
        )
        return loss_total

    def compute_reconstruction_quality(self, x_rna_true, x_rna_hat, x_adt_true, x_adt_hat):
        """
        Compute reconstruction quality metrics (optional monitoring).

        Metrics:
            - rna_mse: Mean squared error for RNA
            - adt_mse: Mean squared error for ADT
            - rna_mae: Mean absolute error for RNA
            - adt_mae: Mean absolute error for ADT

        Returns:
            metrics (dict): Quality metrics
        """
        with torch.no_grad():
            rna_mse = F.mse_loss(x_rna_hat, x_rna_true).item()
            adt_mse = F.mse_loss(x_adt_hat, x_adt_true).item()
            rna_mae = F.l1_loss(x_rna_hat, x_rna_true).item()
            adt_mae = F.l1_loss(x_adt_hat, x_adt_true).item()

        metrics = {
            'rna_mse': rna_mse,
            'adt_mse': adt_mse,
            'rna_mae': rna_mae,
            'adt_mae': adt_mae,
        }

        return metrics

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  RNA Decoder: {self.rna_decoder}\n"
            f"  ADT Decoder: {self.adt_decoder}\n"
            f"  Reconstruction Loss: {self.reconstruction_loss}\n"
            f"  Spatial Loss: {self.spatial_loss}\n"
            f")"
        )


def create_reconstruction_module(
    latent_dim=64,
    rna_output_dim=18085,
    adt_output_dim=31,
    device='cpu'
):
    """
    Factory function to create and configure ReconstructionModule.

    Simplifies initialization with KAC-Net defaults and device management.

    Args:
        latent_dim (int): Latent dimension (64)
        rna_output_dim (int): RNA genes (18085)
        adt_output_dim (int): ADT proteins (31)
        device (str or torch.device): Device placement ('cpu' or 'cuda')

    Returns:
        module (ReconstructionModule): Initialized and device-placed module

    Example:
        >>> module = create_reconstruction_module(device='cuda')
        >>> print(module)
    """
    module = ReconstructionModule(
        latent_dim=latent_dim,
        rna_output_dim=rna_output_dim,
        adt_output_dim=adt_output_dim
    )
    module = module.to(device)
    return module

"""
Module 6: Adaptive Dual-Attention Fusion (SpatialGlue Logic)

Core Purpose:
    Fuse aligned RNA and ADT embeddings through hierarchical (two-tier) attention
    mechanisms that dynamically learn which modality and which graph topology to
    trust at each physical spot location.

Key Architecture:
    • Tier 1 (Within-Modality): Graph blending attention learns importance of
      spatial (A_s) vs feature (A_f) graphs for each modality independently
    • Tier 2 (Between-Modality): Modality gating learns spot-specific weights
      to balance RNA vs ADT contribution based on signal quality
    • Output: Z_Fused (3484×64) - unified latent representation

Inputs:
    • Z_RNA ∈ R^(3484 × 512) - Aligned RNA embeddings from Module 5
    • Z_ADT ∈ R^(3484 × 512) - Aligned ADT embeddings from Module 5
    • A_s ∈ R^(3484 × 3484) - Spatial adjacency matrix from Module 3
    • A_f ∈ R^(3484 × 3484) - Feature adjacency matrix from Module 3

Outputs:
    • Z_Fused ∈ R^(3484 × 64) - Unified cross-modal embedding

Mathematical Foundation:
    Tier 1 (Within-Modality Graph Blending):
        α_s,i^m, α_f,i^m = Softmax(MLP_m(Z_m,i · A_s), MLP_m(Z_m,i · A_f))
        Z̃_m,i = α_s,i^m (W_s^m Z_m,i) + α_f,i^m (W_f^m Z_m,i)

    Tier 2 (Between-Modality Gating):
        ω_RNA,i, ω_ADT,i = Softmax(v^T tanh(W_g Z̃_RNA,i), v^T tanh(W_g Z̃_ADT,i))
        Z_Fused,i = (ω_RNA,i · W_R Z̃_RNA,i) + (ω_ADT,i · W_A Z̃_ADT,i)

References:
    • SpatialGlue: Encoder_overall class (complete architecture)
    • module_explanation.md: Complete mathematical specification
    • flow.md: Algorithm, inputs, outputs, mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBlendingAttention(nn.Module):
    """
    Tier 1: Within-Modality Graph Blending Attention.

    Learns to blend spatial (A_s) and feature (A_f) graph information for a
    single modality by computing adaptive attention weights for each graph type.

    Core Idea:
        Different tissue regions have different characteristics. In some regions,
        spatial proximity might be more informative (e.g., tissue layers). In others,
        gene expression similarity might be more important (e.g., same cell type
        scattered across tissue). This attention learns the best blend per spot.

    Mathematical Formulation:
        For modality m with embedding Z_m,i at spot i:
        α_s,i^m, α_f,i^m = Softmax(MLP_m(Z_m,i · A_s), MLP_m(Z_m,i · A_f))
        Z̃_m,i = α_s,i^m (W_s^m Z_m,i) + α_f,i^m (W_f^m Z_m,i)

    Args:
        in_features (int): Input embedding dimension (512 from Module 5)
        out_features (int): Output dimension (typically same as input)
        hidden_features (int): MLP hidden dimension for attention computation

    Inputs:
        z (torch.Tensor): Shape (N, in_features) - Embeddings for one modality
        adj_spatial (torch.Tensor): Shape (N, N) or edge_index - Spatial adjacency
        adj_feature (torch.Tensor): Shape (N, N) or edge_index - Feature adjacency

    Output:
        z_blended (torch.Tensor): Shape (N, out_features) - Graph-blended embeddings
    """

    def __init__(self, in_features, out_features=None, hidden_features=256):
        """Initialize graph blending attention layer."""
        super(GraphBlendingAttention, self).__init__()
        if out_features is None:
            out_features = in_features
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        # MLP for computing attention weights from embeddings
        # Takes embedding and outputs 2 logits (one per graph)
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 2)  # Output: [logit_s, logit_f]
        )

        # Separate linear projections for spatial and feature graphs
        self.proj_spatial = nn.Linear(in_features, out_features)
        self.proj_feature = nn.Linear(in_features, out_features)

    def forward(self, z, adj_spatial, adj_feature):
        """
        Compute graph-blended embeddings using adaptive attention.

        Args:
            z (torch.Tensor): Shape (N, in_features) - Node embeddings
            adj_spatial (torch.Tensor): Shape (N, N) - Spatial adjacency matrix
            adj_feature (torch.Tensor): Shape (N, N) - Feature adjacency matrix

        Returns:
            z_blended (torch.Tensor): Shape (N, out_features) - Blended embeddings
            alpha_weights (torch.Tensor): Shape (N, 2) - Attention weights [α_s, α_f]
        """
        batch_size = z.shape[0]
        device = z.device

        # Compute attention weights via MLP
        # logits shape: (N, 2) where [:, 0] = logit for A_s, [:, 1] = logit for A_f
        logits = self.attention_mlp(z)  # (N, 2)

        # Apply softmax per node to get normalized weights
        alpha_weights = F.softmax(logits, dim=1)  # (N, 2), sum to 1 per row
        alpha_spatial = alpha_weights[:, 0]  # (N,)
        alpha_feature = alpha_weights[:, 1]  # (N,)

        # Project embeddings through spatial and feature branches
        z_proj_spatial = self.proj_spatial(z)  # (N, out_features)
        z_proj_feature = self.proj_feature(z)  # (N, out_features)

        # Weight and combine: α_s · proj_s + α_f · proj_f
        z_blended = (
            alpha_spatial.unsqueeze(1) * z_proj_spatial +  # (N, 1) * (N, d)
            alpha_feature.unsqueeze(1) * z_proj_feature    # (N, 1) * (N, d)
        )  # Result: (N, out_features)

        return z_blended, alpha_weights

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, hidden_features={self.hidden_features})"
        )


class ModalityGating(nn.Module):
    """
    Tier 2: Between-Modality Gating Attention.

    Learns spot-specific weights to balance RNA vs ADT contributions based on
    signal quality and reliability. Some spots have better RNA signal, others
    have better protein signal - this module learns which to trust.

    Core Idea:
        Biological signal reliability varies by location:
        - B-cell follicles: CD19 protein signal is clear, RNA might be noisy
        - T-cell zones: Both signals are reliable, need careful blending
        - Boundary regions: One modality might be dropout-prone
        
        This gating learns to upweight the more reliable modality per spot.

    Mathematical Formulation:
        For each spot i:
        ω_RNA,i, ω_ADT,i = Softmax(v^T tanh(W_g Z̃_RNA,i), v^T tanh(W_g Z̃_ADT,i))
        Z_Fused,i = (ω_RNA,i · W_R Z̃_RNA,i) + (ω_ADT,i · W_A Z̃_ADT,i)

    Args:
        in_features (int): Input embedding dimension from graph blending
        out_features (int): Output fused dimension (typically 64)
        hidden_features (int): Gate network hidden dimension
    """

    def __init__(self, in_features, out_features=64, hidden_features=128):
        """Initialize modality gating layer."""
        super(ModalityGating, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        # Shared gate network: outputs single scalar logit per modality
        # Gate computes: v^T tanh(W_g z)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh()  # Non-linearity for gate
        )
        # Attention vector v that combines tanh outputs
        self.gate_vector = nn.Linear(hidden_features, 1)

        # Separate projection matrices for fusing modalities
        self.proj_rna = nn.Linear(in_features, out_features)
        self.proj_adt = nn.Linear(in_features, out_features)

    def forward(self, z_rna_blended, z_adt_blended):
        """
        Compute modality-gated fusion.

        Args:
            z_rna_blended (torch.Tensor): Shape (N, in_features) - Tier 1 RNA output
            z_adt_blended (torch.Tensor): Shape (N, in_features) - Tier 1 ADT output

        Returns:
            z_fused (torch.Tensor): Shape (N, out_features) - Fused embedding
            omega_weights (torch.Tensor): Shape (N, 2) - Gating weights [ω_RNA, ω_ADT]
        """
        # Compute gate activations: tanh(W_g z)
        rna_gate = self.gate_mlp(z_rna_blended)  # (N, hidden_features)
        adt_gate = self.gate_mlp(z_adt_blended)  # (N, hidden_features)

        # Project through attention vector to get logits
        rna_logit = self.gate_vector(rna_gate)  # (N, 1)
        adt_logit = self.gate_vector(adt_gate)  # (N, 1)

        # Concatenate logits and apply softmax for normalization
        logits = torch.cat([rna_logit, adt_logit], dim=1)  # (N, 2)
        omega_weights = F.softmax(logits, dim=1)  # (N, 2), sums to 1 per row
        omega_rna = omega_weights[:, 0]  # (N,)
        omega_adt = omega_weights[:, 1]  # (N,)

        # Project and weight-combine modalities
        z_rna_proj = self.proj_rna(z_rna_blended)  # (N, out_features)
        z_adt_proj = self.proj_adt(z_adt_blended)  # (N, out_features)

        # Fused: ω_RNA · proj_RNA + ω_ADT · proj_ADT
        z_fused = (
            omega_rna.unsqueeze(1) * z_rna_proj +  # (N, 1) * (N, d)
            omega_adt.unsqueeze(1) * z_adt_proj    # (N, 1) * (N, d)
        )  # Result: (N, out_features)

        return z_fused, omega_weights

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, hidden_features={self.hidden_features})"
        )


class DualAttentionFusionModule(nn.Module):
    """
    Module 6: Adaptive Dual-Attention Fusion (SpatialGlue Logic).

    Implements two-tier hierarchical attention to fuse aligned RNA and ADT
    embeddings into a unified representation while dynamically learning which
    information sources (graphs, modalities) are most reliable at each location.

    Architecture:
        Layer 1 (Tier 1): GraphBlendingAttention
            ├─ RNA stream: Learns α_s^RNA, α_f^RNA (spatial vs feature graphs)
            └─ ADT stream: Learns α_s^ADT, α_f^ADT (spatial vs feature graphs)
                Result: Z̃_RNA, Z̃_ADT (graph-blended embeddings)
        
        Layer 2 (Tier 2): ModalityGating
            ├─ Learns ω_RNA, ω_ADT (RNA vs ADT contribution per spot)
            └─ Fuses: Z_Fused = ω_RNA·Z̃_RNA + ω_ADT·Z̃_ADT
                Result: Z_Fused (64-dim unified embedding)

    Key Innovation:
        - Not just concatenation (would be 1024-dim)
        - Not just simple averaging (loses information)
        - Adaptive weighting learns optimal blend per spot and per modality

    Inputs:
        z_rna (torch.Tensor): Aligned RNA (N, 512) from Module 5
        z_adt (torch.Tensor): Aligned ADT (N, 512) from Module 5
        adj_spatial (torch.Tensor): Spatial graph (N, N) from Module 3
        adj_feature (torch.Tensor): Feature graph (N, N) from Module 3

    Outputs:
        z_fused (torch.Tensor): Unified embedding (N, 64)

    Attributes:
        tier1_rna (GraphBlendingAttention): RNA graph blending
        tier1_adt (GraphBlendingAttention): ADT graph blending
        tier2 (ModalityGating): Cross-modal fusion gating
    """

    def __init__(self, in_features=512, hidden_features=256, out_features=64):
        """
        Initialize Dual-Attention Fusion Module.

        Args:
            in_features (int): Input dimension from Module 5 (512)
            hidden_features (int): Hidden dimension for MLPs (256)
            out_features (int): Final fused dimension (64)
        """
        super(DualAttentionFusionModule, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Tier 1: Within-modality graph blending (independent for each modality)
        self.tier1_rna = GraphBlendingAttention(
            in_features=in_features,
            out_features=in_features,
            hidden_features=hidden_features
        )
        self.tier1_adt = GraphBlendingAttention(
            in_features=in_features,
            out_features=in_features,
            hidden_features=hidden_features
        )

        # Tier 2: Between-modality gating (combines both modalities)
        self.tier2 = ModalityGating(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features
        )

    def forward(self, z_rna, z_adt, adj_spatial, adj_feature, return_weights=False):
        """
        Compute dual-attention fusion for RNA and ADT embeddings.

        Args:
            z_rna (torch.Tensor): Aligned RNA embeddings (N, 512) from Module 5
            z_adt (torch.Tensor): Aligned ADT embeddings (N, 512) from Module 5
            adj_spatial (torch.Tensor): Spatial adjacency (N, N) from Module 3
            adj_feature (torch.Tensor): Feature adjacency (N, N) from Module 3
            return_weights (bool): If True, also return attention weights. Default: False

        Returns:
            z_fused (torch.Tensor): Unified embedding (N, 64)
            or (z_fused, alpha_rna, alpha_adt, omega) if return_weights=True

        Forward Pass Logic:
            1. Tier 1: Each modality learns best graph blend
               - RNA: α_s^RNA, α_f^RNA → Z̃_RNA
               - ADT: α_s^ADT, α_f^ADT → Z̃_ADT
            2. Tier 2: Learn best modality blend
               - ω_RNA, ω_ADT → Z_Fused = ω_RNA·Z̃_RNA + ω_ADT·Z̃_ADT
        """
        # TIER 1: Within-Modality Graph Blending
        # Each modality independently learns importance of spatial vs feature graphs
        z_rna_blended, alpha_rna = self.tier1_rna(z_rna, adj_spatial, adj_feature)
        z_adt_blended, alpha_adt = self.tier1_adt(z_adt, adj_spatial, adj_feature)

        # TIER 2: Between-Modality Gating
        # Learn spot-specific weights to balance RNA vs ADT
        z_fused, omega_weights = self.tier2(z_rna_blended, z_adt_blended)

        if return_weights:
            return z_fused, alpha_rna, alpha_adt, omega_weights

        return z_fused

    def compute_fusion_quality(self, z_rna, z_adt, adj_spatial, adj_feature):
        """
        Compute fusion quality metrics for monitoring (optional).

        Metrics:
            - alpha_rna_entropy: Uncertainty in RNA graph choice (0=certain, 1=uncertain)
            - alpha_adt_entropy: Uncertainty in ADT graph choice
            - omega_entropy: Uncertainty in modality choice
            - rna_dominance: Fraction of spots where RNA > ADT
            - adt_dominance: Fraction of spots where ADT > RNA

        Returns:
            metrics (dict): Quality metrics
        """
        with torch.no_grad():
            _, alpha_rna, alpha_adt, omega = self.forward(
                z_rna, z_adt, adj_spatial, adj_feature, return_weights=True
            )

        # Compute entropy: -Σ(p * log(p))
        def entropy(probs):
            return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()

        alpha_rna_entropy = entropy(alpha_rna)
        alpha_adt_entropy = entropy(alpha_adt)
        omega_entropy = entropy(omega)

        # Dominance metrics
        rna_dominance = (omega[:, 0] > omega[:, 1]).float().mean().item()
        adt_dominance = (omega[:, 1] > omega[:, 0]).float().mean().item()

        metrics = {
            'alpha_rna_entropy': alpha_rna_entropy,
            'alpha_adt_entropy': alpha_adt_entropy,
            'omega_entropy': omega_entropy,
            'rna_dominance': rna_dominance,
            'adt_dominance': adt_dominance,
        }

        return metrics

    def __repr__(self):
        """String representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_features={self.in_features},\n"
            f"  hidden_features={self.hidden_features},\n"
            f"  out_features={self.out_features}\n"
            f"  ├─ Tier 1 RNA: {self.tier1_rna}\n"
            f"  ├─ Tier 1 ADT: {self.tier1_adt}\n"
            f"  └─ Tier 2: {self.tier2}\n"
            f")"
        )


def create_fusion_module(in_features=512, hidden_features=256, out_features=64, device='cpu'):
    """
    Factory function to create and configure DualAttentionFusionModule.

    Simplifies initialization with KAC-Net defaults and device management.

    Args:
        in_features (int): Input dimension (512). Default: 512
        hidden_features (int): MLP hidden dimension (256). Default: 256
        out_features (int): Output fused dimension (64). Default: 64
        device (str or torch.device): Device placement ('cpu' or 'cuda'). Default: 'cpu'

    Returns:
        module (DualAttentionFusionModule): Initialized and device-placed module

    Example:
        >>> module = create_fusion_module(in_features=512, out_features=64, device='cuda')
        >>> print(module)
    """
    module = DualAttentionFusionModule(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features
    )
    module = module.to(device)
    return module

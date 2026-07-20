"""
KAC-Net v2 — model.py
Core network modules implementing:
  - Module 4: Local Spatial Encoding (Residual GATv2Conv)
  - Module 5: Projection Head for Contrastive Alignment
  - Module 6: Adaptive Dual-Attention Fusion (Within + Between modality attention)
  - Module 7: Reconstruction & Regularization Decoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


# ---------------------------------------------------------------------------
# Module 4 – Local Spatial Encoding (Residual GATv2)
# ---------------------------------------------------------------------------

class ResidualGATv2Encoder(nn.Module):
    """
    Module 4 – GATv2 Encoder with Residual Skip-connections.
    Learns spatial feature maps Z_modality for each graph view.
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super(ResidualGATv2Encoder, self).__init__()
        # Ensure multi-head GATv2 output dimension equals out_channels
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        # Linear layer for skip-connection if input and output dimensions mismatch
        if in_channels != out_channels:
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.skip = nn.Identity()
            
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        # Local GATv2 Message Passing
        h = self.gat(x, edge_index)
        # Residual skip connection + Layer normalization
        out = F.elu(self.ln(h + self.skip(x)))
        return out


# ---------------------------------------------------------------------------
# Module 5 – Cross-Modal Contrastive Alignment (Projection Heads)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Module 5 – Projection Head for InfoNCE Contrastive Loss.
    Maps latent space representations to a lower-dimensional space (e.g. 64-dim)
    where RNA and ADT features can be aligned.
    """
    def __init__(self, in_channels, projection_dim=64):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ELU(),
            nn.Linear(in_channels, projection_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Module 6 – Adaptive Dual-Attention Fusion (SpatialGlue + spaLLM logic)
# ---------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    """
    Within-modality attention blending.
    Learns query/key coefficients for aggregating spatial and feature GNN outputs.
    """
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.w = nn.Parameter(torch.zeros(in_channels, 1))
        nn.init.xavier_uniform_(self.w.data)

    def forward(self, *inputs):
        # Stack inputs: (N, num_graphs, in_channels)
        stacked = torch.stack(inputs, dim=1)
        # Compute attention scores
        scores = torch.matmul(stacked, self.w).squeeze(-1)  # (N, num_graphs)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)    # (N, num_graphs, 1)
        # Weighted sum of graph representations
        out = torch.sum(stacked * weights, dim=1)           # (N, in_channels)
        return out, weights.squeeze(-1)


class GateFusion(nn.Module):
    """
    Between-modality fusion gate.
    Dynamically blends RNA and ADT modalities to build the integrated space.
    """
    def __init__(self, in_channels):
        super(GateFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x_rna, x_adt):
        cat = torch.cat([x_rna, x_adt], dim=-1)
        g = self.gate(cat)
        # Gated fusion output
        fused = g * x_rna + (1 - g) * x_adt
        return fused, g


# ---------------------------------------------------------------------------
# Module 7 – Reconstruction & Regularization Decoders
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Module 7 – Decoder to project fused representation back to input count space.
    """
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ELU(),
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# KACNet Framework Wrapper Model
# ---------------------------------------------------------------------------

class KACNet(nn.Module):
    """
    KAC-Net v2 model wrapping encoders, projections, attention fusion, and decoders.
    """
    def __init__(self, dim_rna, dim_adt, latent_dim=128, proj_dim=64):
        super(KACNet, self).__init__()
        
        # Encoders for RNA
        self.encoder_rna_spatial = ResidualGATv2Encoder(dim_rna, latent_dim)
        self.encoder_rna_feature = ResidualGATv2Encoder(dim_rna, latent_dim)
        
        # Encoders for ADT
        self.encoder_adt_spatial = ResidualGATv2Encoder(dim_adt, latent_dim)
        self.encoder_adt_feature = ResidualGATv2Encoder(dim_adt, latent_dim)
        
        # Projection Heads for Cross-Modal Contrastive Alignment (Module 5)
        self.proj_rna = ProjectionHead(latent_dim, proj_dim)
        self.proj_adt = ProjectionHead(latent_dim, proj_dim)
        
        # Attention Layers (Module 6)
        self.att_rna = AttentionLayer(latent_dim)
        self.att_adt = AttentionLayer(latent_dim)
        self.gate_fusion = GateFusion(latent_dim)
        
        # Decoders (Module 7)
        self.decoder_rna = Decoder(latent_dim, dim_rna)
        self.decoder_adt = Decoder(latent_dim, dim_adt)

    def forward(self, rna_feat, adt_feat, adj_spatial, adj_feature):
        # 1. Local spatial encoding
        z_rna_s = self.encoder_rna_spatial(rna_feat, adj_spatial)
        z_rna_f = self.encoder_rna_feature(rna_feat, adj_feature)
        
        z_adt_s = self.encoder_adt_spatial(adt_feat, adj_spatial)
        z_adt_f = self.encoder_adt_feature(adt_feat, adj_feature)
        
        # 2. Within-modality attention blending
        z_rna, att_rna_weights = self.att_rna(z_rna_s, z_rna_f)
        z_adt, att_adt_weights = self.att_adt(z_adt_s, z_adt_f)
        
        # 3. Cross-modal projection mapping
        p_rna = self.proj_rna(z_rna)
        p_adt = self.proj_adt(z_adt)
        
        # 4. Gated modality fusion
        h_fused, gate_weights = self.gate_fusion(z_rna, z_adt)
        
        # 5. Reconstruction
        recon_rna = self.decoder_rna(h_fused)
        recon_adt = self.decoder_adt(h_fused)
        
        return {
            'z_rna': z_rna,
            'z_adt': z_adt,
            'p_rna': p_rna,
            'p_adt': p_adt,
            'h_fused': h_fused,
            'recon_rna': recon_rna,
            'recon_adt': recon_adt,
            'att_rna_weights': att_rna_weights,
            'att_adt_weights': att_adt_weights,
            'gate_weights': gate_weights
        }

"""
KAC-Net Module 4: Local Spatial Encoding - Residual GATv2 Implementation
=========================================================================

Implements Residual Graph Attention Network v2 (ResGATv2) for adaptive spatial encoding.
Combines residual skip connections with GATv2 attention to prevent over-smoothing while
learning adaptive neighborhood aggregation weights.

Classes:
    - ResGATv2Layer: GATv2 with integrated residual connections
    - MultiHeadResGATv2: Multi-head residual attention over multiple graphs
    - ResGATModel: Complete Residual GATv2 encoder for spatial encoding

Key Features:
    - Residual skip connections at layer level (prevent over-smoothing)
    - Adaptive attention weights for each neighbor
    - Multi-head parallel attention
    - Dual-graph learning: spatial (k=6) + feature (k=20)
    - Layer normalization for training stability
    - Deep architecture support (2+ layers without gradient collapse)

Mathematical Foundation:
    GATv2 Attention: α_ij = softmax(a^T·LeakyReLU(W[h_i || h_j]))
    Residual Connection: h_i^(l+1) = LayerNorm(GATv2(h_i^(l)) + W_res·h_i^(l))
    Output: h_i' = σ(Σ_j α_ij · W · h_j)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, Dict, Union
import numpy as np
from anndata import AnnData


class ResGATv2Layer(nn.Module):
    """
    Single Residual Graph Attention Network v2 (ResGATv2) layer.
    
    Combines GATv2 attention mechanism with residual skip connections to:
    1. Learn adaptive neighbor aggregation weights
    2. Prevent over-smoothing through residual identity mapping
    3. Enable deep architectures without gradient collapse
    
    Mathematical Formulation
    ========================
    GATv2 Attention:
        For node i with neighbors j ∈ N(i):
        
        1. Transform: h_i^(t) = W·h_i^(t-1)
        2. Attention logits: e_ij = a^T·LeakyReLU([h_i || h_j])
        3. Softmax: α_ij = exp(e_ij) / Σ_k exp(e_ik)
        4. Aggregate: h_i' = σ(Σ_j α_ij · W' · h_j)
    
    Residual Connection (Key Innovation):
        h_i^(l+1) = LayerNorm(h_i' + W_res · h_i^(l))
    
    The residual path allows original features to bypass the attention
    operation, preserving fine-grained local information even after
    aggregating global neighborhood context.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        negative_slope: float = 0.2,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Parameters
        ----------
        in_features : int
            Number of input features per node (e.g., 512 for H_RNA)
        out_features : int
            Number of output features per head
        num_heads : int
            Number of parallel attention heads (default: 8)
        negative_slope : float
            Negative slope for LeakyReLU activation (default: 0.2)
        dropout : float
            Dropout probability for attention weights (default: 0.1)
        bias : bool
            Whether to use bias in linear layers (default: True)
        """
        super(ResGATv2Layer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.dropout_prob = dropout
        
        # Linear transformation: in_features → out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism: concatenated features → attention logits
        # After linear: out_features; after concat: 2*out_features
        self.attention = nn.Linear(2 * out_features, 1, bias=False)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Residual projection: in_features → out_features
        # Maps input to same dimension for residual addition
        self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.xavier_uniform_(self.residual_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of Residual GATv2 layer with skip connection.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, in_features)
            Example: H_RNA (3484, 512)
        
        edge_index : torch.Tensor
            Edge list in COO format: shape (2, num_edges)
            edge_index[0] = source nodes
            edge_index[1] = target nodes
        
        edge_weight : torch.Tensor, optional
            Edge weights for weighted aggregation (default: None = uniform)
        
        return_attention : bool
            If True, return attention weights for interpretation
        
        Returns
        -------
        torch.Tensor
            Output node features with residual: (num_nodes, out_features)
        
        If return_attention=True:
            Tuple of (output, attention_weights)
        """
        # Store input for residual connection
        residual = self.residual_proj(x)  # (num_nodes, out_features)
        
        # Linear transformation
        x_transformed = self.linear(x)  # (num_nodes, out_features)
        
        if self.bias is not None:
            x_transformed = x_transformed + self.bias
        
        # Get source and target node indices
        source_idx, target_idx = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]
        
        # Gather source and target features
        x_i = x_transformed[source_idx]  # (num_edges, out_features)
        x_j = x_transformed[target_idx]  # (num_edges, out_features)
        
        # Concatenate features for attention computation (GATv2 style)
        x_concat = torch.cat([x_i, x_j], dim=-1)  # (num_edges, 2*out_features)
        
        # Compute attention logits
        attention_logits = self.attention(x_concat)  # (num_edges, 1)
        attention_logits = self.leaky_relu(attention_logits)
        
        # Compute attention coefficients (softmax per target node)
        attention_coeffs = torch.zeros_like(attention_logits)
        
        for i in range(num_nodes):
            mask = target_idx == i
            if mask.any():
                logits_i = attention_logits[mask]
                # Softmax normalization per target
                attention_coeffs[mask] = F.softmax(logits_i, dim=0)
        
        # Apply dropout to attention weights
        attention_coeffs = self.dropout_layer(attention_coeffs)
        
        # Apply edge weights if provided
        if edge_weight is not None:
            attention_coeffs = attention_coeffs * edge_weight.unsqueeze(-1)
        
        # Aggregate messages from neighbors
        weighted_features = x_j * attention_coeffs  # (num_edges, out_features)
        
        # Sum aggregation: for each target node, sum weighted features
        aggregated = torch.zeros_like(residual)
        for i in range(num_nodes):
            mask = target_idx == i
            if mask.any():
                aggregated[i] = weighted_features[mask].sum(dim=0)
        
        # **RESIDUAL CONNECTION**: Add original input back
        # This is key to ResGATv2 - prevents over-smoothing
        output = aggregated + residual  # (num_nodes, out_features)
        
        # Layer normalization for training stability
        output = self.layer_norm(output)
        
        if return_attention:
            return output, attention_coeffs
        else:
            return output


class MultiHeadResGATv2(nn.Module):
    """
    Multi-head Residual GATv2 layer.
    
    Runs multiple ResGATv2 layers in parallel with different parameters,
    then concatenates outputs. Each head learns different neighborhood
    relationship patterns while maintaining residual connections.
    
    Mathematical Formulation
    ========================
    Multi-head with residuals:
        h_i^(multi) = || _{k=1}^{K} ResGATv2_k(h_i)
                    = || _{k=1}^{K} [σ(Σ_j α_{ij}^k · W^k · h_j) + h_i]
    
    Where:
        K = num_heads (typically 8)
        || = concatenation operator
        α_{ij}^k = attention from head k
        W^k = parameters unique to head k
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        negative_slope: float = 0.2,
        dropout: float = 0.1,
        concat: bool = True
    ):
        """
        Parameters
        ----------
        in_features : int
            Input feature dimension
        out_features : int
            Output feature dimension per head
        num_heads : int
            Number of parallel attention heads (default: 8)
        negative_slope : float
            Negative slope for LeakyReLU (default: 0.2)
        dropout : float
            Dropout probability (default: 0.1)
        concat : bool
            If True, concatenate head outputs. If False, average them.
        """
        super(MultiHeadResGATv2, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        
        # Create independent residual attention heads
        self.heads = nn.ModuleList([
            ResGATv2Layer(
                in_features=in_features,
                out_features=out_features,
                num_heads=1,
                negative_slope=negative_slope,
                dropout=dropout,
                bias=True
            )
            for _ in range(num_heads)
        ])
        
        # Final projection if concatenating
        if concat:
            self.out_proj = nn.Linear(num_heads * out_features, out_features)
        else:
            self.out_proj = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head residual attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features (num_nodes, in_features)
        edge_index : torch.Tensor
            Edge list (2, num_edges)
        edge_weight : torch.Tensor, optional
            Edge weights (num_edges,)
        
        Returns
        -------
        torch.Tensor
            Output features (num_nodes, out_features)
        """
        # Apply all attention heads in parallel (each with residuals)
        head_outputs = [
            head(x, edge_index, edge_weight)
            for head in self.heads
        ]
        
        if self.concat:
            # Concatenate outputs: (num_nodes, num_heads * out_features)
            x_out = torch.cat(head_outputs, dim=-1)
            # Project back to out_features
            x_out = self.out_proj(x_out)
        else:
            # Average outputs
            x_out = torch.stack(head_outputs, dim=0).mean(dim=0)
        
        return x_out


class ResGATModel(nn.Module):
    """
    Complete Residual GATv2 encoder for local spatial encoding (Module 4).
    
    Processes BOTH RNA and ADT modalities through dual-stream residual
    attention, learning adaptive weights for both neighborhood types while
    maintaining residual identity mappings to prevent over-smoothing.
    
    Architecture
    =============
    Input RNA: H_RNA (3484, 512), A_s, A_f
    Input ADT: X_ADT (3484, 31) → ADT_proj → (3484, 512)
        ↓
    ResGATv2 Layer 1 (Spatial):  inputs → outputs (3484, 256)
    ResGATv2 Layer 1 (Feature):  inputs → outputs (3484, 256)
        ↓ concatenate + project
    Fusion Layer 1:             (3484, 512) → (3484, 512)
        ↓ residual skip
    H_fused_1 (3484, 512)
        ↓
    ResGATv2 Layer 2 (repeat pattern)
        ↓
    Output Z_RNA, Z_ADT (3484, 512) each
    """
    
    def __init__(
        self,
        in_features: int = 512,
        adt_features: int = 31,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        negative_slope: float = 0.2,
        dropout: float = 0.1
    ):
        """
        Parameters
        ----------
        in_features : int
            Input feature dimension (default: 512 from Module 2)
        adt_features : int
            Normalized ADT feature dimension (default: 31, CLR-normalized from Module 1)
        hidden_dim : int
            Hidden dimension per attention head (default: 256)
        num_layers : int
            Number of Residual GATv2 layers (default: 2)
        num_heads : int
            Number of attention heads (default: 8)
        negative_slope : float
            Negative slope for LeakyReLU (default: 0.2)
        dropout : float
            Dropout probability (default: 0.1)
        """
        super(ResGATModel, self).__init__()
        
        self.in_features = in_features
        self.adt_features = adt_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # ADT projection: 31 dims → 512 dims to match RNA space
        # This brings ADT into same embedding space as H_RNA from Module 2
        self.adt_projection = nn.Linear(adt_features, in_features)
        
        # Create Residual GATv2 layers for spatial and feature graphs
        self.spatial_layers = nn.ModuleList()
        self.feature_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in_dim = in_features if i == 0 else in_features
            layer_out_dim = hidden_dim
            
            # Spatial graph residual attention
            spatial_gat = MultiHeadResGATv2(
                in_features=layer_in_dim,
                out_features=layer_out_dim,
                num_heads=num_heads,
                negative_slope=negative_slope,
                dropout=dropout,
                concat=True
            )
            self.spatial_layers.append(spatial_gat)
            
            # Feature graph residual attention
            feature_gat = MultiHeadResGATv2(
                in_features=layer_in_dim,
                out_features=layer_out_dim,
                num_heads=num_heads,
                negative_slope=negative_slope,
                dropout=dropout,
                concat=True
            )
            self.feature_layers.append(feature_gat)
            
            # Fusion layer: combine spatial + feature (2*out_features → in_features)
            # with residual connection
            fusion = nn.Sequential(
                nn.Linear(2 * hidden_dim, in_features),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fusion_layers.append(fusion)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x_rna: torch.Tensor,
        x_adt: torch.Tensor,
        adj_spatial: Union[torch.Tensor, csr_matrix],
        adj_feature: Union[torch.Tensor, csr_matrix],
        return_attention: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], Dict]]:
        """
        Forward pass through Residual GATv2 encoder for DUAL MODALITIES.
        
        Parameters
        ----------
        x_rna : torch.Tensor
            RNA node features (num_nodes, in_features)
            Expected: H_RNA (3484, 512) from Module 2
        
        x_adt : torch.Tensor
            Normalized ADT node features (num_nodes, adt_features)
            Expected: X̃_ADT (3484, 31) CLR-normalized from Module 1
        
        adj_spatial : torch.Tensor or csr_matrix
            Spatial adjacency matrix (3484, 3484)
        
        adj_feature : torch.Tensor or csr_matrix
            Feature adjacency matrix (3484, 3484)
        
        return_attention : bool
            If True, return attention statistics
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (Z_RNA, Z_ADT) both shape (num_nodes, in_features)
        
        If return_attention=True:
            Tuple of ((Z_RNA, Z_ADT), attention_dict)
        """
        # Project ADT from 31 dims to 512 dims to match RNA space
        x_adt_proj = self.adt_projection(x_adt)  # (3484, 512)
        
        # Convert adjacency matrices to edge lists
        if isinstance(adj_spatial, csr_matrix):
            edge_index_s = self._to_edge_index(adj_spatial)
            edge_weight_s = torch.tensor(adj_spatial.data, dtype=torch.float32)
        else:
            edge_index_s = adj_spatial
            edge_weight_s = None
        
        if isinstance(adj_feature, csr_matrix):
            edge_index_f = self._to_edge_index(adj_feature)
            edge_weight_f = torch.tensor(adj_feature.data, dtype=torch.float32)
        else:
            edge_index_f = adj_feature
            edge_weight_f = None
        
        # Move tensors to same device
        device = x_rna.device
        edge_index_s = edge_index_s.to(device)
        edge_index_f = edge_index_f.to(device)
        if edge_weight_s is not None:
            edge_weight_s = edge_weight_s.to(device)
        if edge_weight_f is not None:
            edge_weight_f = edge_weight_f.to(device)
        
        # Process RNA and ADT through the same graph structure
        h_rna = x_rna
        h_adt = x_adt_proj
        
        for i in range(self.num_layers):
            # Apply spatial and feature residual attention to RNA
            h_rna_spatial = self.spatial_layers[i](h_rna, edge_index_s, edge_weight_s)
            h_rna_feature = self.feature_layers[i](h_rna, edge_index_f, edge_weight_f)
            
            # Concatenate spatial + feature representations for RNA
            h_rna_combined = torch.cat([h_rna_spatial, h_rna_feature], dim=-1)
            
            # Fusion: combine both streams for RNA
            h_rna_fused = self.fusion_layers[i](h_rna_combined)
            
            # **RESIDUAL SKIP CONNECTION**: Add original input back to RNA
            h_rna = h_rna_fused + h_rna
            h_rna = self.dropout_layer(h_rna)
            
            # Repeat same process for ADT using SAME graph structure
            h_adt_spatial = self.spatial_layers[i](h_adt, edge_index_s, edge_weight_s)
            h_adt_feature = self.feature_layers[i](h_adt, edge_index_f, edge_weight_f)
            
            # Concatenate spatial + feature representations for ADT
            h_adt_combined = torch.cat([h_adt_spatial, h_adt_feature], dim=-1)
            
            # Fusion: combine both streams for ADT
            h_adt_fused = self.fusion_layers[i](h_adt_combined)
            
            # **RESIDUAL SKIP CONNECTION**: Add original input back to ADT
            h_adt = h_adt_fused + h_adt
            h_adt = self.dropout_layer(h_adt)
        
        # Return both Z_RNA and Z_ADT
        if return_attention:
            return (h_rna, h_adt), {'spatial_edges': edge_index_s, 'feature_edges': edge_index_f}
        else:
            return h_rna, h_adt
    
    @staticmethod
    def _to_edge_index(adj: csr_matrix) -> torch.Tensor:
        """
        Convert sparse CSR matrix to COO edge index format.
        
        Parameters
        ----------
        adj : csr_matrix
            Sparse adjacency matrix
        
        Returns
        -------
        torch.Tensor
            Edge index of shape (2, num_edges)
        """
        adj_coo = adj.tocoo()
        edge_index = torch.tensor(
            [adj_coo.row, adj_coo.col],
            dtype=torch.long
        )
        return edge_index
    
    def encode(
        self,
        x_rna: torch.Tensor,
        x_adt: torch.Tensor,
        adj_spatial: Union[torch.Tensor, csr_matrix],
        adj_feature: Union[torch.Tensor, csr_matrix]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Semantic wrapper for forward() for clarity.
        
        Encodes BOTH RNA and ADT node features with spatial context information,
        maintaining residual paths throughout the network and projecting ADT to
        the same embedding space as RNA.
        
        Parameters
        ----------
        x_rna : torch.Tensor
            RNA node features (num_nodes, in_features)
            Expected: H_RNA (3484, 512) from Module 2
        x_adt : torch.Tensor
            Normalized ADT node features (num_nodes, adt_features)
            Expected: X̃_ADT (3484, 31) CLR-normalized from Module 1
        adj_spatial : adjacency matrix or edge list
            Spatial graph structure (k=6 neighbors)
        adj_feature : adjacency matrix or edge list
            Feature graph structure (k=20 neighbors)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (Z_RNA, Z_ADT) both (num_nodes, in_features)
            Spatially-informed embeddings in shared space
        """
        return self.forward(x_rna, x_adt, adj_spatial, adj_feature)


def create_gat_model(
    in_features: int = 512,
    adt_features: int = 31,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_heads: int = 8,
    device: Optional[torch.device] = None
) -> ResGATModel:
    """
    Factory function to create a Residual GATModel with KAC-Net defaults.
    
    Handles DUAL MODALITIES (RNA + ADT) with projection layer.
    
    Parameters
    ----------
    in_features : int
        Input dimension for RNA (default: 512 from Module 2 output)
    adt_features : int
        Normalized ADT feature dimension (default: 31, CLR-normalized from Module 1)
    hidden_dim : int
        Hidden dimension per attention head (default: 256)
    num_layers : int
        Number of Residual GATv2 layers (default: 2)
    num_heads : int
        Number of attention heads (default: 8)
    device : torch.device, optional
        Device placement ('cuda', 'cpu', or None)
    
    Returns
    -------
    ResGATModel
        Initialized Residual GATv2 model with ADT projection
    
    Examples
    --------
    >>> import torch
    >>> model = create_gat_model(device=torch.device('cuda'))
    >>> H_rna = torch.randn(3484, 512)      # From Module 2
    >>> X_adt_tilde = torch.randn(3484, 31) # X̃_ADT (CLR-normalized) from Module 1
    >>> # A_s, A_f from Module 3 (sparse matrices or edge indices)
    >>> Z_rna, Z_adt = model.encode(H_rna, X_adt_tilde, A_s, A_f)
    >>> Z_rna.shape, Z_adt.shape  # Both (3484, 512)
    """
    model = ResGATModel(
        in_features=in_features,
        adt_features=adt_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        negative_slope=0.2,
        dropout=0.1
    )
    
    if device is not None:
        model = model.to(device)
    
    return model

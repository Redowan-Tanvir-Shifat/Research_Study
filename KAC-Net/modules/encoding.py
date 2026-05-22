"""
KAC-Net Module 2: Knowledge-Enriched Encoding
==============================================

Implements transformer-based biological encoder to recover dropout signals 
from gene expression data using learned co-expression patterns.

Classes:
    - MultiHeadAttention: Multi-head self-attention mechanism
    - TransformerEncoderLayer: Single transformer layer with attention + FFN
    - TransformerModel: Full transformer encoder for gene expression

Architecture:
    - Multi-head self-attention (allows model to attend to different gene relationships)
    - Feed-forward networks (expand representational capacity)
    - Layer normalization (stabilize training)
    - Residual connections (enable deep architectures)

Mathematical Foundation:
    Attention: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k))·V
    Multi-head: Multiple attention heads in parallel
    Transformer: Stack of attention + FFN layers with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for gene expression encoding.
    
    Allows the model to attend to different aspects of gene relationships
    simultaneously through multiple parallel attention heads.
    
    Mathematical Formulation
    ========================
    For each head h:
        Q_h = X·W_Q^h     [Query projection]
        K_h = X·W_K^h     [Key projection]
        V_h = X·W_V^h     [Value projection]
        
        Attention_h(Q_h, K_h, V_h) = softmax(Q_h·K_h^T / sqrt(d_k))·V_h
    
    Multi-head concatenation:
        MultiHeadAttention = Concat(Attention_1, ..., Attention_h)·W_O
    
    Where h = num_heads (typically 8 or 16)
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Parameters
        ----------
        embed_dim : int
            Dimension of embeddings (must be divisible by num_heads)
        num_heads : int
            Number of parallel attention heads
        dropout : float
            Dropout rate for attention weights
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scale factor: 1/sqrt(d_k)
        
        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch, seq_len, embed_dim)
        key : torch.Tensor, optional
            Key tensor. If None, uses query (self-attention)
        value : torch.Tensor, optional
            Value tensor. If None, uses query (self-attention)
        mask : torch.Tensor, optional
            Attention mask to prevent attending to certain positions
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (output, attention_weights) where:
            - output: (batch, seq_len, embed_dim)
            - attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        # (batch, seq_len, embed_dim) → (batch, seq_len, num_heads, head_dim)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        # (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights via softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        # (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        # (batch, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.embed_dim)
        
        # Final output projection
        output = self.out_proj(context)
        
        # Average attention weights across heads for interpretability
        attn_weights_avg = attn_weights.mean(dim=1)
        
        return output, attn_weights_avg


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) in transformer.
    
    Applies two linear transformations with activation:
        FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
    
    The first layer expands to hidden_dim, then contracts back.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 2048,
        dropout: float = 0.1
    ):
        """
        Parameters
        ----------
        embed_dim : int
            Input/output dimension
        hidden_dim : int
            Intermediate hidden dimension (typically 2-4x embed_dim)
        dropout : float
            Dropout rate
        """
        super(FeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim)
        
        Returns
        -------
        torch.Tensor
            Output of same shape
        """
        x = self.linear1(x)
        x = F.relu(x)  # ReLU activation
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    
    Components:
        1. Multi-head self-attention
        2. Position-wise feed-forward network
        3. Layer normalization (applied before each sub-layer)
        4. Residual connections
    
    Architecture:
        Input → LayerNorm → MultiHeadAttention → Residual Add
             → LayerNorm → FeedForwardNetwork → Residual Add
             → Output
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1
    ):
        """
        Parameters
        ----------
        embed_dim : int
            Embedding dimension
        num_heads : int
            Number of attention heads
        hidden_dim : int
            Hidden dimension in FFN
        dropout : float
            Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim)
        mask : torch.Tensor, optional
            Attention mask
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (output, attention_weights)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, mask=mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = x + ffn_output  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        return x, attn_weights


class TransformerModel(nn.Module):
    """
    Complete Transformer Encoder for Knowledge-Enriched Gene Expression Encoding.
    
    Purpose:
        Recover dropout signals and infer missing gene expression values using
        learned co-expression patterns from multi-head attention.
    
    Architecture:
        1. Gene embedding projection (18,085 genes → embed_dim)
        2. Stack of N transformer encoder layers
        3. Output projection to final dimension (512)
    
    Mechanism:
        - Each gene attends to all other genes via self-attention
        - Multi-head attention captures different regulatory relationships:
            * Head 1: Pathway-level co-regulation
            * Head 2: Cell-type specific markers
            * Head 3: Developmental stage indicators
            * etc.
        - Residual connections and layer norm enable deep networks
        - FFN expands capacity for non-linear representations
    
    Mathematical Flow
    =================
    Input: X_RNA ∈ ℝ^(batch × 18085)  [Raw gene counts]
        ↓
    Gene Embedding: E = X_RNA·W_e ∈ ℝ^(batch × embed_dim)
        ↓
    Transformer Layers (stack of N):
        For each layer l in 1 to N:
            Attention: A_l = MultiHeadAttention(E_{l-1})
            E_l = FFN(A_l) + E_{l-1}  [with residuals and norm]
        ↓
    Output Projection: H_RNA = E_N·W_o ∈ ℝ^(batch × 512)
        ↓
    Output: H_RNA ∈ ℝ^(batch × 512)  [Enriched embedding]
    """
    
    def __init__(
        self,
        input_dim: int = 18085,
        embed_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Parameters
        ----------
        input_dim : int
            Input feature dimension (number of genes). Default: 18,085
        embed_dim : int
            Embedding dimension for transformer layers. Default: 512
        output_dim : int
            Output dimension (enriched embedding). Default: 512
        num_layers : int
            Number of transformer encoder layers. Default: 6
        num_heads : int
            Number of attention heads. Default: 8
        hidden_dim : int
            Hidden dimension in FFN. Default: 2048 (4x embed_dim)
        dropout : float
            Dropout rate. Default: 0.1
        activation : str
            Activation function ('relu' or 'gelu'). Default: 'relu'
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection: raw genes → embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Stack of transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection: embed_dim → output_dim
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input gene expression of shape (batch_size, input_dim)
            where input_dim = 18,085 (number of genes)
        mask : torch.Tensor, optional
            Attention mask. Shape: (batch_size, 1, seq_len, seq_len) or
            (batch_size, num_heads, seq_len, seq_len)
        return_attention : bool
            If True, also return attention weights from each layer
        
        Returns
        -------
        torch.Tensor
            Output enriched embedding of shape (batch_size, output_dim)
            If return_attention=True, returns tuple (output, attention_list)
        """
        # Input projection
        # (batch_size, 18085) → (batch_size, embed_dim)
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Process through transformer layers
        attention_weights = [] if return_attention else None
        
        for layer in self.encoder_layers:
            x, attn_w = layer(x, mask=mask)
            if return_attention:
                attention_weights.append(attn_w)
        
        # Output projection
        # (batch_size, embed_dim) → (batch_size, output_dim)
        output = self.output_proj(x)
        
        if return_attention:
            return output, attention_weights
        else:
            return output
    
    def encode(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Encode gene expression to enriched embedding.
        
        Alias for forward() for semantic clarity.
        
        Parameters
        ----------
        x : torch.Tensor
            Raw gene counts of shape (n_cells, 18085)
        return_attention : bool
            If True, return attention weights
        
        Returns
        -------
        torch.Tensor
            Enriched embedding (n_cells, 512)
        """
        return self.forward(x, return_attention=return_attention)


def create_transformer_model(
    input_dim: int = 18085,
    embed_dim: int = 512,
    output_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    device: torch.device = None
) -> TransformerModel:
    """
    Factory function to create a TransformerModel with default KAC-Net settings.
    
    Parameters
    ----------
    input_dim : int
        Number of input genes (default: 18,085 for lymph node data)
    embed_dim : int
        Transformer embedding dimension (default: 512)
    output_dim : int
        Output enriched embedding dimension (default: 512)
    num_layers : int
        Number of transformer layers (default: 6)
    num_heads : int
        Number of attention heads (default: 8)
    device : torch.device, optional
        Device to move model to ('cuda' or 'cpu')
    
    Returns
    -------
    TransformerModel
        Initialized transformer model
    
    Examples
    --------
    >>> model = create_transformer_model(device=torch.device('cuda'))
    >>> X_enriched = model.encode(X_normalized)  # Input: (3484, 18085)
                                                  # Output: (3484, 512)
    """
    model = TransformerModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=embed_dim * 4,
        dropout=0.1
    )
    
    if device is not None:
        model = model.to(device)
    
    return model

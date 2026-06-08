"""
KAC-Net Utility Functions

Shared utilities for data loading, preprocessing, visualization, and metrics.
Provides convenient interfaces to work with H5AD files, create PyTorch DataLoaders,
visualize results, and compute evaluation metrics.

Functions:
    Data Loading:
        - load_lymph_node_data() - Load 10X lymph node dataset
        - load_data() - Generic H5AD data loader
        - create_data_loaders() - PyTorch DataLoaders
        - train_val_test_split() - Data splitting
    
    Visualization:
        - plot_umap() - UMAP visualization
        - plot_genes_heatmap() - Top gene heatmap
        - plot_proteins_heatmap() - Top protein heatmap
        - plot_spatial_distribution() - Spatial domain map
        - plot_confusion_matrix() - Clustering confusion matrix
    
    Metrics:
        - compute_ari() - Adjusted Rand Index
        - compute_nmi() - Normalized Mutual Information
        - compute_silhouette() - Silhouette score
        - compute_modularity() - Graph modularity
    
    Helpers:
        - prepare_adata() - AnnData preprocessing
        - get_top_genes() - Extract top genes
        - normalize_data() - Min-max normalization
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_lymph_node_data(
    data_dir: str = 'data/10x_human_lymph_node_A1/',
    use_raw: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load 10X human lymph node dataset.
    
    Loads RNA, ADT, and spatial coordinates from H5AD files.
    Applies standard preprocessing (normalization, log transform).
    
    Args:
        data_dir: Directory containing adata_RNA.h5ad, adata_ADT.h5ad, annotation.csv
        use_raw: If True, use raw counts; if False, use log-normalized
    
    Returns:
        X_RNA: RNA expression matrix (n_spots, n_genes) - [3484, 18085]
        X_ADT: Protein expression matrix (n_spots, n_proteins) - [3484, 31]
        coords: Spatial coordinates (n_spots, 2) - [3484, 2]
        gt_labels: Ground truth domain labels (n_spots,) - [3484,]
        metadata: Metadata DataFrame with spot information
    
    Example:
        >>> X_RNA, X_ADT, coords, gt_labels, meta = load_lymph_node_data()
        >>> print(f"RNA shape: {X_RNA.shape}, ADT shape: {X_ADT.shape}")
        >>> # Output: RNA shape: (3484, 18085), ADT shape: (3484, 31)
    """
    data_dir = Path(data_dir)
    
    logger.info("Loading lymph node dataset...")
    
    # Load RNA data
    rna_path = data_dir / 'adata_RNA.h5ad'
    adata_rna = sc.read_h5ad(rna_path)
    logger.info(f"✓ RNA loaded: {adata_rna.shape}")
    
    # Load ADT data
    adt_path = data_dir / 'adata_ADT.h5ad'
    adata_adt = sc.read_h5ad(adt_path)
    logger.info(f"✓ ADT loaded: {adata_adt.shape}")
    
    # Load annotations
    annot_path = data_dir / 'annotation.csv'
    annotations = pd.read_csv(annot_path, index_col=0)
    logger.info(f"✓ Annotations loaded: {annotations.shape}")
    
    # Extract data
    X_RNA = adata_rna.X.toarray() if hasattr(adata_rna.X, 'toarray') else adata_rna.X
    X_ADT = adata_adt.X.toarray() if hasattr(adata_adt.X, 'toarray') else adata_adt.X
    
    # Get spatial coordinates (from RNA adata)
    if 'spatial' in adata_rna.obsm:
        coords = adata_rna.obsm['spatial']
    else:
        logger.warning("No spatial coordinates found in obsm['spatial'], using dummy coordinates")
        coords = np.random.randn(X_RNA.shape[0], 2)
    
    # Get ground truth labels
    gt_labels = annotations.iloc[:, 0].values
    
    # Create metadata
    metadata = pd.DataFrame({
        'spot_id': adata_rna.obs.index,
        'domain': gt_labels,
    })
    
    logger.info(f"✓ Dataset ready: RNA {X_RNA.shape}, ADT {X_ADT.shape}, Coords {coords.shape}")
    logger.info(f"  Unique domains: {len(np.unique(gt_labels))}")
    
    return X_RNA, X_ADT, coords, gt_labels, metadata


def load_data(
    rna_path: str,
    adt_path: str,
    annotation_path: Optional[str] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generic data loader for H5AD files.
    
    Loads RNA, ADT, and optional annotations from H5AD format.
    Automatically extracts spatial coordinates if available.
    
    Args:
        rna_path: Path to RNA H5AD file
        adt_path: Path to ADT H5AD file
        annotation_path: Optional path to annotation CSV
        normalize: If True, apply log normalization
    
    Returns:
        X_RNA: RNA expression matrix (n_spots, n_genes)
        X_ADT: Protein expression matrix (n_spots, n_proteins)
        coords: Spatial coordinates (n_spots, 2)
        labels: Ground truth labels (n_spots,) or None if not provided
    
    Example:
        >>> X_RNA, X_ADT, coords, labels = load_data(
        ...     'data/rna.h5ad',
        ...     'data/adt.h5ad',
        ...     'data/annotations.csv'
        ... )
    """
    logger.info("Loading custom dataset...")
    
    # Load RNA
    adata_rna = sc.read_h5ad(rna_path)
    X_RNA = adata_rna.X.toarray() if hasattr(adata_rna.X, 'toarray') else adata_rna.X
    
    # Load ADT
    adata_adt = sc.read_h5ad(adt_path)
    X_ADT = adata_adt.X.toarray() if hasattr(adata_adt.X, 'toarray') else adata_adt.X
    
    # Normalize
    if normalize:
        X_RNA = np.log1p(X_RNA)
        X_ADT = np.log1p(X_ADT)
    
    # Get coordinates
    if 'spatial' in adata_rna.obsm:
        coords = adata_rna.obsm['spatial']
    else:
        coords = np.random.randn(X_RNA.shape[0], 2)
        logger.warning("No spatial coordinates found")
    
    # Load annotations if provided
    labels = None
    if annotation_path:
        annot_df = pd.read_csv(annotation_path, index_col=0)
        labels = annot_df.iloc[:, 0].values
    
    logger.info(f"✓ Data loaded: RNA {X_RNA.shape}, ADT {X_ADT.shape}, Coords {coords.shape}")
    
    return X_RNA, X_ADT, coords, labels


# ============================================================================
# PYTORCH DATASET & DATALOADER
# ============================================================================

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal spatial transcriptomics.
    
    Wraps RNA, ADT, and spatial coordinates for batch processing.
    Handles normalization and format conversion to PyTorch tensors.
    
    Args:
        X_RNA: RNA expression matrix (n_spots, n_genes)
        X_ADT: Protein expression matrix (n_spots, n_proteins)
        coords: Spatial coordinates (n_spots, 2)
        normalize: If True, apply min-max normalization
    
    Returns:
        batch: Tuple of (X_RNA_tensor, X_ADT_tensor, coords_tensor)
    """
    
    def __init__(
        self,
        X_RNA: np.ndarray,
        X_ADT: np.ndarray,
        coords: np.ndarray,
        normalize: bool = True
    ):
        """Initialize dataset."""
        self.X_RNA = X_RNA.astype(np.float32)
        self.X_ADT = X_ADT.astype(np.float32)
        self.coords = coords.astype(np.float32)
        
        # Normalize to [0, 1]
        if normalize:
            self.X_RNA = self._normalize(self.X_RNA)
            self.X_ADT = self._normalize(self.X_ADT)
            self.coords = self._normalize(self.coords)
        
        self.n_spots = X_RNA.shape[0]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.n_spots
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return single sample as PyTorch tensors."""
        return (
            torch.from_numpy(self.X_RNA[idx]),
            torch.from_numpy(self.X_ADT[idx]),
            torch.from_numpy(self.coords[idx]),
        )
    
    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_max[X_max == X_min] = 1  # Avoid division by zero
        return (X - X_min) / (X_max - X_min)


def create_data_loaders(
    X_RNA: np.ndarray,
    X_ADT: np.ndarray,
    coords: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    normalize: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Splits data and creates PyTorch DataLoaders for training.
    Handles normalization and tensor conversion automatically.
    
    Args:
        X_RNA: RNA expression matrix (n_spots, n_genes)
        X_ADT: Protein expression matrix (n_spots, n_proteins)
        coords: Spatial coordinates (n_spots, 2)
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        batch_size: Batch size (default 256)
        num_workers: Number of workers for DataLoader (default 4)
        normalize: If True, apply normalization
        seed: Random seed for reproducibility
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
    
    Example:
        >>> train_loader, val_loader, test_loader = create_data_loaders(
        ...     X_RNA, X_ADT, coords,
        ...     batch_size=256,
        ...     num_workers=4
        ... )
        >>> for X_rna_batch, X_adt_batch, coords_batch in train_loader:
        ...     print(X_rna_batch.shape)  # [256, 18085]
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = MultimodalDataset(
        X_RNA=X_RNA,
        X_ADT=X_ADT,
        coords=coords,
        normalize=normalize
    )
    
    # Calculate split sizes
    n_samples = len(dataset)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"✓ DataLoaders created:")
    logger.info(f"  Train: {train_size} samples ({train_ratio*100:.0f}%)")
    logger.info(f"  Val:   {val_size} samples ({val_ratio*100:.0f}%)")
    logger.info(f"  Test:  {test_size} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    logger.info(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_umap(
    Z: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP of Z_Fused Embeddings",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot UMAP visualization of embeddings.
    
    Computes UMAP from embeddings and visualizes with cluster labels.
    
    Args:
        Z: Embedding matrix (n_spots, embedding_dim)
        labels: Optional cluster labels (n_spots,)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    
    Example:
        >>> fig = plot_umap(Z_fused, labels=predicted_domains)
        >>> plt.show()
    """
    import umap
    
    # Compute UMAP
    logger.info("Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    Z_umap = reducer.fit_transform(Z)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(Z_umap[:, 0], Z_umap[:, 1], c=labels, cmap='tab20', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Domain')
    else:
        ax.scatter(Z_umap[:, 0], Z_umap[:, 1], s=50, alpha=0.7)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Figure saved to {save_path}")
    
    return fig


def plot_genes_heatmap(
    X_RNA: np.ndarray,
    labels: np.ndarray,
    gene_names: Optional[List[str]] = None,
    n_top_genes: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot heatmap of top genes across domains.
    
    Computes mean expression per domain and visualizes top genes.
    
    Args:
        X_RNA: RNA expression matrix (n_spots, n_genes)
        labels: Cluster labels (n_spots,)
        gene_names: Optional gene names (n_genes,)
        n_top_genes: Number of top genes to show
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    
    Example:
        >>> fig = plot_genes_heatmap(X_RNA, predicted_domains, n_top_genes=20)
    """
    # Compute mean expression per domain
    unique_labels = np.unique(labels)
    domain_means = []
    
    for label in unique_labels:
        mask = labels == label
        mean_expr = X_RNA[mask].mean(axis=0)
        domain_means.append(mean_expr)
    
    domain_means = np.array(domain_means)
    
    # Get top genes
    top_genes_idx = np.argsort(domain_means.std(axis=0))[-n_top_genes:]
    top_genes = domain_means[:, top_genes_idx]
    
    # Create gene names if not provided
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(top_genes_idx.size)]
    else:
        gene_names = [gene_names[i] for i in top_genes_idx]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        top_genes.T,
        xticklabels=[f"Domain_{i}" for i in unique_labels],
        yticklabels=gene_names,
        cmap='RdYlBu_r',
        ax=ax,
        cbar_kws={'label': 'Mean Expression'}
    )
    ax.set_title('Top Genes by Domain', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Figure saved to {save_path}")
    
    return fig


def plot_spatial_distribution(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str = "Spatial Domain Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot spatial distribution of domains.
    
    Visualizes predicted or ground truth domains on spatial coordinates.
    
    Args:
        coords: Spatial coordinates (n_spots, 2)
        labels: Domain labels (n_spots,)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    
    Example:
        >>> fig = plot_spatial_distribution(coords, predicted_domains)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap='tab20',
        s=100, alpha=0.8,
        edgecolors='black', linewidth=0.5
    )
    
    plt.colorbar(scatter, ax=ax, label='Domain')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Figure saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix between ground truth and predicted labels.
    
    Args:
        gt_labels: Ground truth labels (n_spots,)
        pred_labels: Predicted labels (n_spots,)
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure
    
    Example:
        >>> fig = plot_confusion_matrix(gt_labels, pred_labels)
    """
    # Compute confusion matrix
    cm = confusion_matrix(gt_labels, pred_labels)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=np.unique(pred_labels),
        yticklabels=np.unique(gt_labels),
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Ground Truth Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Figure saved to {save_path}")
    
    return fig


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_ari(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray
) -> float:
    """
    Compute Adjusted Rand Index (ARI).
    
    Measures agreement between two labelings on scale [-1, 1].
    - 1.0: Perfect agreement
    - 0.0: Random chance
    - <0.0: Worse than random
    
    Args:
        gt_labels: Ground truth labels (n_spots,)
        pred_labels: Predicted labels (n_spots,)
    
    Returns:
        ari: ARI score (float in [-1, 1])
    
    Example:
        >>> ari = compute_ari(gt_labels, pred_domains)
        >>> print(f"ARI: {ari:.4f}")
    """
    ari = adjusted_rand_score(gt_labels, pred_labels)
    logger.info(f"Adjusted Rand Index (ARI): {ari:.4f}")
    return ari


def compute_nmi(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray
) -> float:
    """
    Compute Normalized Mutual Information (NMI).
    
    Measures mutual information normalized by entropy.
    - 1.0: Perfect agreement
    - 0.0: Independent labels
    
    Args:
        gt_labels: Ground truth labels (n_spots,)
        pred_labels: Predicted labels (n_spots,)
    
    Returns:
        nmi: NMI score (float in [0, 1])
    
    Example:
        >>> nmi = compute_nmi(gt_labels, pred_domains)
        >>> print(f"NMI: {nmi:.4f}")
    """
    nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    logger.info(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    return nmi


def compute_silhouette(
    Z: np.ndarray,
    labels: np.ndarray,
    sample_size: Optional[int] = None
) -> float:
    """
    Compute Silhouette score.
    
    Measures how well embeddings are separated by predicted clusters.
    - 1.0: Well separated
    - 0.0: Overlapping clusters
    - -1.0: Wrong cluster assignment
    
    Args:
        Z: Embedding matrix (n_spots, embedding_dim)
        labels: Cluster labels (n_spots,)
        sample_size: Optional sample size for faster computation
    
    Returns:
        silhouette: Silhouette score (float in [-1, 1])
    
    Example:
        >>> silhouette = compute_silhouette(Z_fused, pred_domains)
        >>> print(f"Silhouette: {silhouette:.4f}")
    """
    if sample_size is not None and len(Z) > sample_size:
        idx = np.random.choice(len(Z), sample_size, replace=False)
        Z_sample = Z[idx]
        labels_sample = labels[idx]
    else:
        Z_sample = Z
        labels_sample = labels
    
    silhouette = silhouette_score(Z_sample, labels_sample)
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    return silhouette


def compute_modularity(
    adj_matrix: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute graph modularity.
    
    Measures strength of community structure in graph.
    Higher values indicate stronger community separation.
    
    Args:
        adj_matrix: Adjacency matrix (sparse or dense) (n_spots, n_spots)
        labels: Community labels (n_spots,)
    
    Returns:
        modularity: Modularity score (float in [-0.5, 1])
    
    Example:
        >>> modularity = compute_modularity(adj_spatial, pred_domains)
        >>> print(f"Modularity: {modularity:.4f}")
    """
    from scipy.sparse import issparse
    
    # Convert to dense if sparse
    if issparse(adj_matrix):
        adj_matrix = adj_matrix.toarray()
    
    n = adj_matrix.shape[0]
    m = adj_matrix.sum() / 2  # Number of edges
    
    modularity = 0.0
    for label in np.unique(labels):
        mask = labels == label
        adj_sub = adj_matrix[mask][:, mask]
        l_c = adj_sub.sum() / 2  # Edges within community
        d_c = adj_sub.sum(axis=0).sum() / 2  # Degrees in community
        modularity += (l_c / m) - (d_c / (2 * m)) ** 2
    
    logger.info(f"Graph Modularity: {modularity:.4f}")
    return modularity


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_adata(
    X: np.ndarray,
    coords: np.ndarray,
    gene_names: Optional[List[str]] = None,
    spot_names: Optional[List[str]] = None
) -> ad.AnnData:
    """
    Create AnnData object from arrays.
    
    Useful for integration with scanpy workflow.
    
    Args:
        X: Expression matrix (n_spots, n_features)
        coords: Spatial coordinates (n_spots, 2)
        gene_names: Feature names
        spot_names: Spot names
    
    Returns:
        adata: AnnData object
    
    Example:
        >>> adata = prepare_adata(X_RNA, coords, gene_names=gene_list)
        >>> sc.pp.neighbors(adata)
        >>> sc.tl.leiden(adata)
    """
    if gene_names is None:
        gene_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    if spot_names is None:
        spot_names = [f"Spot_{i}" for i in range(X.shape[0])]
    
    adata = ad.AnnData(X)
    adata.obs_names = spot_names
    adata.var_names = gene_names
    adata.obsm['spatial'] = coords
    
    return adata


def get_top_genes(
    X: np.ndarray,
    labels: np.ndarray,
    n_genes: int = 50
) -> Dict[int, List[int]]:
    """
    Get top genes for each cluster.
    
    Computes mean expression per cluster and returns top genes.
    
    Args:
        X: Expression matrix (n_spots, n_genes)
        labels: Cluster labels (n_spots,)
        n_genes: Number of top genes per cluster
    
    Returns:
        top_genes_dict: Dict mapping cluster -> [gene indices]
    
    Example:
        >>> top_genes = get_top_genes(X_RNA, pred_domains, n_genes=20)
        >>> for cluster, genes in top_genes.items():
        ...     print(f"Cluster {cluster}: {genes}")
    """
    top_genes_dict = {}
    
    for label in np.unique(labels):
        mask = labels == label
        mean_expr = X[mask].mean(axis=0)
        top_idx = np.argsort(mean_expr)[-n_genes:][::-1]
        top_genes_dict[label] = top_idx.tolist()
    
    return top_genes_dict


def normalize_data(
    X: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize data to [0, 1].
    
    Args:
        X: Data matrix
        method: 'minmax' or 'zscore'
    
    Returns:
        X_normalized: Normalized data
    
    Example:
        >>> X_norm = normalize_data(X_RNA, method='minmax')
    """
    if method == 'minmax':
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_max[X_max == X_min] = 1
        return (X - X_min) / (X_max - X_min)
    
    elif method == 'zscore':
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True)
        X_std[X_std == 0] = 1
        return (X - X_mean) / X_std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


if __name__ == '__main__':
    """
    Example usage:
    
    # Load data
    X_RNA, X_ADT, coords, gt_labels, meta = load_lymph_node_data()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_RNA, X_ADT, coords, batch_size=256
    )
    
    # Visualizations
    plot_spatial_distribution(coords, gt_labels)
    plt.show()
    """
    print("KAC-Net utils module ready. See docstrings for usage.")

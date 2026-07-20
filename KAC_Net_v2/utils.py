"""
KAC-Net v2 — utils.py
Evaluation & Visualization utilities containing:
  - Leiden/Louvain clustering
  - Adjusted Rand Index (ARI) score computation
  - Screen resolution search (optimal cluster search)
  - Visual plotting routines: UMAP, Spatial Domains, Gate Weights, Loss Curves
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from umap import UMAP


# ---------------------------------------------------------------------------
# Clustering & Validation
# ---------------------------------------------------------------------------

def clustering(embedding, resolution=0.5, n_neighbors=50, random_seed=42, method='leiden'):
    """
    Perform spatial clustering on learned fused embeddings.
    
    Parameters
    ----------
    embedding   : np.ndarray of shape (N, D)
    resolution  : float                      –  resolution parameter
    n_neighbors : int                        –  number of KNN neighbors
    random_seed : int                        –  reproducibility seed
    method      : str                        –  'leiden' or 'louvain'
    
    Returns
    -------
    clusters    : list of shape (N,)
    """
    adata = sc.AnnData(embedding)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    
    if method.lower() == 'leiden':
        sc.tl.leiden(adata, resolution=float(resolution), random_state=random_seed)
        return adata.obs['leiden'].values.astype(int)
    elif method.lower() == 'louvain':
        sc.tl.louvain(adata, resolution=float(resolution), random_state=random_seed)
        return adata.obs['louvain'].values.astype(int)
    else:
        raise ValueError(f"Unknown clustering method: {method}. Use 'leiden' or 'louvain'.")


def compute_ari(ground_truth_labels, predicted_clusters):
    """
    Compute the Adjusted Rand Index (ARI) metric.
    """
    return adjusted_rand_score(ground_truth_labels, predicted_clusters)


def search_res(embedding, ground_truth_labels, target_clusters=None, start=0.1, end=1.5, step=0.05, n_neighbors=50, random_seed=42):
    """
    Screen resolutions to locate the highest-performing partition.
    If `target_clusters` is provided, seeks the resolution returning that count.
    
    Parameters
    ----------
    embedding       : np.ndarray
    ground_truth_labels : list/numpy array of true categories
    target_clusters : int or None
        Optional. If provided, returns the best resolution matching this number of clusters.
    """
    best_ari = -1.0
    best_res = 0.5
    best_clusters = None
    
    resolutions = np.arange(start, end, step)
    for res in resolutions:
        pred = clustering(embedding, resolution=res, n_neighbors=n_neighbors, random_seed=random_seed, method='leiden')
        n_clusters = len(np.unique(pred))
        ari = compute_ari(ground_truth_labels, pred)
        
        print(f"Resolution: {res:.2f} | ARI: {ari:.4f} | Cluster Count: {n_clusters}")
        
        if target_clusters is not None:
            if n_clusters == target_clusters:
                return res, ari, pred
        else:
            if ari > best_ari:
                best_ari = ari
                best_res = res
                best_clusters = pred
                
    if target_clusters is not None:
        print(f"[KAC-Net Warning] Target cluster count {target_clusters} not found. Returning max ARI.")
        
    return best_res, best_ari, best_clusters


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_spatial_domains(adata, cluster_key, spatial_coords_key='spatial', save_path=None):
    """
    Plot spatial domains mapping clustered regions back to physical slide coordinates.
    """
    coords = adata.obsm[spatial_coords_key]
    clusters = adata.obs[cluster_key]
    
    plt.figure(figsize=(8, 6))
    unique_clusters = np.unique(clusters)
    
    # Use color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for idx, cluster in enumerate(unique_clusters):
        mask = (clusters == cluster)
        plt.scatter(
            coords[mask, 0], coords[mask, 1], 
            label=f"Domain {cluster}", 
            color=colors[idx], 
            s=25, alpha=0.9
        )
        
    plt.title(f"KAC-Net Spatial Domains: {cluster_key}", fontsize=14)
    plt.xlabel("Spatial X", fontsize=12)
    plt.ylabel("Spatial Y", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_yaxis()  # Match Visium orientation
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_umap(embedding, labels, title="KAC-Net Latent UMAP", save_path=None):
    """
    Render 2D UMAP projections of learned multi-omics representations.
    """
    labels = np.array(labels)
    print("[KAC-Net] Fitting UMAP representation...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(embedding)
    
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, lbl in enumerate(unique_labels):
        mask = (labels == lbl)
        plt.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            label=str(lbl),
            color=colors[idx],
            s=15, alpha=0.8
        )
        
    plt.title(title, fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_modality_weights(gate_weights, save_path=None):
    """
    Visualize how KAC-Net blends RNA vs ADT across different spots.
    RNA gate weights 'g' are represented as heatmaps.
    """
    plt.figure(figsize=(8, 6))
    
    # Gate weight for RNA is (N, D). Let's take the mean across the dimensions per spot
    if len(gate_weights.shape) > 1:
        gate_mean = np.mean(gate_weights, axis=1)
    else:
        gate_mean = gate_weights
        
    plt.hist(gate_mean, bins=40, color='purple', alpha=0.7, edgecolor='black')
    plt.title("Distribution of RNA Modality Weights (Gate g)", fontsize=14)
    plt.xlabel("RNA Weight (g = 1 indicates pure RNA, g = 0 pure ADT)", fontsize=12)
    plt.ylabel("Spot Count", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_loss_curve(loss_history, save_path=None):
    """
    Plot loss optimization curve across epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='teal', linewidth=2, label='Total Loss')
    plt.title("KAC-Net Convergence Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Objective Loss Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

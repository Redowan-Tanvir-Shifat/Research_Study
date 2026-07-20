"""
KAC-Net v2 — graph_builder.py
Module 3: Multi-Graph Construction

Builds:
  1. Spatial Graph (G_s) using Euclidean KNN (k=6) on coordinates.
  2. Feature Graph (G_f) using Cosine KNN (k=20) on concatenated RNA + ADT features.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import torch


def construct_graph_by_coordinate(cell_position, n_neighbors=6):
    """
    Construct spatial graph edges based on Euclidean distance of coordinates.
    Returns a pandas DataFrame containing source (x), target (y) indices, and weights.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean').fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    return pd.DataFrame({'x': x, 'y': y, 'value': np.ones(x.size)})


def transform_adjacent_matrix(adjacent_df, n_spots):
    """Convert adjacency dataframe to a scipy coo_matrix."""
    return sp.coo_matrix(
        (adjacent_df['value'], (adjacent_df['x'], adjacent_df['y'])),
        shape=(n_spots, n_spots)
    )


def preprocess_graph(adj):
    """
    Apply GCN-style symmetric normalization: D^{-1/2} (A + I) D^{-1/2}
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_mx_to_torch_edge_list(sparse_mx):
    """Convert scipy sparse matrix to PyTorch Geometric style edge_index [2, E]."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_index = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_index


def build_all_graphs(adata_rna, adata_adt, H_RNA, n_spatial_neighbors=6, n_feature_neighbors=20):
    """
    Module 3 – Construct Multi-Graph Lattices.
    
    Parameters
    ----------
    adata_rna : AnnData
        RNA AnnData. Must contain 'spatial' in adata_rna.obsm.
    adata_adt : AnnData
        Normalized ADT AnnData.
    H_RNA : np.ndarray
        Biological foundation model embedding (shape: spots x 512).
    n_spatial_neighbors : int
        Number of neighbors for spatial Euclidean graph (default 6 for Visium grids).
    n_feature_neighbors : int
        Number of neighbors for feature Cosine graph (default 20).
        
    Returns
    -------
    graphs_dict : dict
        A dictionary containing:
        - 'adj_spatial': PyG edge_index for spatial G_s
        - 'adj_feature': PyG edge_index for feature G_f
        - 'adj_spatial_tensor': Normalized PyTorch sparse tensor for G_s (used in Laplaican loss)
    """
    n_spots = adata_rna.shape[0]
    cell_coords = adata_rna.obsm['spatial']

    # 1. Spatial Graph (G_s)
    df_spatial = construct_graph_by_coordinate(cell_coords, n_neighbors=n_spatial_neighbors)
    adj_spatial = transform_adjacent_matrix(df_spatial, n_spots)
    
    # Make spatial adjacency symmetric (undirected)
    adj_spatial = adj_spatial + adj_spatial.T
    adj_spatial.data = np.where(adj_spatial.data > 1, 1.0, adj_spatial.data)
    
    # Keep normalized version for GNN message aggregation/Laplacian regularizer
    adj_spatial_norm = preprocess_graph(adj_spatial)
    
    # 2. Feature Graph (G_f)
    # Concatenate enriched RNA + normalized ADT
    X_ADT = adata_adt.X.toarray() if sp.issparse(adata_adt.X) else np.array(adata_adt.X)
    concat_features = np.hstack((H_RNA, X_ADT))
    
    # KNN using Cosine Similarity (metric='correlation' / 'cosine')
    adj_feature = kneighbors_graph(
        concat_features, 
        n_neighbors=n_feature_neighbors, 
        mode='connectivity', 
        metric='cosine', 
        include_self=False
    )
    
    # Make feature adjacency symmetric
    adj_feature = adj_feature + adj_feature.T
    adj_feature.data = np.where(adj_feature.data > 1, 1.0, adj_feature.data)
    
    adj_feature_norm = preprocess_graph(adj_feature)

    # Convert to PyTorch tensors
    edge_index_spatial = sparse_mx_to_torch_edge_list(adj_spatial_norm)
    edge_index_feature = sparse_mx_to_torch_edge_list(adj_feature_norm)
    
    adj_spatial_tensor = sparse_mx_to_torch_sparse_tensor(adj_spatial_norm)

    return {
        'adj_spatial': edge_index_spatial,
        'adj_feature': edge_index_feature,
        'adj_spatial_tensor': adj_spatial_tensor
    }

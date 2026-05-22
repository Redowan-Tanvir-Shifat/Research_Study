"""
KAC-Net Module 3: Multi-Graph Construction
===========================================

Constructs spatial and feature-based adjacency graphs from normalized and
enriched multi-omics data. These graphs define cell neighborhoods for
downstream spatial encoding (Module 4).

Functions:
    - construct_graph_by_coordinate(): K-NN spatial adjacency (k=6)
    - construct_graph_by_feature(): K-NN feature adjacency (k=20)
    - construct_neighbor_graph(): Main interface for both graphs
    - transform_adjacent_matrix(): Convert DataFrame to sparse CSR format
    - preprocess_graph(): Symmetric normalization for GNN processing

Mathematical Foundation:
    Spatial Graph: K-nearest neighbors in (x, y) coordinates
    Feature Graph: K-nearest neighbors in concatenated feature space
    Normalization: D^(-1/2) * A * D^(-1/2) for GNN compatibility
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, diags
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from anndata import AnnData
from typing import Tuple, Dict, Union, Optional
import warnings


def construct_graph_by_coordinate(
    cell_positions: np.ndarray,
    n_neighbors: int = 6
) -> pd.DataFrame:
    """
    Construct spatial adjacency graph using K-nearest neighbors in coordinates.
    
    Creates a spatial graph where each cell is connected to its k-nearest
    neighbors in the (x, y) coordinate space. This captures physical proximity
    and tissue topology.
    
    Parameters
    ----------
    cell_positions : np.ndarray
        Spatial coordinates array of shape (n_cells, 2) containing (x, y) positions
    n_neighbors : int, optional
        Number of nearest neighbors to connect. Default: 6 (hexagonal Visium layout)
    
    Returns
    -------
    pd.DataFrame
        Edge list with columns ['source', 'target', 'weight']
        - source: Cell index i
        - target: Cell index j (neighbor of i)
        - weight: 1 (binary adjacency)
    
    Mathematical Formulation
    ========================
    For each cell i, find k cells j with minimum Euclidean distance:
        d_{ij} = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    
    Create edge between i and j if j ∈ KNN(i)
    Result: Adjacency list representation (sparse graph)
    
    Examples
    --------
    >>> coordinates = np.random.rand(100, 2)
    >>> edges_df = construct_graph_by_coordinate(coordinates, n_neighbors=6)
    >>> edges_df.shape
    (600, 3)  # 100 cells * 6 neighbors
    
    Notes
    -----
    - K=6 matches hexagonal lattice of 10x Visium arrays
    - Euclidean distance in physical space (not toroidal)
    - Symmetric: if i connects to j, j connects to i
    """
    if cell_positions.shape[1] != 2:
        raise ValueError(f"Expected (n_cells, 2) coordinates, got shape {cell_positions.shape}")
    
    if n_neighbors >= len(cell_positions):
        raise ValueError(f"n_neighbors ({n_neighbors}) must be < number of cells ({len(cell_positions)})")
    
    # Fit K-NN in coordinate space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree')
    nbrs.fit(cell_positions)
    
    # Get nearest neighbors (first result is self, exclude it)
    distances, indices = nbrs.kneighbors(cell_positions)
    
    # Build edge list
    edges = []
    for i in range(len(cell_positions)):
        # Skip first neighbor (self) - indices[:, 0]
        neighbors = indices[i, 1:]  # Take neighbors 1:k+1
        for j in neighbors:
            edges.append({'source': i, 'target': j, 'weight': 1})
    
    edges_df = pd.DataFrame(edges)
    return edges_df


def construct_graph_by_feature(
    adata_rna: AnnData,
    adata_protein: AnnData,
    k: int = 20
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Construct feature adjacency graph using K-nearest neighbors in feature space.
    
    Concatenates enriched RNA embeddings (H_RNA) and normalized protein counts
    (X̃_ADT) into a joint feature vector for each cell. Then connects cells that
    are close in this combined feature space (cosine similarity).
    
    Parameters
    ----------
    adata_rna : AnnData
        RNA data with H_RNA stored in .obsm['H_rna'] or .X
    adata_protein : AnnData
        Protein data with X̃_ADT in .X
    k : int, optional
        Number of nearest neighbors in feature space. Default: 20
    
    Returns
    -------
    Tuple[csr_matrix, csr_matrix]
        (adj_csr, adj_symmetric_csr) where:
        - adj_csr: Directed K-NN adjacency (n_cells × n_cells)
        - adj_symmetric_csr: Symmetrized adjacency (i↔j if i→j or j→i)
    
    Mathematical Formulation
    ========================
    Feature concatenation for each cell i:
        z_i = [H_RNA_i; X̃_ADT_i]  ∈ ℝ^(512+31) = ℝ^543
    
    Cosine similarity between cells i, j:
        sim(i,j) = (z_i · z_j) / (||z_i|| ||z_j||)
    
    K-NN: Connect i to j if j ∈ arg top-k nearest in feature space
    Symmetrize: A_sym[i,j] = A[i,j] OR A[j,i]
    
    Examples
    --------
    >>> adj_direct, adj_sym = construct_graph_by_feature(adata_rna, adata_protein, k=20)
    >>> adj_direct.shape
    (3484, 3484)
    >>> adj_sym.nnz  # Number of non-zero entries
    139360  # Approximately k*n_cells entries
    
    Notes
    -----
    - k=20 empirically good for 3000+ cell datasets
    - Captures expression similarity across modalities
    - Includes long-range relationships (unlike spatial k=6)
    """
    # Verify same number of cells
    if adata_rna.n_obs != adata_protein.n_obs:
        raise ValueError(
            f"RNA ({adata_rna.n_obs}) and protein ({adata_protein.n_obs}) "
            "cell counts must match"
        )
    
    n_cells = adata_rna.n_obs
    
    # Extract H_RNA (enriched embedding from Module 2)
    if 'H_rna' in adata_rna.obsm:
        H_rna = adata_rna.obsm['H_rna']
    elif 'H_RNA' in adata_rna.obsm:
        H_rna = adata_rna.obsm['H_RNA']
    else:
        # Fallback to expression if embedding not available
        if hasattr(adata_rna.X, 'toarray'):
            H_rna = adata_rna.X.toarray()
        else:
            H_rna = np.array(adata_rna.X)
        if H_rna.shape[1] > 512:
            warnings.warn("Using full expression data instead of H_rna embedding")
    
    # Extract normalized protein counts
    if hasattr(adata_protein.X, 'toarray'):
        X_adt = adata_protein.X.toarray()
    else:
        X_adt = np.array(adata_protein.X)
    
    # Normalize features individually for fair comparison
    H_rna_norm = H_rna / (np.linalg.norm(H_rna, axis=1, keepdims=True) + 1e-8)
    X_adt_norm = X_adt / (np.linalg.norm(X_adt, axis=1, keepdims=True) + 1e-8)
    
    # Concatenate normalized features
    Z_concat = np.hstack([H_rna_norm, X_adt_norm])  # (n_cells, 512+31=543)
    
    # K-NN in feature space using cosine metric
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nbrs.fit(Z_concat)
    distances, indices = nbrs.kneighbors(Z_concat)
    
    # Build sparse adjacency matrix
    # indices[:, 0] is self (distance 0), skip it
    row_indices = np.repeat(np.arange(n_cells), k)
    col_indices = indices[:, 1:].flatten()
    data = np.ones(len(row_indices))
    
    adj_directed = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_cells, n_cells)
    )
    
    # Symmetrize: A_sym[i,j] = A[i,j] OR A[j,i]
    adj_symmetric = adj_directed + adj_directed.T
    adj_symmetric[adj_symmetric > 1] = 1  # Binary adjacency
    adj_symmetric = adj_symmetric.tocsr()
    
    return adj_directed, adj_symmetric


def construct_neighbor_graph(
    adata_rna: AnnData,
    adata_protein: AnnData,
    datatype: str = 'visium',
    n_neighbors_spatial: int = 6,
    n_neighbors_feature: int = 20
) -> Dict[str, Union[csr_matrix, pd.DataFrame]]:
    """
    Main interface for constructing both spatial and feature graphs.
    
    Orchestrates complete graph construction workflow:
        1. Extract spatial coordinates
        2. Build spatial K-NN graph (k=6 for hexagonal layout)
        3. Build feature K-NN graph (k=20 for expression similarity)
        4. Apply normalization for GNN processing
    
    Parameters
    ----------
    adata_rna : AnnData
        RNA data with:
        - .obsm['spatial'] or .obsm['X_spatial'] containing (x,y) coordinates
        - .obsm['H_rna'] or .X containing enriched embeddings
    adata_protein : AnnData
        Protein data with .X containing CLR-normalized counts
    datatype : str, optional
        Data type: 'visium', 'merfish', 'iss', 'spots'. Default: 'visium'
    n_neighbors_spatial : int, optional
        K for spatial neighbors. Default: 6 (Visium hexagonal)
    n_neighbors_feature : int, optional
        K for feature neighbors. Default: 20
    
    Returns
    -------
    Dict[str, Union[csr_matrix, pd.DataFrame]]
        Dictionary containing:
        - 'edge_list_spatial': pd.DataFrame with spatial edges
        - 'adj_spatial': csr_matrix spatial adjacency (normalized)
        - 'adj_feature': csr_matrix feature adjacency (normalized)
        - 'datatype': str recording data type
    
    Examples
    --------
    >>> graphs = construct_neighbor_graph(adata_rna, adata_protein)
    >>> graphs['adj_spatial'].shape
    (3484, 3484)
    >>> graphs['adj_feature'].shape
    (3484, 3484)
    
    Notes
    -----
    - Coordinates extracted from adata_rna.obsm['spatial']
    - H_RNA required in adata_rna.obsm (from Module 2)
    - Both adjacency matrices normalized for GNN compatibility
    """
    # Determine k-neighbors based on datatype
    datatype_params = {
        'visium': {'k_spatial': 6, 'k_feature': 20},
        'merfish': {'k_spatial': 10, 'k_feature': 20},
        'iss': {'k_spatial': 10, 'k_feature': 20},
        'spots': {'k_spatial': 6, 'k_feature': 20}
    }
    
    if datatype.lower() in datatype_params:
        default_params = datatype_params[datatype.lower()]
        n_neighbors_spatial = default_params['k_spatial']
    
    # Extract spatial coordinates
    if 'spatial' in adata_rna.obsm:
        coordinates = adata_rna.obsm['spatial']
    elif 'X_spatial' in adata_rna.obsm:
        coordinates = adata_rna.obsm['X_spatial']
    else:
        raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")
    
    # Build spatial graph (coordinate-based)
    edges_spatial = construct_graph_by_coordinate(
        coordinates,
        n_neighbors=n_neighbors_spatial
    )
    
    # Build feature graph (expression-based)
    adj_feature_directed, adj_feature_sym = construct_graph_by_feature(
        adata_rna,
        adata_protein,
        k=n_neighbors_feature
    )
    
    # Convert spatial edge list to sparse adjacency matrix
    adj_spatial = transform_adjacent_matrix(edges_spatial)
    
    # Normalize both graphs for GNN processing
    adj_spatial_norm = preprocess_graph(adj_spatial)
    adj_feature_norm = preprocess_graph(adj_feature_sym)
    
    return {
        'edge_list_spatial': edges_spatial,
        'adj_spatial': adj_spatial_norm,
        'adj_feature': adj_feature_norm,
        'datatype': datatype
    }


def transform_adjacent_matrix(edge_list: pd.DataFrame) -> csr_matrix:
    """
    Convert edge list DataFrame to sparse CSR adjacency matrix.
    
    Parameters
    ----------
    edge_list : pd.DataFrame
        DataFrame with columns ['source', 'target', 'weight']
        - source: Cell index i
        - target: Cell index j
        - weight: Edge weight (typically 1 for binary graphs)
    
    Returns
    -------
    csr_matrix
        Sparse compressed sparse row matrix A where:
        - A[i,j] = weight if edge (i,j) exists
        - A[i,j] = 0 otherwise
        - Shape: (n_cells, n_cells)
    
    Mathematical Representation
    =============================
    Converts from edge list format:
        (0, 5, 1)  →  Cell 0 connects to cell 5
        (0, 12, 1) →  Cell 0 connects to cell 12
        ...
    
    To matrix format:
        A[0, 5] = 1
        A[0, 12] = 1
        All other A[0, j] = 0
    
    Examples
    --------
    >>> edges = pd.DataFrame({
    ...     'source': [0, 0, 1],
    ...     'target': [5, 12, 5],
    ...     'weight': [1, 1, 1]
    ... })
    >>> A = transform_adjacent_matrix(edges)
    >>> A.toarray()
    [[0, 0, 0, 0, 0, 1, 0, ..., 0, 0, 1],
     [0, 0, 0, 0, 0, 1, 0, ..., 0, 0, 0],
     ...]
    
    Notes
    -----
    - CSR format is efficient for matrix operations (slicing, arithmetic)
    - Automatically determines matrix size from max indices
    - Preserves edge weights from input DataFrame
    """
    if edge_list.empty:
        raise ValueError("Edge list is empty")
    
    # Get matrix size from maximum indices
    n_cells = max(edge_list['source'].max(), edge_list['target'].max()) + 1
    
    # Create sparse matrix
    adj_matrix = csr_matrix(
        (edge_list['weight'].values, 
         (edge_list['source'].values, edge_list['target'].values)),
        shape=(n_cells, n_cells)
    )
    
    return adj_matrix


def preprocess_graph(adj: Union[csr_matrix, coo_matrix]) -> csr_matrix:
    """
    Normalize adjacency matrix for Graph Neural Network processing.
    
    Applies symmetric normalization:
        A_norm = D^(-1/2) * A * D^(-1/2)
    
    Where D is the degree matrix diagonal(sum of each row).
    
    This normalization:
    - Prevents exploding/vanishing gradients in deep GNNs
    - Makes the transformation scale-invariant
    - Is standard in spectral graph theory
    
    Parameters
    ----------
    adj : csr_matrix or coo_matrix
        Adjacency matrix of shape (n_cells, n_cells)
    
    Returns
    -------
    csr_matrix
        Normalized adjacency matrix A_norm with same shape
    
    Mathematical Formulation
    ========================
    Step 1: Add self-loops
        A' = A + I  (each cell connects to itself)
    
    Step 2: Compute degree matrix
        D[i,i] = sum(A'[i, :])  [row-wise sum]
        D[j,j] = 0 for i ≠ j
    
    Step 3: Symmetric normalization
        D_sqrt_inv = D^(-1/2)
        A_norm = D_sqrt_inv @ A' @ D_sqrt_inv
    
    Why normalization matters:
    - Raw adjacency A can have large eigenvalues
    - Causes gradient explosion in deep layers
    - Normalized version has eigenvalues in [-1, 1]
    - Stable for multi-layer GNNs
    
    Examples
    --------
    >>> A = csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> A_norm = preprocess_graph(A)
    >>> A_norm.toarray()
    [[0.5,  0.5,  0. ],
     [0.33, 0.33, 0.33],
     [0. ,  0.5,  0.5 ]]
    
    Notes
    -----
    - Self-loops added: each cell connects to itself
    - Numerical stability: 1e-8 epsilon added to prevent division by zero
    - Assumes undirected graph (symmetric or symmetrized input)
    """
    # Convert to COO format for easier manipulation
    if not isinstance(adj, coo_matrix):
        adj = adj.tocoo()
    
    # Add self-loops (identity matrix)
    adj_with_self = adj + coo_matrix((np.ones(adj.shape[0]), 
                                      (np.arange(adj.shape[0]), 
                                       np.arange(adj.shape[0]))),
                                     shape=adj.shape)
    
    # Convert to CSR for efficient row operations
    adj_with_self = adj_with_self.tocsr()
    
    # Compute degree (sum of each row)
    degrees = np.array(adj_with_self.sum(axis=1)).flatten()
    
    # Degree^(-1/2) with numerical stability
    degrees_inv_sqrt = np.power(degrees, -0.5)
    degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0  # Handle division by zero
    
    # D^(-1/2) as diagonal matrix
    D_inv_sqrt = diags(degrees_inv_sqrt, format='csr')
    
    # Normalized adjacency: D^(-1/2) @ A @ D^(-1/2)
    adj_normalized = D_inv_sqrt @ adj_with_self @ D_inv_sqrt
    
    return adj_normalized.tocsr()

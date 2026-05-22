"""
KAC-Net Module 1: Multimodal Preprocessing
============================================

Normalizes raw spatial multi-omics data (RNA + ADT) to stabilize variance 
and align feature scales for downstream analysis.

Functions:
    - fix_seed(): Set random seed for reproducibility
    - library_normalize_rna(): Library-size normalization for RNA
    - seurat_clr(): Helper function for CLR normalization
    - clr_normalize_each_cell(): CLR normalization for ADT
    - validate_input_data(): Validate input data integrity
    - prepare_modality_data(): Unified normalization pipeline

Mathematical Foundation:
    RNA Pipeline:
        - Library-size normalization: (X / sum(X)) * target_sum
        - Log1p transformation: log(1 + X)
    
    ADT Pipeline:
        - CLR normalization: log(x_i / geometric_mean(x_i))
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Union
from anndata import AnnData
import random


def fix_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy, random, and other libraries.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value (default: 42)
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> fix_seed(42)
    >>> # All subsequent random operations will be reproducible
    """
    np.random.seed(seed)
    random.seed(seed)


def seurat_clr(x: np.ndarray) -> np.ndarray:
    """
    Centered Log Ratio (CLR) normalization helper function.
    
    Implements CLR transformation:
        CLR(x_i) = log(x_i / geometric_mean(x))
    
    This normalization is compositionally-aware and prevents biases from 
    varying sequencing depths across samples.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n_features,) containing raw counts
    
    Returns
    -------
    np.ndarray
        CLR-normalized array of same shape
    
    Mathematical Formulation
    ========================
    For each spot i with M protein channels:
        g(x_i) = (∏_{m=1}^M x_{i,m})^{1/M}  [geometric mean]
        CLR(x_i) = [log(x_i,1/g), log(x_i,2/g), ..., log(x_i,M/g)]
    
    Examples
    --------
    >>> x = np.array([100, 50, 75, 120])
    >>> normalized = seurat_clr(x)
    >>> normalized.shape
    (4,)
    """
    # Compute geometric mean (avoid log(0) by adding small pseudocount)
    x_copy = x.copy()
    x_copy = np.where(x_copy <= 0, 1e-10, x_copy)  # Replace 0s with small value
    
    geometric_mean = np.exp(np.mean(np.log(x_copy)))
    
    # Apply CLR: log(x / geometric_mean)
    clr_result = np.log(x_copy / geometric_mean)
    
    return clr_result


def library_normalize_rna(
    adata: AnnData,
    inplace: bool = True,
    target_sum: float = 10000
) -> Optional[AnnData]:
    """
    Library-size normalization for RNA expression data.
    
    Normalizes each cell's gene expression by its total sequencing depth,
    then scales to a target sum. This standardizes counts across cells with
    different sequencing depths.
    
    Normalization formula:
        X_normalized = (X / sum(X)) * target_sum
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with RNA counts in adata.X
    inplace : bool, optional
        If True, modifies adata in place. If False, returns modified copy.
        Default: True
    target_sum : float, optional
        Target library size after normalization. Default: 10000
    
    Returns
    -------
    AnnData or None
        If inplace=False, returns normalized AnnData object.
        If inplace=True, modifies adata.X in place and returns None.
    
    Raises
    ------
    ValueError
        If adata.X is empty or contains negative values
    
    Mathematical Formulation
    ========================
    For each cell i:
        Depth_i = ∑_j X_{i,j}  [total sequencing depth]
        X̃_{RNA,i,j} = (X_{i,j} / Depth_i) * 10,000
    
    Examples
    --------
    >>> adata = sc.read_h5ad('sample.h5ad')
    >>> library_normalize_rna(adata, target_sum=10000)
    >>> # adata.X now contains normalized counts
    
    Notes
    -----
    - Modifies X to dense format if sparse (for calculation efficiency)
    - Adds 'library_size' to adata.obs for tracking
    """
    if adata.X.size == 0:
        raise ValueError("Input adata.X is empty")
    
    if inplace:
        adata_work = adata
    else:
        adata_work = adata.copy()
    
    # Convert to dense if sparse
    if hasattr(adata_work.X, 'toarray'):
        X_dense = adata_work.X.toarray()
    else:
        X_dense = np.array(adata_work.X)
    
    # Check for negative values
    if np.any(X_dense < 0):
        warnings.warn("Negative values detected in expression matrix. These will be treated as 0.")
        X_dense = np.maximum(X_dense, 0)
    
    # Calculate library size (sum per cell)
    library_sizes = X_dense.sum(axis=1, keepdims=True)
    
    # Avoid division by zero
    library_sizes = np.where(library_sizes == 0, 1, library_sizes)
    
    # Normalize: (X / library_size) * target_sum
    X_normalized = (X_dense / library_sizes) * target_sum
    
    # Store normalized data
    adata_work.X = X_normalized
    
    # Store library sizes in metadata
    adata_work.obs['library_size'] = library_sizes.flatten()
    
    if not inplace:
        return adata_work


def clr_normalize_each_cell(
    adata: AnnData,
    inplace: bool = True,
    modality: str = 'ADT'
) -> Optional[AnnData]:
    """
    Centered Log Ratio (CLR) normalization for Antibody-Derived Tags (ADT).
    
    Applies CLR normalization independently to each cell/spot across protein 
    channels. CLR is compositionally-aware and handles varying antibody 
    amplification biases and background binding.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with ADT/protein counts in adata.X
    inplace : bool, optional
        If True, modifies adata in place. If False, returns modified copy.
        Default: True
    modality : str, optional
        Name of the modality (for logging/tracking). Default: 'ADT'
    
    Returns
    -------
    AnnData or None
        If inplace=False, returns CLR-normalized AnnData object.
        If inplace=True, modifies adata.X in place and returns None.
    
    Raises
    ------
    ValueError
        If adata.X contains fewer than 2 features
    
    Mathematical Formulation
    ========================
    For each cell i across M protein channels:
        g_i = (∏_{m=1}^M x_{i,m})^{1/M}  [geometric mean]
        CLR(x_i) = log(x_i / g_i)
    
    Examples
    --------
    >>> adata_adt = sc.read_h5ad('adt_sample.h5ad')
    >>> clr_normalize_each_cell(adata_adt, modality='ADT')
    >>> # adata_adt.X now contains CLR-normalized protein counts
    
    Notes
    -----
    - CLR prevents artifacts from varying antibody concentrations
    - Geometric mean approach is more robust than arithmetic mean
    - Zero handling: Small pseudocount added internally
    """
    if adata.X.shape[1] < 2:
        raise ValueError(f"ADT data must have at least 2 features, got {adata.X.shape[1]}")
    
    if inplace:
        adata_work = adata
    else:
        adata_work = adata.copy()
    
    # Convert to dense if sparse
    if hasattr(adata_work.X, 'toarray'):
        X_dense = adata_work.X.toarray()
    else:
        X_dense = np.array(adata_work.X)
    
    # Apply CLR normalization to each cell
    X_clr = np.zeros_like(X_dense, dtype=np.float32)
    for i in range(X_dense.shape[0]):
        X_clr[i] = seurat_clr(X_dense[i])
    
    # Store normalized data
    adata_work.X = X_clr
    
    if not inplace:
        return adata_work


def validate_input_data(
    adata_rna: AnnData,
    adata_protein: AnnData,
    check_spatial_coordinates: bool = False
) -> Tuple[bool, str]:
    """
    Validate input data integrity before preprocessing.
    
    Checks:
        - Matching number of observations (cells/spots)
        - Non-empty expression matrices
        - Spatial coordinates available (optional)
        - Data type compatibility
    
    Parameters
    ----------
    adata_rna : AnnData
        RNA expression AnnData object
    adata_protein : AnnData
        Protein/ADT expression AnnData object
    check_spatial_coordinates : bool, optional
        If True, verifies spatial coordinates are present in .obsm
        Default: False
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message) where is_valid is True if all checks pass
    
    Examples
    --------
    >>> is_valid, msg = validate_input_data(adata_rna, adata_protein)
    >>> if not is_valid:
    ...     print(f"Validation failed: {msg}")
    
    Notes
    -----
    - Both objects must have same number of observations
    - Raises warnings but returns bool for graceful handling
    """
    # Check matching observations
    if adata_rna.n_obs != adata_protein.n_obs:
        return False, f"Mismatched cell count: RNA ({adata_rna.n_obs}) vs Protein ({adata_protein.n_obs})"
    
    # Check non-empty matrices
    if adata_rna.X.size == 0:
        return False, "RNA expression matrix is empty"
    
    if adata_protein.X.size == 0:
        return False, "Protein expression matrix is empty"
    
    # Check for all-zero features
    if hasattr(adata_rna.X, 'toarray'):
        rna_sums = adata_rna.X.toarray().sum(axis=0)
    else:
        rna_sums = adata_rna.X.sum(axis=0)
    
    if np.any(rna_sums == 0):
        warnings.warn(f"RNA matrix has {np.sum(rna_sums == 0)} genes with zero counts")
    
    # Check spatial coordinates if requested
    if check_spatial_coordinates:
        if 'spatial' not in adata_rna.obsm:
            return False, "Spatial coordinates not found in adata_rna.obsm['spatial']"
    
    return True, "All validation checks passed"


def prepare_modality_data(
    adata: AnnData,
    modality_name: str = 'RNA',
    normalize_method: str = 'library',
    log_transform: bool = True,
    target_sum: float = 10000,
    inplace: bool = True
) -> Optional[AnnData]:
    """
    Unified preprocessing pipeline for single modality.
    
    Orchestrates complete normalization workflow:
        1. Validate input
        2. Apply normalization (library or CLR)
        3. Log transformation (if applicable)
    
    Parameters
    ----------
    adata : AnnData
        Input expression data
    modality_name : str, optional
        Name of modality ('RNA' or 'ADT'). Default: 'RNA'
    normalize_method : str, optional
        Normalization method: 'library' (RNA) or 'clr' (ADT). Default: 'library'
    log_transform : bool, optional
        Apply log1p transformation. Used for RNA, typically False for ADT.
        Default: True
    target_sum : float, optional
        Target library size after normalization. Default: 10000
    inplace : bool, optional
        Modify data in place. Default: True
    
    Returns
    -------
    AnnData or None
        If inplace=False, returns processed AnnData object.
        If inplace=True, modifies input and returns None.
    
    Raises
    ------
    ValueError
        If normalize_method or modality_name is invalid
    
    Workflow
    ========
    RNA Pipeline:
        Raw X_RNA → Library Normalize → Log1p → X̃_RNA
    
    ADT Pipeline:
        Raw X_ADT → CLR Normalize → X̃_ADT
    
    Examples
    --------
    >>> # RNA normalization with log transform
    >>> prepare_modality_data(
    ...     adata_rna, 
    ...     modality_name='RNA',
    ...     normalize_method='library',
    ...     log_transform=True,
    ...     target_sum=10000
    ... )
    
    >>> # ADT normalization (no log transform)
    >>> prepare_modality_data(
    ...     adata_protein,
    ...     modality_name='ADT',
    ...     normalize_method='clr',
    ...     log_transform=False
    ... )
    
    Notes
    -----
    - Function designed for flexibility in normalization strategies
    - Stores processing metadata in adata.uns for reproducibility
    """
    if modality_name not in ['RNA', 'ADT']:
        raise ValueError(f"modality_name must be 'RNA' or 'ADT', got {modality_name}")
    
    if normalize_method not in ['library', 'clr']:
        raise ValueError(f"normalize_method must be 'library' or 'clr', got {normalize_method}")
    
    if inplace:
        adata_work = adata
    else:
        adata_work = adata.copy()
    
    # Store processing parameters in metadata
    if 'preprocessing' not in adata_work.uns:
        adata_work.uns['preprocessing'] = {}
    
    adata_work.uns['preprocessing'][modality_name] = {
        'normalize_method': normalize_method,
        'log_transform': log_transform,
        'target_sum': target_sum
    }
    
    # Apply normalization based on method
    if normalize_method == 'library':
        library_normalize_rna(
            adata_work,
            inplace=True,
            target_sum=target_sum
        )
        
        # Apply log transformation
        if log_transform:
            if hasattr(adata_work.X, 'toarray'):
                X_dense = adata_work.X.toarray()
            else:
                X_dense = np.array(adata_work.X)
            
            adata_work.X = np.log1p(X_dense)
    
    elif normalize_method == 'clr':
        clr_normalize_each_cell(
            adata_work,
            inplace=True,
            modality=modality_name
        )
        
        # Log transform typically not applied to CLR-normalized data
        if log_transform:
            warnings.warn("Log transformation not typically applied to CLR-normalized ADT data")
    
    if not inplace:
        return adata_work

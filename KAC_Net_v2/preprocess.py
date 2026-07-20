"""
KAC-Net v2 — preprocess.py
Module 1 (Multimodal Preprocessing) + Module 2 (Knowledge-Enriched Encoding)

Module 1:
    RNA  → library-size normalization + log1p transform
    ADT  → CLR (Centered Log Ratio) normalization

Module 2 (spaLLM logic):
    scGPT / Geneformer foundation-model encoding  →  H_RNA ∈ R^(N × 512)
    Fallback: PCA to 512 dims via get_mock_embedding() when no checkpoint is available.
"""

import os
import random

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from torch.backends import cudnn
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_seed(seed: int = 2024) -> None:
    """
    Fix all random seeds for full reproducibility.
    Mirrors the pattern used in spaLLM/preprocess.py and SpatialGlue/preprocess.py.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Module 1 – Multimodal Preprocessing
# ---------------------------------------------------------------------------

def normalize_rna(adata_rna, target_sum: float = 1e4, inplace: bool = True):
    """
    Module 1 – RNA pipeline.

    Steps:
        1. Library-size normalization  (sc.pp.normalize_total, target_sum=1e4)
        2. Log1p transformation        (sc.pp.log1p)

    Transforms X_RNA ∈ R^(N × G_raw) → X̃_RNA ∈ R^(N × G_raw).

    Parameters
    ----------
    adata_rna   : AnnData  –  raw RNA count matrix (spots × genes).
    target_sum  : float    –  library-size normalization target (default 1e4).
    inplace     : bool     –  modify adata in place (default True).

    Returns
    -------
    adata_rna (modified in place when inplace=True).
    """
    if not inplace:
        adata_rna = adata_rna.copy()

    sc.pp.normalize_total(adata_rna, target_sum=target_sum)
    sc.pp.log1p(adata_rna)
    return adata_rna


def normalize_adt(adata_adt, inplace: bool = True):
    """
    Module 1 – ADT pipeline.

    Applies Centered Log Ratio (CLR) normalization independently for each spot:
        x̃_{i,m} = ln( x_{i,m} / g(x_i) )
    where g(x_i) is the geometric mean of the protein panel for spot i.

    This is the same CLR implementation used in SpatialGlue and spaLLM.

    Parameters
    ----------
    adata_adt : AnnData  –  raw ADT count matrix (spots × proteins).
    inplace   : bool     –  modify adata in place (default True).

    Returns
    -------
    adata_adt (modified in place when inplace=True).
    """
    if not inplace:
        adata_adt = adata_adt.copy()

    def seurat_clr(x):
        # Seurat-style CLR: ignores zeros in log-sum for the geometric mean
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    X = adata_adt.X.toarray() if sp.issparse(adata_adt.X) else np.array(adata_adt.X)
    adata_adt.X = np.apply_along_axis(seurat_clr, axis=1, arr=X)
    return adata_adt


# ---------------------------------------------------------------------------
# Module 2 – Knowledge-Enriched Encoding  (spaLLM logic)
# ---------------------------------------------------------------------------

def get_scgpt_embedding(
    adata_rna,
    model_dir: str,
    gene_col: str = "index",
    batch_size: int = 64,
    embedding_dim: int = 512,
    device: str = "cuda",
):
    """
    Module 2 – scGPT foundation-model encoding (spaLLM logic).

    Passes the full normalized RNA matrix through a pre-trained scGPT model to
    obtain biologically-enriched embeddings H_RNA ∈ R^(N × embedding_dim).

    This function uses scGPT's embed_data() utility (the same approach as spaLLM).
    Requires:  pip install scgpt

    Parameters
    ----------
    adata_rna     : AnnData  –  *normalized* RNA AnnData (output of normalize_rna).
    model_dir     : str      –  path to the local scGPT pre-trained model directory.
    gene_col      : str      –  column in adata.var that stores gene names ('index'
                                means use adata.var.index).
    batch_size    : int      –  cells per batch during inference (default 64).
    embedding_dim : int      –  expected output dimension (scGPT default = 512).
    device        : str      –  'cuda' or 'cpu' (default 'cuda').

    Returns
    -------
    H_RNA : np.ndarray of shape (N, embedding_dim)
        Dense, biologically-smoothed transcriptomic representation.
    """
    try:
        import scgpt
    except ImportError as exc:
        raise ImportError(
            "scGPT is not installed.  Install it with:\n"
            "  pip install scgpt\n"
            "Or use get_mock_embedding() for a PCA-based fallback."
        ) from exc

    print("[KAC-Net] Running scGPT embedding (Module 2)...")

    cell_embeddings = scgpt.tasks.embed_data(
        adata_rna,
        model_dir=model_dir,
        gene_col=gene_col,
        batch_size=batch_size,
        device=device,
        return_new_adata=False,
    )

    # scGPT stores embeddings in adata_rna.obsm["X_scGPT"]
    H_RNA = adata_rna.obsm["X_scGPT"].astype(np.float32)
    print(f"[KAC-Net] scGPT embedding shape: {H_RNA.shape}")
    return H_RNA


def get_mock_embedding(adata_rna, n_comps: int = 512, random_state: int = 42):
    """
    Module 2 – PCA-based fallback for knowledge-enriched encoding.

    When a scGPT checkpoint is not available, this function compresses the
    normalized RNA matrix to `n_comps` dimensions via PCA, providing a
    structurally equivalent (though biologically shallower) substitute for the
    foundation-model embedding used in spaLLM.

    Parameters
    ----------
    adata_rna    : AnnData  –  *normalized* RNA AnnData (output of normalize_rna).
    n_comps      : int      –  PCA output dimension (default 512, matching scGPT).
    random_state : int      –  PCA random seed for reproducibility.

    Returns
    -------
    H_RNA : np.ndarray of shape (N, n_comps)
    """
    print(f"[KAC-Net] Computing mock (PCA) embedding – dim={n_comps}  (Module 2)")

    X = adata_rna.X.toarray() if sp.issparse(adata_rna.X) else np.array(adata_rna.X)
    # Clip n_comps to the number of features available
    n_comps = min(n_comps, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comps, random_state=random_state)
    H_RNA = pca.fit_transform(X).astype(np.float32)
    print(f"[KAC-Net] Mock embedding shape: {H_RNA.shape}")
    return H_RNA

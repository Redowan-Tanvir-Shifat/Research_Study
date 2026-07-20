"""
KAC-Net v2: Knowledge-Enriched Attentive Contrastive Network
A novel spatial multi-omics integration model combining:
  - spaLLM logic  (scGPT/Geneformer knowledge-enriched encoding)
  - SpatialGlue logic (adaptive dual-attention fusion)
  - COSMOS logic  (cross-modal contrastive alignment via InfoNCE)

Public API:
    from KAC_Net_v2 import (
        normalize_rna, normalize_adt, get_scgpt_embedding, get_mock_embedding,
        build_all_graphs,
        Train_KACNet,
        clustering, compute_ari, search_res,
        plot_spatial_domains, plot_umap, plot_modality_weights, plot_loss_curve,
    )
"""

from .preprocess import (
    fix_seed,
    normalize_rna,
    normalize_adt,
    get_scgpt_embedding,
    get_mock_embedding,
)
from .graph_builder import build_all_graphs
from .trainer import Train_KACNet
from .utils import (
    clustering,
    compute_ari,
    search_res,
    plot_spatial_domains,
    plot_umap,
    plot_modality_weights,
    plot_loss_curve,
)

__all__ = [
    "fix_seed",
    "normalize_rna",
    "normalize_adt",
    "get_scgpt_embedding",
    "get_mock_embedding",
    "build_all_graphs",
    "Train_KACNet",
    "clustering",
    "compute_ari",
    "search_res",
    "plot_spatial_domains",
    "plot_umap",
    "plot_modality_weights",
    "plot_loss_curve",
]

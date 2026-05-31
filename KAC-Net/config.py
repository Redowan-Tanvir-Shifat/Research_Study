"""
KAC-Net Configuration File

Centralized configuration for all hyperparameters, model dimensions, training settings,
and data paths. Modify this file to customize KAC-Net for different datasets or
experimental settings.

Usage:
    from config import get_config
    config = get_config()
    model = create_kac_net(config, device='cuda')
"""

import torch
from pathlib import Path


def get_config(dataset_type: str = 'lymph_node') -> dict:
    """
    Get configuration dictionary for KAC-Net.
    
    Args:
        dataset_type (str): Type of dataset - 'lymph_node' (default) or 'custom'
    
    Returns:
        config (dict): Complete configuration dictionary
    
    Example:
        >>> config = get_config('lymph_node')
        >>> config['learning_rate'] = 5e-4  # Override specific parameter
    """
    
    if dataset_type == 'lymph_node':
        return LYMPH_NODE_CONFIG
    else:
        return CUSTOM_CONFIG


# ============================================================================
# LYMPH NODE DATASET CONFIGURATION (10X Human Lymph Node A1)
# ============================================================================

LYMPH_NODE_CONFIG = {
    # ========== DATA PATHS ==========
    'data': {
        'rna_path': 'data/10x_human_lymph_node_A1/adata_RNA.h5ad',
        'adt_path': 'data/10x_human_lymph_node_A1/adata_ADT.h5ad',
        'spatial_path': None,  # Spatial coordinates extracted from adata_RNA.h5ad
        'annotation_path': 'data/10x_human_lymph_node_A1/annotation.csv',
        'output_dir': 'results/lymph_node/',
        'checkpoint_dir': 'checkpoints/lymph_node/',
    },
    
    # ========== DEVICE ==========
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    
    # ========== DATA DIMENSIONS ==========
    'input_dims': {
        'rna_dim': 18085,
        'adt_dim': 31,
        'spatial_dim': 2,
        'n_spots': 3484,
    },
    
    # ========== MODULE 1: PREPROCESSING ==========
    'preprocessing': {
        'clr_normalize_adt': True,
        'log_transform_rna': True,
        'library_scale_adt': True,
    },
    
    # ========== MODULE 2: ENCODING ==========
    'encoding': {
        'encoding_dim': 512,
        'encoding_layers': 2,
        'encoding_heads': 8,
        'encoding_dropout': 0.1,
    },
    
    # ========== MODULE 3: GRAPH CONSTRUCTION ==========
    'graph_construction': {
        'k_spatial': 6,
        'similarity_metric': 'cosine',
        'n_neighbors_umap': 15,
        'normalize_adjacency': True,
    },
    
    # ========== MODULE 4: SPATIAL ENCODING ==========
    'spatial_encoding': {
        'gat_hidden': 256,
        'latent_dim': 64,
        'n_attention_heads': 4,
        'n_gat_layers': 2,
        'dropout': 0.1,
        'residual_connections': True,
    },
    
    # ========== MODULE 5: CONTRASTIVE ALIGNMENT ==========
    'contrastive_alignment': {
        'embedding_dim': 64,
        'temperature': 0.07,
        'projection_dim': 64,
        'normalize_embeddings': True,
    },
    
    # ========== MODULE 6: DUAL-ATTENTION FUSION ==========
    'dual_attention_fusion': {
        'latent_dim': 64,
        'output_dim': 64,
        'tier1_hidden': 128,
        'tier2_hidden': 64,
        'dropout': 0.1,
        'fusion_type': 'weighted',
    },
    
    # ========== MODULE 7: RECONSTRUCTION ==========
    'reconstruction': {
        'fusion_dim': 64,
        'rna_dim': 18085,
        'adt_dim': 31,
        'decoder_hidden': 512,
        'n_decoder_layers': 3,
        'dropout': 0.1,
        'reconstruct_rna': True,
        'reconstruct_adt': True,
    },
    
    # ========== MODULE 8: CLUSTERING ==========
    'clustering': {
        'leiden_resolution': 1.0,
        'leiden_resolution_start': 0.2,
        'leiden_resolution_end': 2.0,
        'n_resolution_steps': 15,
        'n_neighbors': 15,
        'umap_n_components': 2,
        'umap_min_dist': 0.1,
        'compute_ari': True,
    },
    
    # ========== TRAINING PARAMETERS ==========
    'training': {
        'num_epochs': 50,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'optimizer': 'adam',
        'grad_clip_norm': 1.0,
        'lr_scheduler': 'step',
        'lr_decay_steps': 10,
        'lr_decay_gamma': 0.5,
        'save_checkpoint_freq': 10,
        'early_stopping': False,
        'early_stopping_patience': 20,
    },
    
    # ========== LOSS WEIGHTS ==========
    'losses': {
        'lambda_contrastive': 0.5,
        'lambda_reconstruction': 1.0,
        'lambda_spatial': 0.3,
        'lambda_rna_recon': 1.0,
        'lambda_adt_recon': 1.0,
        'spatial_loss_type': 'graph_laplacian',
    },
    
    # ========== EVALUATION ==========
    'evaluation': {
        'compute_ari': True,
        'compute_nmi': True,
        'compute_modularity': True,
        'n_clusters_expected': 7,
        'compute_silhouette': False,
    },
    
    # ========== LOGGING & VISUALIZATION ==========
    'logging': {
        'verbose': True,
        'log_freq': 10,
        'save_plots': True,
        'plot_freq': 10,
        'plot_dir': 'results/lymph_node/plots/',
        'tensorboard': False,
        'tensorboard_dir': 'runs/lymph_node/',
    },
}


# ============================================================================
# CUSTOM DATASET CONFIGURATION (Template for other datasets)
# ============================================================================

CUSTOM_CONFIG = {
    'data': {
        'rna_path': 'data/custom/rna.h5ad',  # or rna.csv
        'adt_path': 'data/custom/adt.h5ad',  # or adt.csv
        'spatial_path': None,  # Extract from rna_path or provide path if separate
        'annotation_path': 'data/custom/annotation.csv',
        'output_dir': 'results/custom/',
        'checkpoint_dir': 'checkpoints/custom/',
    },
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    
    'input_dims': {
        'rna_dim': 20000,
        'adt_dim': 40,
        'spatial_dim': 2,
        'n_spots': 5000,
    },
    
    'preprocessing': LYMPH_NODE_CONFIG['preprocessing'],
    'encoding': LYMPH_NODE_CONFIG['encoding'],
    'graph_construction': LYMPH_NODE_CONFIG['graph_construction'],
    'spatial_encoding': LYMPH_NODE_CONFIG['spatial_encoding'],
    'contrastive_alignment': LYMPH_NODE_CONFIG['contrastive_alignment'],
    'dual_attention_fusion': LYMPH_NODE_CONFIG['dual_attention_fusion'],
    'reconstruction': LYMPH_NODE_CONFIG['reconstruction'],
    'clustering': LYMPH_NODE_CONFIG['clustering'],
    'training': LYMPH_NODE_CONFIG['training'],
    'losses': LYMPH_NODE_CONFIG['losses'],
    'evaluation': LYMPH_NODE_CONFIG['evaluation'],
    'logging': LYMPH_NODE_CONFIG['logging'],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def update_config(config: dict, **kwargs) -> dict:
    """Update configuration with custom values. Supports nested keys."""
    for key, value in kwargs.items():
        if '.' in key:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value
    return config


def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_keys = [
        'data', 'device', 'input_dims', 'training', 'losses',
        'preprocessing', 'encoding', 'graph_construction',
        'spatial_encoding', 'contrastive_alignment',
        'dual_attention_fusion', 'reconstruction', 'clustering'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if config['input_dims']['rna_dim'] <= 0:
        raise ValueError("rna_dim must be positive")
    if config['input_dims']['adt_dim'] <= 0:
        raise ValueError("adt_dim must be positive")
    if config['input_dims']['n_spots'] <= 0:
        raise ValueError("n_spots must be positive")
    
    total_weight = (
        config['losses']['lambda_contrastive'] +
        config['losses']['lambda_reconstruction'] +
        config['losses']['lambda_spatial']
    )
    if total_weight == 0:
        raise ValueError("Total loss weight cannot be zero")
    
    if config['training']['num_epochs'] <= 0:
        raise ValueError("num_epochs must be positive")
    if config['training']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate must be positive")
    
    if config['spatial_encoding']['latent_dim'] != config['contrastive_alignment']['embedding_dim']:
        raise ValueError(
            "spatial_encoding.latent_dim must equal "
            "contrastive_alignment.embedding_dim"
        )
    
    if config['dual_attention_fusion']['output_dim'] != config['reconstruction']['fusion_dim']:
        raise ValueError(
            "dual_attention_fusion.output_dim must equal "
            "reconstruction.fusion_dim"
        )
    
    print("✅ Configuration validated successfully!")
    return True


def print_config(config: dict, indent: int = 0) -> None:
    """Pretty-print configuration."""
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{'  ' * indent}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{'  ' * indent}{key}: {value}")


def save_config(config: dict, save_path: str) -> None:
    """Save configuration to YAML file. Requires: pip install pyyaml"""
    import yaml
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✅ Configuration saved to {save_path}")


def load_config(load_path: str) -> dict:
    """Load configuration from YAML file. Requires: pip install pyyaml"""
    import yaml
    with open(load_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Configuration loaded from {load_path}")
    return config


if __name__ == '__main__':
    config = get_config('lymph_node')
    validate_config(config)
    print("\n" + "="*80)
    print("KAC-Net Configuration")
    print("="*80 + "\n")
    print_config(config)

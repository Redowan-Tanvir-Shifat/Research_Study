# Config.py: KAC-Net Configuration Guide

## Overview

**Purpose:** Centralized configuration file containing all hyperparameters, model dimensions, training settings, loss weights, and data paths for KAC-Net.

**File Location:** `KAC-Net/config.py`

**Usage:**
```python
from config import get_config, validate_config, update_config

# Load default configuration
config = get_config('lymph_node')

# Validate parameters
validate_config(config)

# Update specific values
config = update_config(config, learning_rate=5e-4, num_epochs=100)

# Use in model
model = create_kac_net(config, device='cuda')
```

---

## Configuration Structure

### **1. Data Paths**

```python
'data': {
    'rna_path': 'data/10x_human_lymph_node_A1/adata_RNA.h5ad',
    'adt_path': 'data/10x_human_lymph_node_A1/adata_ADT.h5ad',
    'spatial_path': None,  # Spatial coordinates extracted from adata_RNA.h5ad
    'annotation_path': 'data/10x_human_lymph_node_A1/annotation.csv',
    'output_dir': 'results/lymph_node/',
    'checkpoint_dir': 'checkpoints/lymph_node/',
}
```

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `rna_path` | RNA expression data (includes spatial coordinates) | H5AD format |
| `adt_path` | ADT protein data | H5AD format |
| `spatial_path` | Spatial coordinates | CSV with x, y columns |
| `annotation_path` | Ground truth domains | CSV with annotations |
| `output_dir` | Save results here | 'results/lymph_node/' |
| `checkpoint_dir` | Save checkpoints | 'checkpoints/lymph_node/' |

---

### **2. Device & Compute**

```python
'device': 'cuda' if torch.cuda.is_available() else 'cpu',
'num_workers': 4,
```

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `device` | 'cuda' or 'cpu' | Auto-detect GPU |
| `num_workers` | DataLoader workers | 4 |

---

### **3. Input Dimensions**

```python
'input_dims': {
    'rna_dim': 18085,              # Number of genes
    'adt_dim': 31,                 # Number of proteins
    'spatial_dim': 2,              # x, y coordinates
    'n_spots': 3484,               # Total spots
}
```

**For Lymph Node Dataset (10X Human):**
- RNA genes: 18,085
- Proteins (ADT): 31
- Spots: 3,484

**To adapt for other datasets:** Update `rna_dim`, `adt_dim`, `n_spots`

---

### **4. Module-Specific Parameters**

#### **Module 1: Preprocessing**
```python
'preprocessing': {
    'clr_normalize_adt': True,     # CLR normalization for proteins
    'log_transform_rna': True,     # log1p for RNA
    'library_scale_adt': True,     # Scale by library size
}
```

#### **Module 2: Knowledge-Enriched Encoding**
```python
'encoding': {
    'encoding_dim': 512,           # spaLLM output dimension
    'encoding_layers': 2,          # Transformer layers
    'encoding_heads': 8,           # Attention heads
    'encoding_dropout': 0.1,       # Dropout rate
}
```

#### **Module 3: Graph Construction**
```python
'graph_construction': {
    'k_spatial': 6,                # k-NN neighbors (spatial)
    'similarity_metric': 'cosine', # Feature graph metric
    'n_neighbors_umap': 15,        # Graph neighbors
    'normalize_adjacency': True,   # Normalize matrices
}
```

#### **Module 4: Spatial Encoding (Residual GATv2)**
```python
'spatial_encoding': {
    'gat_hidden': 256,             # Hidden dimension
    'latent_dim': 64,              # Output Z_RNA, Z_ADT
    'n_attention_heads': 4,        # GATv2 heads
    'n_gat_layers': 2,             # GATv2 layers
    'dropout': 0.1,
    'residual_connections': True,  # Skip connections
}
```

#### **Module 5: Contrastive Alignment**
```python
'contrastive_alignment': {
    'embedding_dim': 64,           # Input from Module 4
    'temperature': 0.07,           # InfoNCE temperature
    'projection_dim': 64,          # Projection dimension
    'normalize_embeddings': True,  # L2 normalize
}
```

#### **Module 6: Dual-Attention Fusion**
```python
'dual_attention_fusion': {
    'latent_dim': 64,              # Input dimension
    'output_dim': 64,              # Z_Fused output
    'tier1_hidden': 128,           # Graph-level gating
    'tier2_hidden': 64,            # Modality-level gating
    'dropout': 0.1,
    'fusion_type': 'weighted',     # 'weighted' or 'concatenation'
}
```

#### **Module 7: Reconstruction & Loss**
```python
'reconstruction': {
    'fusion_dim': 64,              # Input from Module 6
    'rna_dim': 18085,              # Reconstructed RNA
    'adt_dim': 31,                 # Reconstructed ADT
    'decoder_hidden': 512,         # Decoder hidden size
    'n_decoder_layers': 3,         # Decoder layers
    'dropout': 0.1,
    'reconstruct_rna': True,
    'reconstruct_adt': True,
}
```

#### **Module 8: Clustering**
```python
'clustering': {
    'leiden_resolution': 1.0,      # Fixed resolution
    'leiden_resolution_start': 0.2,
    'leiden_resolution_end': 2.0,
    'n_resolution_steps': 15,      # Sweep 15 values
    'n_neighbors': 15,             # k-NN for Leiden
    'umap_n_components': 2,        # 2D visualization
    'umap_min_dist': 0.1,
    'compute_ari': True,           # Compute ARI
}
```

---

### **5. Training Parameters**

```python
'training': {
    'num_epochs': 50,              # Total epochs
    'batch_size': 256,             # Batch size
    'learning_rate': 1e-3,         # Initial LR
    'weight_decay': 1e-5,          # L2 regularization
    'optimizer': 'adam',           # Optimizer type
    'grad_clip_norm': 1.0,         # Gradient clipping
    
    # Learning rate scheduler
    'lr_scheduler': 'step',        # Type: 'step', 'exponential'
    'lr_decay_steps': 10,          # Decay every N epochs
    'lr_decay_gamma': 0.5,         # Decay factor
    
    # Checkpointing
    'save_checkpoint_freq': 10,    # Save every N epochs
    'early_stopping': False,
    'early_stopping_patience': 20,
}
```

| Parameter | Purpose | Default | Recommended |
|-----------|---------|---------|-------------|
| `num_epochs` | Training epochs | 50 | 50-100 |
| `batch_size` | Batch size | 256 | 128-512 |
| `learning_rate` | Initial LR | 1e-3 | 1e-4 to 1e-3 |
| `weight_decay` | L2 regularization | 1e-5 | 1e-5 to 1e-4 |
| `grad_clip_norm` | Gradient clipping | 1.0 | 0.5-1.0 |
| `lr_decay_steps` | LR decay frequency | 10 | 5-15 |
| `lr_decay_gamma` | LR decay factor | 0.5 | 0.3-0.7 |

---

### **6. Loss Weights**

```python
'losses': {
    'lambda_contrastive': 0.5,     # Module 5 weight
    'lambda_reconstruction': 1.0,  # Module 7 weight
    'lambda_spatial': 0.3,         # Spatial regularization
    
    'lambda_rna_recon': 1.0,       # RNA reconstruction
    'lambda_adt_recon': 1.0,       # ADT reconstruction
    
    'spatial_loss_type': 'graph_laplacian',
}
```

**Total Loss Formula:**
```
L_total = 0.5 * L_contrastive
        + 1.0 * L_reconstruction
        + 0.3 * L_spatial
```

**To adjust loss balance:**

| Scenario | Adjust | Reason |
|----------|--------|--------|
| Poor spatial coherence | ↑ `lambda_spatial` | Enforce spatial relationships |
| Poor reconstruction | ↑ `lambda_reconstruction` | Better gene recovery |
| Poor alignment | ↑ `lambda_contrastive` | Stronger RNA-ADT coupling |
| Training instability | ↓ all weights | Reduce gradient magnitude |

---

### **7. Evaluation**

```python
'evaluation': {
    'compute_ari': True,           # Adjusted Rand Index
    'compute_nmi': True,           # Normalized Mutual Information
    'compute_modularity': True,    # Modularity Q
    'n_clusters_expected': 7,      # Expected clusters
    'compute_silhouette': False,   # Silhouette (slow)
}
```

---

### **8. Logging & Visualization**

```python
'logging': {
    'verbose': True,               # Print progress
    'log_freq': 10,                # Log every 10 epochs
    'save_plots': True,            # Save figures
    'plot_freq': 10,               # Plot every 10 epochs
    'plot_dir': 'results/lymph_node/plots/',
    'tensorboard': False,          # TensorBoard logging
    'tensorboard_dir': 'runs/lymph_node/',
}
```

---

## Helper Functions

### **1. `get_config(dataset_type='lymph_node')`**

Get pre-configured settings.

```python
# Lymph node (10X Human)
config = get_config('lymph_node')

# Custom dataset
config = get_config('custom')
```

---

### **2. `validate_config(config)`**

Validate all parameters before training.

```python
config = get_config('lymph_node')
validate_config(config)  # Raises error if invalid
```

**Checks:**
- Required keys present
- Positive dimensions
- Loss weights sum non-zero
- Latent dimensions consistent
- Positive training parameters

---

### **3. `update_config(config, **kwargs)`**

Update specific parameters (supports nested keys).

```python
config = get_config('lymph_node')

# Single updates
config = update_config(
    config,
    learning_rate=5e-4,
    num_epochs=100,
    batch_size=128
)

# Nested updates
config = update_config(
    config,
    **{'training.learning_rate': 5e-4}
)
```

---

### **4. `print_config(config, indent=0)`**

Pretty-print entire configuration.

```python
config = get_config('lymph_node')
print_config(config)

# Output:
# data:
#   rna_path: data/10x_human_lymph_node_A1/adata_RNA.h5ad
#   adt_path: data/10x_human_lymph_node_A1/adata_ADT.h5ad
#   ...
# device: cuda
# input_dims:
#   rna_dim: 18085
#   adt_dim: 31
#   ...
```

---

### **5. `save_config(config, save_path)`**

Save configuration to YAML (requires PyYAML).

```python
config = get_config('lymph_node')
save_config(config, 'results/config_final.yaml')

# Output:
# ✅ Configuration saved to results/config_final.yaml
```

---

### **6. `load_config(load_path)`**

Load configuration from YAML.

```python
config = load_config('results/config_final.yaml')
```

---

## Usage Examples

### **Example 1: Default Setup (Lymph Node)**

```python
from config import get_config, validate_config
from kac_net_main import create_kac_net

# Get configuration
config = get_config('lymph_node')
validate_config(config)

# Create model
model = create_kac_net(config, device='cuda')

# Print configuration
from config import print_config
print_config(config)
```

---

### **Example 2: Custom Dataset**

```python
from config import get_config, update_config, validate_config

# Start with template
config = get_config('custom')

# Update for your dataset
config = update_config(
    config,
    rna_dim=20000,           # Your gene count
    adt_dim=40,              # Your protein count
    n_spots=5000,            # Your spot count
    rna_path='data/my_rna.h5ad',
    adt_path='data/my_adt.h5ad',
    output_dir='results/my_dataset/',
)

validate_config(config)
```

---

### **Example 3: Hyperparameter Tuning**

```python
from config import get_config, update_config

# Test different learning rates
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]

for lr in learning_rates:
    config = get_config('lymph_node')
    config = update_config(config, learning_rate=lr)
    
    model = create_kac_net(config)
    # ... train and evaluate
```

---

### **Example 4: Adjust Loss Weights**

```python
from config import get_config, update_config

# For better spatial coherence
config = get_config('lymph_node')
config = update_config(
    config,
    **{
        'losses.lambda_spatial': 0.5,      # Increase spatial weight
        'losses.lambda_contrastive': 0.3,  # Decrease contrastive
    }
)
```

---

## Customization Guide

### **For Different Datasets**

1. **Modify input dimensions:**
   ```python
   config['input_dims']['rna_dim'] = your_n_genes
   config['input_dims']['adt_dim'] = your_n_proteins
   config['input_dims']['n_spots'] = your_n_spots
   ```

2. **Update data paths:**
   ```python
   config['data']['rna_path'] = 'path/to/your/rna.h5ad'  # or .csv if available
   config['data']['adt_path'] = 'path/to/your/adt.h5ad'  # or .csv if available
   ```

3. **Adjust model capacity:**
   ```python
   # For larger datasets
   config['encoding']['encoding_dim'] = 768
   config['spatial_encoding']['gat_hidden'] = 512
   config['reconstruction']['decoder_hidden'] = 1024
   ```

4. **Tune for speed:**
   ```python
   # Faster training
   config['training']['batch_size'] = 512
   config['training']['num_epochs'] = 30
   config['clustering']['n_resolution_steps'] = 5
   ```

5. **Tune for accuracy:**
   ```python
   # More thorough training
   config['training']['learning_rate'] = 5e-4
   config['training']['num_epochs'] = 100
   config['clustering']['n_resolution_steps'] = 20
   config['losses']['lambda_spatial'] = 0.5
   ```

---

## Default Configuration Summary

| Category | Parameter | Lymph Node Default |
|----------|-----------|-------------------|
| **Data** | RNA genes | 18,085 |
| | Proteins | 31 |
| | Spots | 3,484 |
| **Training** | Epochs | 50 |
| | Batch size | 256 |
| | Learning rate | 1e-3 |
| **Model** | Latent dim | 64 |
| | GATv2 heads | 4 |
| | spaLLM output | 512 |
| **Loss** | λ_contrastive | 0.5 |
| | λ_reconstruction | 1.0 |
| | λ_spatial | 0.3 |
| **Clustering** | Leiden resolution | 1.0 |
| | Resolution sweep | 0.2-2.0 (15 steps) |

---

## Troubleshooting

### **Training Instability**

**Problem:** Loss explodes or becomes NaN

**Solution:**
```python
config = update_config(
    config,
    learning_rate=5e-4,  # Reduce by 2x
    grad_clip_norm=0.5,  # Increase clipping
)
```

### **Poor Clustering Results**

**Problem:** ARI < 0.5

**Solution:**
```python
config = update_config(
    config,
    **{'losses.lambda_spatial': 0.5},  # Emphasize spatial
    num_epochs=100,  # Train longer
)
```

### **Out of Memory**

**Problem:** CUDA out of memory

**Solution:**
```python
config = update_config(
    config,
    batch_size=128,  # Reduce batch size
)
```

### **Slow Training**

**Problem:** Too slow, want faster results

**Solution:**
```python
config = update_config(
    config,
    batch_size=512,  # Larger batches
    num_epochs=30,   # Fewer epochs
    **{'clustering.n_resolution_steps': 5}  # Faster clustering
)
```

---

## Summary

**config.py provides:**
- ✅ Centralized parameter management
- ✅ Pre-configured for lymph node dataset
- ✅ Easy customization for other datasets
- ✅ Validation before training
- ✅ Save/load functionality
- ✅ Helper functions for common tasks

**Use `config.py` to:**
- Reproducibly configure experiments
- Easily switch between datasets
- Systematically tune hyperparameters
- Document experimental settings

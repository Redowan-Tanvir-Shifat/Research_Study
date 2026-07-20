# losses.py Explanation

**File:** `losses.py`  
**Lines:** 550+  
**Purpose:** Centralized, modular loss functions for training  
**Status:** ✅ Production-ready

---

## 📌 Why Do You Need losses.py?

### **The Problem (Without losses.py)**

Loss functions scattered everywhere:
- ❌ Contrastive loss in `modules/contrastive_alignment.py`
- ❌ Reconstruction loss in `modules/reconstruction_loss.py`
- ❌ Spatial loss in `modules/reconstruction_loss.py`
- ❌ Combined loss in `trainer.py`

**Issues:**
- Hard to understand overall loss strategy
- Difficult to experiment with loss weights
- Can't easily swap loss functions
- Loss logic mixed with module logic
- Can't reuse losses for other experiments

### **The Solution (losses.py)**

**Centralized Loss Library:**
```python
# Before (scattered):
# In trainer.py:
L_total = (0.5 * L_cl + 1.0 * L_recon + 0.3 * L_spatial)  # Where are these from?

# After (organized):
from losses import combined_loss
L_total = combined_loss(
    Z_RNA, Z_ADT,
    X_RNA_recon, X_RNA_true,
    X_ADT_recon, X_ADT_true,
    Z_fused, adj_spatial,
    lambda_contrastive=0.5,
    lambda_reconstruction=1.0,
    lambda_spatial=0.3
)
```

**Benefits:**
✅ **Clear Loss Strategy** - All losses in one place  
✅ **Easy Experimentation** - Try different loss functions/weights  
✅ **Modular** - Use individual losses or combine them  
✅ **Swappable** - Replace loss functions without touching model code  
✅ **Reproducible** - Exact loss formulations documented  
✅ **Reusable** - Use same losses for multiple projects  

---

## 🏗️ Loss Functions Overview

### **4 Core Losses**

```
losses.py
├── contrastive_loss() ← Cross-modal alignment (Module 5)
├── reconstruction_loss() ← Data reconstruction (Module 7)
├── spatial_loss() ← Spatial smoothness (Module 7)
└── combined_loss() ← All 3 weighted together

OPTIONAL:
├── kl_divergence_loss() ← VAE regularization
├── wasserstein_loss() ← Robust reconstruction
└── triplet_loss() ← Metric learning
```

---

## 🔍 Core Loss Functions

### **1. `contrastive_loss()` - Cross-Modal Alignment**

**What it does:**
Aligns RNA and ADT embeddings by bringing same-spot pairs close and pushing different-spot pairs far apart.

**Formula (InfoNCE):**
$$L_{cl} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(Z_{\text{RNA},i}, Z_{\text{ADT},i})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(Z_{\text{RNA},i}, Z_{\text{ADT},j})/\tau)} + \log \frac{\exp(\text{sim}(Z_{\text{ADT},i}, Z_{\text{RNA},i})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(Z_{\text{ADT},i}, Z_{\text{RNA},j})/\tau)} \right]$$

where:
- $\text{sim}(a, b) = \frac{a \cdot b}{|a||b|}$ (cosine similarity)
- $\tau$ = temperature parameter (sharpness)

**Parameters:**
- `Z_RNA`: RNA embeddings (batch_size, embedding_dim)
- `Z_ADT`: ADT embeddings (batch_size, embedding_dim)
- `temperature`: Controls softness (default 0.1, typical 0.05-0.2)
- `use_cosine_sim`: Use cosine (True) or dot product (False)

**Behavior:**
- Lower temperature → harder negatives (sharper)
- Higher temperature → easier negatives (softer)

**Example:**
```python
from losses import contrastive_loss

Z_RNA = torch.randn(256, 64)  # RNA embeddings
Z_ADT = torch.randn(256, 64)  # ADT embeddings

# Compute loss with different temperatures
loss_sharp = contrastive_loss(Z_RNA, Z_ADT, temperature=0.05)
loss_soft = contrastive_loss(Z_RNA, Z_ADT, temperature=0.2)

print(f"Sharp (τ=0.05): {loss_sharp:.4f}")
print(f"Soft (τ=0.2):   {loss_soft:.4f}")  # Usually higher
```

**Good for:**
- Aligning modalities in latent space
- Learning shared representations
- Typical in multi-modal models

**Tuning:**
- Lower λ_contrastive if reconstruction is more important
- Higher λ_contrastive for stronger modality alignment
- Typical: 0.3-0.7

---

### **2. `reconstruction_loss()` - Data Reconstruction**

**What it does:**
Penalizes difference between reconstructed and true data.
Ensures embeddings preserve information for reconstruction.

**Formula (MSE):**
$$L_{\text{recon}} = w_{\text{RNA}} \cdot \text{MSE}(\hat{X}_{\text{RNA}}, X_{\text{RNA}}) + w_{\text{ADT}} \cdot \text{MSE}(\hat{X}_{\text{ADT}}, X_{\text{ADT}})$$

**Alternative Formulas:**
- MAE: Less sensitive to outliers
- Huber: Combination of MSE and MAE

**Parameters:**
- `X_RNA_recon`: Reconstructed RNA (batch_size, n_genes)
- `X_RNA_true`: True RNA (batch_size, n_genes)
- `X_ADT_recon`: Reconstructed ADT (batch_size, n_proteins)
- `X_ADT_true`: True ADT (batch_size, n_proteins)
- `rna_weight`, `adt_weight`: Relative importance
- `loss_type`: 'mse', 'mae', or 'huber'

**Example:**
```python
from losses import reconstruction_loss

X_RNA_recon = decoder_rna(Z_fused)  # [256, 18085]
X_ADT_recon = decoder_adt(Z_fused)  # [256, 31]

# MSE loss (default, sensitive to outliers)
loss_mse = reconstruction_loss(
    X_RNA_recon, X_RNA_batch,
    X_ADT_recon, X_ADT_batch,
    loss_type='mse'
)

# MAE loss (robust to outliers)
loss_mae = reconstruction_loss(
    X_RNA_recon, X_RNA_batch,
    X_ADT_recon, X_ADT_batch,
    loss_type='mae'
)

print(f"MSE: {loss_mse:.4f}, MAE: {loss_mae:.4f}")
```

**Good for:**
- Ensuring information preservation
- Preventing information loss during embedding
- Autoencoder-style training

**Tuning:**
- Lower λ_reconstruction if you want unsupervised discovery
- Higher λ_reconstruction for faithful reconstruction
- Typical: 0.8-1.2

**MSE vs MAE vs Huber:**

| Loss | Outliers | Gradient | Use When |
|------|----------|----------|----------|
| MSE | Sensitive | Large at outliers | Data is clean |
| MAE | Robust | Constant | Many outliers |
| Huber | Balanced | Balanced | Mixed |

---

### **3. `spatial_loss()` - Spatial Smoothness**

**What it does:**
Encourages similar embeddings for spatially adjacent spots.
Enforces spatial coherence in learned representation.

**Formula (Graph Laplacian):**
$$L_{\text{spatial}} = \sum_{(i,j) \in E} A_{ij} \cdot \|Z_i - Z_j\|^2$$

where $E$ is the edge set of spatial adjacency graph, $A$ is adjacency matrix.

**Intuition:**
- If spots $i$ and $j$ are spatially adjacent: $A_{ij} = 1$
- Penalizes large differences: $\|Z_i - Z_j\|^2$ is minimized
- Result: Adjacent spots have similar embeddings

**Parameters:**
- `Z`: Embeddings (n_spots, embedding_dim)
- `adj_spatial`: Spatial adjacency matrix (n_spots, n_spots)
- `loss_type`: 'laplacian' (default), 'smoothness'

**Example:**
```python
from losses import spatial_loss
from modules.graph_construction import construct_spatial_graph

# Build spatial adjacency (k-NN from coordinates)
adj_spatial = construct_spatial_graph(coords, k=6)  # 6 nearest neighbors

# Compute spatial loss
Z_fused = model.get_embeddings(loader)  # [3484, 64]
loss_spatial = spatial_loss(Z_fused, adj_spatial)

print(f"Spatial Loss: {loss_spatial:.4f}")
```

**Good for:**
- Preserving spatial structure
- Domain discovery (clusters form coherent regions)
- Avoiding fragmented predictions

**Tuning:**
- Lower λ_spatial if domains are sparse/scattered
- Higher λ_spatial for strong spatial coherence
- Typical: 0.2-0.5

---

### **4. `combined_loss()` - All Three Together**

**What it does:**
Weighted sum of all three loss components. Main loss used during training.

**Formula:**
$$L_{\text{total}} = \lambda_{\text{cl}} \cdot L_{\text{cl}} + \lambda_{\text{recon}} \cdot L_{\text{recon}} + \lambda_{\text{spatial}} \cdot L_{\text{spatial}}$$

**Default Weights (Lymph Node):**
- $\lambda_{\text{cl}} = 0.5$ (moderate cross-modal alignment)
- $\lambda_{\text{recon}} = 1.0$ (strong reconstruction)
- $\lambda_{\text{spatial}} = 0.3$ (moderate spatial smoothness)
- Total: 1.8 (normalized or not doesn't matter, only ratios)

**Parameters:**
- All parameters from the three individual losses
- `lambda_*`: Weights for each component
- `return_components`: If True, returns dict with individual loss values

**Example:**
```python
from losses import combined_loss

# Standard usage (just total loss)
L_total = combined_loss(
    Z_RNA, Z_ADT,
    X_RNA_recon, X_RNA_batch,
    X_ADT_recon, X_ADT_batch,
    Z_fused, adj_spatial,
    lambda_contrastive=0.5,
    lambda_reconstruction=1.0,
    lambda_spatial=0.3
)
L_total.backward()

# With component tracking
L_total, components = combined_loss(
    Z_RNA, Z_ADT,
    X_RNA_recon, X_RNA_batch,
    X_ADT_recon, X_ADT_batch,
    Z_fused, adj_spatial,
    lambda_contrastive=0.5,
    lambda_reconstruction=1.0,
    lambda_spatial=0.3,
    return_components=True
)

print(f"Total: {components['L_total']:.4f}")
print(f"  L_cl: {components['L_cl']:.4f}")
print(f"  L_recon: {components['L_recon']:.4f}")
print(f"  L_spatial: {components['L_spatial']:.4f}")
```

**How to Use in Training:**

```python
from trainer import KACNetTrainer
from losses import combined_loss

class CustomTrainer(KACNetTrainer):
    def train_epoch(self, train_loader):
        for batch in train_loader:
            # Forward pass
            X_RNA_recon, X_ADT_recon, L_cl, L_spatial = self.model(X_RNA, X_ADT, coords)
            
            # Use centralized loss
            L_total = combined_loss(
                Z_RNA, Z_ADT,
                X_RNA_recon, X_RNA_batch,
                X_ADT_recon, X_ADT_batch,
                Z_fused, adj_spatial,
                **self.config['losses']  # Unpack lambda values
            )
            
            # Backward
            L_total.backward()
```

---

## 🎛️ Loss Weight Tuning Guide

### **Understanding Loss Weights**

**Scenario 1: Strong Reconstruction (Most Important)**
```python
combined_loss(
    ...,
    lambda_contrastive=0.3,      # Lower: less strict alignment
    lambda_reconstruction=1.5,    # Higher: preserve data
    lambda_spatial=0.2,           # Lower: allow scattered
)
```
**Use when:** Information preservation is critical

**Scenario 2: Balanced (Default)**
```python
combined_loss(
    ...,
    lambda_contrastive=0.5,       # Moderate alignment
    lambda_reconstruction=1.0,    # Baseline
    lambda_spatial=0.3,           # Moderate smoothness
)
```
**Use when:** All objectives equally important

**Scenario 3: Strong Spatial Coherence**
```python
combined_loss(
    ...,
    lambda_contrastive=0.3,       # Lower: focus on domains
    lambda_reconstruction=0.8,    # Lower: allow noise
    lambda_spatial=0.7,           # Much higher: strong spatial structure
)
```
**Use when:** Finding compact, spatially coherent domains

**Scenario 4: Strong Cross-Modal Alignment**
```python
combined_loss(
    ...,
    lambda_contrastive=1.0,       # Much higher: align modalities
    lambda_reconstruction=0.8,    # Lower: less strict
    lambda_spatial=0.3,           # Moderate: preserve space
)
```
**Use when:** Multi-modal integration is critical

### **Tuning Process**

1. **Start with defaults**
   ```python
   λ_cl=0.5, λ_recon=1.0, λ_spatial=0.3
   ```

2. **Monitor loss components** during training
   ```python
   L_total, components = combined_loss(..., return_components=True)
   print(f"L_cl: {components['L_cl']:.4f}, "
         f"L_recon: {components['L_recon']:.4f}, "
         f"L_spatial: {components['L_spatial']:.4f}")
   ```

3. **Adjust based on results**
   - If L_cl much larger than others → Increase λ_cl
   - If L_recon much larger than others → Increase λ_recon
   - If L_spatial vanishes → Increase λ_spatial

4. **Validate on test set**
   - ARI (how well domains match) → Check contrastive weight
   - Reconstruction error → Check reconstruction weight
   - Domain coherence (modularity) → Check spatial weight

---

## 📈 Optional Advanced Losses

### **1. KL Divergence Loss (VAE Regularization)**

```python
from losses import kl_divergence_loss

# Adds VAE-style regularization
mu, logvar = encoder(Z)
L_kl = kl_divergence_loss(mu, logvar)

L_total = combined_loss(...) + 0.1 * L_kl
```

**When to use:**
- Ensuring smooth latent space
- Preventing posterior collapse
- Sampling from latent space

---

### **2. Wasserstein Loss (Robust Reconstruction)**

```python
from losses import wasserstein_loss

# More robust to outliers than MSE
L_recon = wasserstein_loss(X_recon, X_true, p=2)
```

**When to use:**
- Noisy/outlier-prone data
- Less is known about data distribution

---

### **3. Triplet Loss (Metric Learning)**

```python
from losses import triplet_loss

# Push positive pairs together, negative pairs apart
L_triplet = triplet_loss(anchor, positive, negative, margin=1.0)
```

**When to use:**
- Learning discriminative embeddings
- When contrastive loss not sufficient

---

## 🔧 Helper Functions

### **`compute_loss_weights()` - Extract from Config**

```python
from losses import compute_loss_weights
from config import get_config

config = get_config('lymph_node')
lambda_cl, lambda_recon, lambda_spatial = compute_loss_weights(config)

print(f"λ_cl={lambda_cl}, λ_recon={lambda_recon}, λ_spatial={lambda_spatial}")
```

---

### **`log_loss_components()` - Pretty Logging**

```python
from losses import log_loss_components

loss, components = combined_loss(..., return_components=True)
log_loss_components(components, epoch=25, print_interval=5)

# Output (every 5 epochs):
# Epoch  25 | L_total=0.1623 L_cl=0.0412 L_recon=0.0987 L_spatial=0.0224
```

---

## 📊 Complete Training Example

```python
import torch
from config import get_config
from kac_net_main import create_kac_net
from trainer import KACNetTrainer
from losses import combined_loss, compute_loss_weights
from utils import create_data_loaders, load_lymph_node_data

# ========== SETUP ==========
config = get_config('lymph_node')
device = 'cuda'

# Load data
X_RNA, X_ADT, coords, gt_labels, _ = load_lymph_node_data()
train_loader, val_loader, _ = create_data_loaders(X_RNA, X_ADT, coords)

# Model & trainer
model = create_kac_net(config, device)
trainer = KACNetTrainer(model, config, device=device)

# Extract loss weights from config
lambda_cl, lambda_recon, lambda_spatial = compute_loss_weights(config)

# ========== TRAINING LOOP ==========
for epoch in range(50):
    total_losses = []
    
    for batch_idx, (X_rna, X_adt, coords_batch) in enumerate(train_loader):
        # Forward pass (all 8 modules)
        X_rna_recon, X_adt_recon, L_cl_internal, L_spatial_internal = model(
            X_rna.to(device),
            X_adt.to(device),
            coords_batch.to(device)
        )
        
        # Use centralized loss function
        L_total = combined_loss(
            Z_RNA=...,  # From model internal states
            Z_ADT=...,
            X_RNA_recon=X_rna_recon,
            X_RNA_true=X_rna,
            X_ADT_recon=X_adt_recon,
            X_ADT_true=X_adt,
            Z_fused=...,
            adj_spatial=...,
            lambda_contrastive=lambda_cl,
            lambda_reconstruction=lambda_recon,
            lambda_spatial=lambda_spatial
        )
        
        # Backward
        trainer.optimizer.zero_grad()
        L_total.backward()
        trainer.optimizer.step()
        
        total_losses.append(L_total.item())
    
    avg_loss = np.mean(total_losses)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

print("Training complete!")
```

---

## ✅ Design Principles

1. **Modularity** - Use individual losses independently
2. **Composability** - Combine any losses with any weights
3. **Configurability** - All parameters passed as arguments
4. **Clarity** - Clear formulas and documentation
5. **Extensibility** - Easy to add new loss functions

---

## 🔗 Integration with Other Modules

```
losses.py (loss functions)
    ↓
trainer.py (uses combined_loss in train_epoch)
    ↓
kac_net_main.py (model outputs fed to loss)
    ↓
config.py (provides lambda values)
```

---

## 📋 Quick Reference

| Loss | Purpose | Formula | Lambda |
|------|---------|---------|--------|
| `contrastive_loss()` | Cross-modal alignment | InfoNCE | 0.5 |
| `reconstruction_loss()` | Information preservation | MSE/MAE | 1.0 |
| `spatial_loss()` | Spatial coherence | Laplacian | 0.3 |
| `combined_loss()` | All three | Weighted sum | - |

---

## ✅ Summary

**losses.py provides:**
- ✅ 3 core loss functions (well-researched)
- ✅ Modular, composable design
- ✅ Easy loss weight tuning
- ✅ Optional advanced losses
- ✅ Helper utilities for logging

**Why it matters:**
- Separates loss logic from model/trainer
- Enables systematic experimentation
- Clear, documented loss formulations
- Reusable across projects


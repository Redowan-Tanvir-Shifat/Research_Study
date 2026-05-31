# Module 7: Reconstruction & Regularization (Decoder & Hub)

## Overview

**Purpose:**
Force the neural network to prove it hasn't discarded vital biological information during compression (1024→64 dimensions). Module 7 serves as the training hub where all three loss components are computed: contrastive alignment (Module 5), reconstruction fidelity (this module), and spatial smoothness (this module).

**Core Problem:**
Compressing Z_RNA and Z_ADT (512+512=1024 dims) into Z_Fused (64 dims) represents a 16x compression. This aggressive dimensionality reduction could lose critical biological signals. How do we verify the compressed space still contains meaningful information?

**Solution:**
Module 7 implements a dual-verification system:
1. **Reconstruction Path:** Attempt to rebuild 18,085 RNA genes and 31 proteins from 64-dim latent space
2. **Regularization Path:** Enforce spatial smoothness so biologically similar neighboring spots have similar latent codes
3. **Loss Hub:** Aggregate all three loss types for unified training objective

**Key Characteristics:**
- ✅ **Reconstruction Loss:** MSE between decoded and preprocessed data (Module 1)
- ✅ **Spatial Regularization:** Graph Laplacian smoothness using A_s adjacency
- ✅ **Dual Decoders:** Separate MLPs for RNA (512→18085) and ADT (128→31)
- ✅ **Total Loss Aggregation:** Combines L_cl (Module 5) + L_recon + L_spat
- ✅ **SpatialGlue Compliant:** Follows decoder architecture + custom losses

---

## Complete Architecture

```
INPUT LAYER (From Modules 1, 3, 6):
┌─────────────────────────────────────────────────────────────┐
│  Z_Fused ∈ R^(3484 × 64) - Unified from Module 6            │
│  X̃_RNA ∈ R^(3484 × 18085) - Preprocessed from Module 1     │
│  X̃_ADT ∈ R^(3484 × 31) - Preprocessed from Module 1        │
│  A_s ∈ R^(3484 × 3484) - Spatial graph from Module 3        │
└─────────────────────────────────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
RECONSTRUCTION DECODERS:

RNA DECODER:                    ADT DECODER:
Linear(64→512)                  Linear(64→128)
ReLU                            ReLU
Linear(512→18085)               Linear(128→31)
│                               │
▼                               ▼
X̂_RNA (3484, 18085)             X̂_ADT (3484, 31)

            └───────────┬───────────┘
                        ▼
LOSS COMPUTATION LAYER:

┌──────────────────────────────────────────────────────────┐
│  RECONSTRUCTION LOSS (MSE):                              │
│                                                         │
│  L_recon = (1/N) * [                                   │
│      Σ_i ||X̃_RNA,i - X̂_RNA,i||²                       │
│    + Σ_i ||X̃_ADT,i - X̂_ADT,i||²                       │
│  ]                                                      │
│                                                         │
│  Measures: Accuracy of reconstruction from 64-dim     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  SPATIAL REGULARIZATION LOSS (Graph Laplacian):         │
│                                                         │
│  L_spat = Σ_{i,j} A_{s,ij} * ||Z_Fused,i - Z_Fused,j||² 
│         = trace(Z_Fused^T * L * Z_Fused)               │
│  where L = D - A_s (Laplacian matrix)                  │
│                                                         │
│  Measures: Spatial coherence (penalizes salt-pepper)  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  TOTAL LOSS (All 3 Modules):                            │
│                                                         │
│  L_total = λ_cl * L_cl                                 │
│          + λ_recon * L_recon                           │
│          + λ_spat * L_spat                             │
│                                                         │
│  Where:                                                 │
│    L_cl from Module 5 (contrastive alignment)          │
│    L_recon from Module 7 (reconstruction)              │
│    L_spat from Module 7 (spatial smoothness)           │
│    λ_* are hyperparameters (default: 1.0 each)        │
└──────────────────────────────────────────────────────────┘

            │
            ▼
BACKPROPAGATION:
┌──────────────────────────────────────────────────────────┐
│  Loss flows backward through:                            │
│  ├─ Decoders (Module 7)                                 │
│  ├─ Module 6 (Dual-Attention Fusion)                   │
│  ├─ Module 5 (Contrastive Alignment)                   │
│  ├─ Module 4 (Spatial Encoding)                        │
│  ├─ Module 3 (Graph Construction - no params)          │
│  ├─ Module 2 (Encoding - foundation model frozen)      │
│  └─ Module 1 (Preprocessing - no params)               │
└──────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Reconstruction Loss (MSE)

**Purpose:** Measure how accurately the 64-dim latent space can reconstruct original 18,085-dim RNA and 31-dim ADT data.

$$\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum_{i=1}^{N} \left\|\tilde{\mathbf{X}}_{\text{RNA}, i} - \hat{\mathbf{X}}_{\text{RNA}, i}\right\|^2 + \frac{1}{N}\sum_{i=1}^{N} \left\|\tilde{\mathbf{X}}_{\text{ADT}, i} - \hat{\mathbf{X}}_{\text{ADT}, i}\right\|^2$$

Where:
- $\tilde{\mathbf{X}}_{\text{RNA}, i} \in \mathbb{R}^{18085}$: Preprocessed RNA from Module 1 (ground truth)
- $\hat{\mathbf{X}}_{\text{RNA}, i} \in \mathbb{R}^{18085}$: Reconstructed by RNA decoder
- $\tilde{\mathbf{X}}_{\text{ADT}, i} \in \mathbb{R}^{31}$: Preprocessed ADT from Module 1 (ground truth)
- $\hat{\mathbf{X}}_{\text{ADT}, i} \in \mathbb{R}^{31}$: Reconstructed by ADT decoder
- $N = 3484$: Total number of spots

**Interpretation:**
- Value < 0.1: Excellent reconstruction (latent space preserves information)
- Value 0.1-0.5: Good reconstruction (acceptable information preservation)
- Value > 1.0: Poor reconstruction (significant information loss during compression)

### Spatial Regularization Loss (Graph Laplacian)

**Purpose:** Penalize sudden changes in latent embeddings between physically adjacent spots. Enforces tissue structure coherence.

$$\mathcal{L}_{\text{spat}} = \sum_{i,j} \mathbf{A}_{s,ij} \left\|\mathbf{Z}_{\text{Fused}, i} - \mathbf{Z}_{\text{Fused}, j}\right\|^2$$

**Equivalent Formulation (using Graph Laplacian):**

$$\mathcal{L}_{\text{spat}} = \text{trace}\left(\mathbf{Z}_{\text{Fused}}^T \mathbf{L} \mathbf{Z}_{\text{Fused}}\right)$$

Where:
- $\mathbf{L} = \mathbf{D} - \mathbf{A}_s$ is the Graph Laplacian matrix
- $\mathbf{D}$ is the degree matrix (diagonal, $D_{ii} = \sum_j A_{s,ij}$)
- $\mathbf{A}_s$: Spatial adjacency (k=6 nearest neighbors)
- $\mathbf{Z}_{\text{Fused}} \in \mathbb{R}^{3484 \times 64}$: Latent embeddings

**Computation Breakdown:**

Step 1: Compute degree matrix from adjacency
$$D_i = \sum_{j=1}^{N} A_{s,ij}$$ (number of neighbors for spot $i$)

Step 2: Apply Laplacian to embeddings
$$(\mathbf{L} \mathbf{Z})_i = D_i \mathbf{Z}_i - \sum_j A_{s,ij} \mathbf{Z}_j$$

Step 3: Compute loss (squared norm with original embeddings)
$$\mathcal{L}_{\text{spat}} = \sum_i \mathbf{Z}_i \cdot (\mathbf{L} \mathbf{Z})_i$$

**Interpretation:**
- Value < 0.01: Extremely smooth (might be over-regularized)
- Value 0.01-0.1: Good spatial coherence (tissue structure preserved)
- Value > 0.5: Under-smoothed (too much noise or fragmentation)

### Total Training Loss

**The Complete Optimization Objective:**

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{cl} + \lambda_2 \mathcal{L}_{\text{recon}} + \lambda_3 \mathcal{L}_{\text{spat}}$$

Where:
- $\mathcal{L}_{cl}$: Contrastive alignment loss from Module 5 (InfoNCE)
- $\mathcal{L}_{\text{recon}}$: Reconstruction fidelity loss (MSE)
- $\mathcal{L}_{\text{spat}}$: Spatial regularization loss (Laplacian smoothing)
- $\lambda_1, \lambda_2, \lambda_3$: Balancing hyperparameters (default: 1.0 each)

**Typical Hyperparameter Settings:**
```
λ_cl = 1.0       (primary: cross-modal alignment)
λ_recon = 1.0    (information preservation)
λ_spat = 0.1-0.5 (weak regularization, preserve boundaries)
```

---

## Complete Function Reference

### Class: RNADecoder

**Purpose:** MLP to reconstruct RNA (18,085 genes) from 64-dim latent space.

**Location:** `reconstruction_loss.py`, lines 36-89

**Architecture:**
```
Input: (N, 64)
  ↓
Linear(64 → 512)
  ↓
ReLU
  ↓
Linear(512 → 18085)
  ↓
Output: (N, 18085)
```

**Inputs:**
- `z_fused` ∈ ℝ^(N×64): Fused embeddings from Module 6

**Outputs:**
- `x_rna_hat` ∈ ℝ^(N×18085): Reconstructed RNA counts

**Usage:**
```python
from reconstruction_loss import RNADecoder

decoder_rna = RNADecoder(latent_dim=64, output_dim=18085, hidden_dim=512)
z_fused = torch.randn(3484, 64)
x_rna_hat = decoder_rna(z_fused)
print(x_rna_hat.shape)  # (3484, 18085)
```

---

### Class: ADTDecoder

**Purpose:** MLP to reconstruct ADT (31 proteins) from 64-dim latent space.

**Location:** `reconstruction_loss.py`, lines 92-145

**Architecture:**
```
Input: (N, 64)
  ↓
Linear(64 → 128)
  ↓
ReLU
  ↓
Linear(128 → 31)
  ↓
Output: (N, 31)
```

**Inputs:**
- `z_fused` ∈ ℝ^(N×64): Fused embeddings from Module 6

**Outputs:**
- `x_adt_hat` ∈ ℝ^(N×31): Reconstructed ADT counts

**Usage:**
```python
from reconstruction_loss import ADTDecoder

decoder_adt = ADTDecoder(latent_dim=64, output_dim=31, hidden_dim=128)
z_fused = torch.randn(3484, 64)
x_adt_hat = decoder_adt(z_fused)
print(x_adt_hat.shape)  # (3484, 31)
```

---

### Class: ReconstructionLoss

**Purpose:** MSE loss for data reconstruction fidelity.

**Location:** `reconstruction_loss.py`, lines 148-215

**Mathematical Specification:**
$$L_{\text{recon}} = \text{MSE}(\hat{X}_{\text{RNA}}, \tilde{X}_{\text{RNA}}) + \text{MSE}(\hat{X}_{\text{ADT}}, \tilde{X}_{\text{ADT}})$$

**Key Methods:**

```python
class ReconstructionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """reduction: 'mean' (normalized by N) or 'sum'"""
    
    def forward(self, x_rna_true, x_rna_hat, x_adt_true, x_adt_hat) -> Tensor:
        """
        Compute MSE reconstruction loss for both modalities.
        
        Returns: Scalar loss
        """
```

**Usage:**
```python
from reconstruction_loss import ReconstructionLoss

loss_fn = ReconstructionLoss(reduction='mean')

x_rna_true = torch.randn(3484, 18085)  # From Module 1
x_rna_hat = torch.randn(3484, 18085)   # From decoder
x_adt_true = torch.randn(3484, 31)     # From Module 1
x_adt_hat = torch.randn(3484, 31)      # From decoder

loss_recon = loss_fn(x_rna_true, x_rna_hat, x_adt_true, x_adt_hat)
print(f"Reconstruction loss: {loss_recon.item():.4f}")
```

---

### Class: SpatialRegularizationLoss

**Purpose:** Graph Laplacian smoothness loss for spatial coherence.

**Location:** `reconstruction_loss.py`, lines 218-306

**Mathematical Specification:**
$$L_{\text{spat}} = \sum_{i,j} A_{s,ij} \cdot \|Z_i - Z_j\|^2 = \text{trace}(Z^T L Z)$$

**Key Methods:**

```python
class SpatialRegularizationLoss(nn.Module):
    def __init__(self, normalize=True):
        """normalize: whether to divide by number of edges"""
    
    def forward(self, z_fused, adj_spatial) -> Tensor:
        """
        Compute spatial regularization via graph Laplacian.
        
        Algorithm:
        1. Convert sparse adj_spatial to dense if needed
        2. Compute degree matrix D from adjacency
        3. Compute Laplacian L = D - A_s
        4. Return trace(Z^T * L * Z)
        
        Returns: Scalar loss
        """
```

**Usage:**
```python
from reconstruction_loss import SpatialRegularizationLoss

loss_fn = SpatialRegularizationLoss(normalize=True)

z_fused = torch.randn(3484, 64)  # From Module 6
A_s = torch.randn(3484, 3484)    # From Module 3

loss_spat = loss_fn(z_fused, A_s)
print(f"Spatial loss: {loss_spat.item():.4f}")
```

---

### Class: ReconstructionModule (Main)

**Purpose:** Complete Module 7 with decoders + all losses.

**Location:** `reconstruction_loss.py`, lines 309-465

**Architecture:**
```
Inputs: Z_Fused (64), X̃_RNA (18085), X̃_ADT (31), A_s
  ├─ RNADecoder: 64 → 18085
  ├─ ADTDecoder: 64 → 31
  ├─ ReconstructionLoss: MSE
  └─ SpatialRegularizationLoss: Laplacian
Outputs: X̂_RNA, X̂_ADT, L_recon, L_spat
```

**Key Methods:**

```python
class ReconstructionModule(nn.Module):
    def __init__(self, latent_dim=64, rna_output_dim=18085, 
                 adt_output_dim=31, ...):
        """Initialize with decoders and loss functions"""
    
    def forward(self, z_fused, x_rna_true, x_adt_true, adj_spatial):
        """
        Forward pass through decoders and loss computation.
        
        Returns: (x_rna_hat, x_adt_hat, loss_recon, loss_spat)
        """
    
    def compute_total_loss(self, loss_cl, loss_recon, loss_spat,
                          lambda_cl=1.0, lambda_recon=1.0, lambda_spat=1.0):
        """
        Aggregate all three loss components.
        
        Returns: L_total = λ_cl·L_cl + λ_recon·L_recon + λ_spat·L_spat
        """
    
    def compute_reconstruction_quality(self, x_rna_true, x_rna_hat, 
                                      x_adt_true, x_adt_hat) -> dict:
        """
        Compute reconstruction quality metrics.
        
        Returns: {'rna_mse', 'adt_mse', 'rna_mae', 'adt_mae'}
        """
```

**Full Integration Example:**
```python
import torch
from reconstruction_loss import ReconstructionModule

# Initialize Module 7
module7 = ReconstructionModule(
    latent_dim=64,
    rna_output_dim=18085,
    adt_output_dim=31
)

# Simulate inputs
z_fused = torch.randn(3484, 64)      # From Module 6
x_rna_true = torch.randn(3484, 18085)  # From Module 1
x_adt_true = torch.randn(3484, 31)     # From Module 1
A_s = torch.randn(3484, 3484)          # From Module 3
loss_cl = torch.tensor(2.5)            # From Module 5

# Forward pass
x_rna_hat, x_adt_hat, loss_recon, loss_spat = module7(
    z_fused, x_rna_true, x_adt_true, A_s
)

# Total loss
loss_total = module7.compute_total_loss(
    loss_cl, loss_recon, loss_spat,
    lambda_cl=1.0, lambda_recon=1.0, lambda_spat=0.1
)

# Monitor quality
quality = module7.compute_reconstruction_quality(
    x_rna_true, x_rna_hat, x_adt_true, x_adt_hat
)
print(f"RNA MSE: {quality['rna_mse']:.4f}")
print(f"ADT MSE: {quality['adt_mse']:.4f}")

# Backpropagation
loss_total.backward()
```

---

### Function: create_reconstruction_module

**Purpose:** Factory function for simplified initialization.

**Location:** `reconstruction_loss.py`, lines 468-500

**Signature:**
```python
def create_reconstruction_module(latent_dim=64, rna_output_dim=18085,
                                adt_output_dim=31, device='cpu'):
    """
    Factory for creating ReconstructionModule.
    
    Returns: Initialized and device-placed module
    """
```

**Usage:**
```python
from reconstruction_loss import create_reconstruction_module

# Simple initialization with defaults
module7 = create_reconstruction_module(device='cuda')

# Forward pass
x_rna_hat, x_adt_hat, loss_recon, loss_spat = module7(
    z_fused, x_rna_true, x_adt_true, A_s
)
```

---

## Integration Points

### Input from Module 1 (Preprocessing):
- X̃_RNA: (3484 × 18085) preprocessed RNA (ground truth for reconstruction)
- X̃_ADT: (3484 × 31) preprocessed ADT (ground truth for reconstruction)
  - These are the targets, NOT the raw counts

### Input from Module 3 (Graph Construction):
- A_s: (3484 × 3484) spatial adjacency matrix
  - Used in spatial regularization loss

### Input from Module 5 (Contrastive Alignment):
- L_cl: Contrastive loss scalar
  - Combined with reconstruction and spatial losses

### Input from Module 6 (Dual-Attention Fusion):
- Z_Fused: (3484 × 64) unified embeddings
  - Input to both decoders

### Output for Total Training Objective:
- X̂_RNA, X̂_ADT: Reconstructions for analysis
- L_recon: Reconstruction loss for training
- L_spat: Spatial loss for training
- L_total: Combined loss for backpropagation

### Output to Module 8 (Clustering):
- Z_Fused embeddings (used during inference, not training)

---

## Verification Against Master Pipeline

### ✅ flow.md Compliance (Lines 167-183)

| Requirement | flow.md Detail | Implementation | Status |
|---|---|---|---|
| **Inputs** | "Z_Fused, A_s" | forward(z_fused, ..., adj_spatial) | ✅ |
| **Decoder** | "Parallel MLPs reconstruct RNA & ADT" | RNADecoder + ADTDecoder | ✅ |
| **Regularizer** | "Graph Laplacian Spatial Smoothness" | SpatialRegularizationLoss | ✅ |
| **Loss 1** | "L_recon → Reconstruction Loss (MSE)" | ReconstructionLoss class | ✅ |
| **Loss 2** | "L_spat → Spatial Regularization Loss" | SpatialRegularizationLoss class | ✅ |
| **Total** | "L_total = L_cl + L_recon + L_spat" | compute_total_loss() method | ✅ |
| **Training** | "Backpropagation across network" | torch.backward() support | ✅ |

**Result:** ✅ 7/7 flow.md requirements met

### ✅ module_explanation.md Compliance (Lines 229-258)

| Specification | Master Detail | Implementation | Status |
|---|---|---|---|
| **Problem** | Verify latent space didn't lose information | Reconstruction verification | ✅ |
| **Mechanism 1** | "Feature Stretching: Decode to high dims" | RNADecoder (64→18085) + ADTDecoder (64→31) | ✅ |
| **Mechanism 2** | "Fidelity Cross-Check (MSE)" | ReconstructionLoss (MSE) | ✅ |
| **Mechanism 3** | "Neighborhood Evaluation (Spatial)" | SpatialRegularizationLoss (Laplacian) | ✅ |
| **Loss Formula 1** | MSE for reconstruction | Lines 177-179 exact formula | ✅ |
| **Loss Formula 2** | Graph Laplacian smoothing | Lines 183-185 exact formula | ✅ |
| **Loss Formula 3** | Total = λ₁·L_cl + λ₂·L_recon + λ₃·L_spat | compute_total_loss() method | ✅ |

**Result:** ✅ 7/7 module_explanation.md specifications correct

### ✅ KAC-Net_MASTER_PLAN.md Compliance (Lines 87-102)

| Master Plan Item | Specification | Implementation | Status |
|---|---|---|---|
| **Source** | "SpatialGlue + Custom" | SpatialGlue decoders + custom losses | ✅ |
| **Extract** | "Decoder class" | RNADecoder + ADTDecoder | ✅ |
| **Extract** | "Loss functions" | ReconstructionLoss + SpatialRegularizationLoss | ✅ |
| **Input** | "Z_Fused (3484×64)" | forward(z_fused, ...) parameter | ✅ |
| **Output** | "X̂_RNA (3484×18085)" | Returns from RNADecoder | ✅ |
| **Output** | "X̂_ADT (3484×31)" | Returns from ADTDecoder | ✅ |
| **Output** | "L_total loss" | compute_total_loss() method | ✅ |

**Result:** ✅ 7/7 KAC-Net_MASTER_PLAN requirements met

---

## Usage in Training Pipeline

### Complete Training Loop Integration
```python
import torch
import torch.optim as optim
from reconstruction_loss import create_reconstruction_module
from contrastive_alignment import ContrastiveAlignmentModule
from dual_attention_fusion import DualAttentionFusionModule

# Initialize all modules
module5 = ContrastiveAlignmentModule()
module6 = DualAttentionFusionModule()
module7 = create_reconstruction_module(device='cuda')

optimizer = optim.Adam(
    list(module5.parameters()) + 
    list(module6.parameters()) + 
    list(module7.parameters()),
    lr=1e-4
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
        # Get batch data
        z_rna, z_adt = batch['z_rna'], batch['z_adt']
        x_rna_true, x_adt_true = batch['x_rna'], batch['x_adt']
        A_s, A_f = batch['A_s'], batch['A_f']

        # Module 5: Contrastive alignment
        loss_cl = module5(z_rna, z_adt)

        # Module 6: Dual-attention fusion
        z_fused = module6(z_rna, z_adt, A_s, A_f)

        # Module 7: Reconstruction & regularization
        x_rna_hat, x_adt_hat, loss_recon, loss_spat = module7(
            z_fused, x_rna_true, x_adt_true, A_s
        )

        # Aggregate losses
        loss_total = module7.compute_total_loss(
            loss_cl, loss_recon, loss_spat,
            lambda_cl=1.0,
            lambda_recon=1.0,
            lambda_spat=0.1
        )

        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # Monitoring
        if batch_idx % 10 == 0:
            quality = module7.compute_reconstruction_quality(
                x_rna_true, x_rna_hat, x_adt_true, x_adt_hat
            )
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"  L_cl: {loss_cl.item():.4f}")
            print(f"  L_recon: {loss_recon.item():.4f}")
            print(f"  L_spat: {loss_spat.item():.4f}")
            print(f"  L_total: {loss_total.item():.4f}")
            print(f"  RNA MSE: {quality['rna_mse']:.4f}")
```

### Monitoring Reconstruction Quality
```python
# After each epoch
quality = module7.compute_reconstruction_quality(
    x_rna_true, x_rna_hat, x_adt_true, x_adt_hat
)

# Log metrics
wandb.log({
    'rna_mse': quality['rna_mse'],      # Should decrease
    'adt_mse': quality['adt_mse'],      # Should decrease
    'rna_mae': quality['rna_mae'],      # Should decrease
    'adt_mae': quality['adt_mae'],      # Should decrease
})

# Good signs:
# - RNA MSE < 0.3 (good reconstruction)
# - ADT MSE < 0.2 (good reconstruction)
# - Losses decreasing during training
```

---

## Summary

**Module 7 achieves dual verification through reconstruction and regularization:**

- ✅ **Input:** Z_Fused (3484×64) from Module 6, X̃_RNA/X̃_ADT from Module 1, A_s from Module 3
- ✅ **RNA Decoder:** 64 → 18085 via MLP (512 hidden)
- ✅ **ADT Decoder:** 64 → 31 via MLP (128 hidden)
- ✅ **Reconstruction Loss:** MSE between decoded and preprocessed data
- ✅ **Spatial Loss:** Graph Laplacian smoothness (penalizes sudden changes)
- ✅ **Total Loss:** Aggregates L_cl (Module 5) + L_recon + L_spat
- ✅ **Integration:** Seamless with Modules 5, 6, and training loop
- ✅ **SpatialGlue Compliance:** 100% adherence to master pipeline specification

**Next Module:** Module 8 (Spatial Domain Identification) uses trained Z_Fused embeddings for Leiden/Louvain clustering and domain discovery.

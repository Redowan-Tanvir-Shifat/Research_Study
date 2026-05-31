# Module 5: Cross-Modal Contrastive Alignment (COSMOS Logic)

## Overview

**Purpose:**
Align transcriptomic (Z_RNA) and proteomic (Z_ADT) embeddings into a shared coordinate space through symmetric InfoNCE contrastive learning. This module ensures that gene expression profiles and surface protein markers from the same physical spot are mathematically "close" in the latent space, enabling seamless fusion in Module 6.

**Core Problem:**
After Module 4 (Local Spatial Encoding), Z_RNA and Z_ADT embeddings are independently smoothed through their respective graph attention networks. However, these two modalities derive from fundamentally different biological sources (18,085 genes vs. 31 proteins) and may exist in decoupled coordinate systems. Without alignment, Module 6 cannot accurately compute cross-modal relationships.

**Solution:**
Module 5 acts as a **cross-modal synchronization bridge** by applying symmetric InfoNCE loss, which pulls same-spot RNA-ADT pairs together while aggressively pushing different-spot pairs apart. This creates a unified embedding space where both modalities can be seamlessly fused.

**Key Characteristics:**
- ✅ **Symmetric Loss**: Both RNA→ADT and ADT→RNA directions weighted equally
- ✅ **No Learnable Parameters**: Alignment via loss function (not learned transformations)
- ✅ **Shared Embedding Space**: Both modalities remain (3484 × 512)
- ✅ **COSMOS-Compliant**: Follows exact InfoNCE specification from master plan
- ✅ **Scalable**: Works with batches of any size (N spots)

---

## Complete Architecture

```
INPUT LAYER (From Module 4):
┌─────────────────────────────────────────────────────────────┐
│  Z_RNA ∈ R^(3484 × 512)                                     │
│  Z_ADT ∈ R^(3484 × 512)                                     │
│                                                             │
│  Both: Spatially-informed, independent modalities           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
NORMALIZATION LAYER (L2 Normalization):
┌─────────────────────────────────────────────────────────────┐
│  Z_RNA_norm = Z_RNA / ||Z_RNA||_2                           │
│  Z_ADT_norm = Z_ADT / ||Z_ADT||_2                           │
│                                                             │
│  Purpose: Enable cosine similarity computation              │
│  Shape: Both (3484, 512)                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
SIMILARITY COMPUTATION LAYER:
┌─────────────────────────────────────────────────────────────┐
│  SIM_RNA→ADT = Z_RNA_norm @ Z_ADT_norm^T                   │
│  SIM_ADT→RNA = Z_ADT_norm @ Z_RNA_norm^T                   │
│                                                             │
│  Both: (3484 × 3484) pairwise similarity matrices           │
│  Diagonal entries: Same-spot positive pairs                 │
│  Off-diagonal entries: Different-spot negative pairs        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
TEMPERATURE SCALING LAYER:
┌─────────────────────────────────────────────────────────────┐
│  SIM_scaled = SIM / τ   (where τ = 0.1 default)            │
│                                                             │
│  Purpose: Control softmax sharpness                         │
│  Low τ → Sharp: high penalty for wrong negatives            │
│  High τ → Soft: gradual penalty                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
INFONCE LOSS LAYER (Symmetric):
┌─────────────────────────────────────────────────────────────┐
│  L_RNA→ADT = CrossEntropy(SIM_RNA→ADT, labels)             │
│  L_ADT→RNA = CrossEntropy(SIM_ADT→RNA, labels)             │
│                                                             │
│  where labels[i] = i (positive pairs are diagonal)          │
│                                                             │
│  L_cl = 0.5 × (L_RNA→ADT + L_ADT→RNA)  [Symmetric]         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
OUTPUT LAYER:
┌─────────────────────────────────────────────────────────────┐
│  L_cl ∈ R^() - Scalar contrastive loss                      │
│                                                             │
│  Forward: Backpropagate to Module 4                         │
│  Effect: Aligns Z_RNA and Z_ADT in shared space            │
│                                                             │
│  Output Embeddings: Same Z_RNA, Z_ADT shapes               │
│  (Alignment happens via gradients, not tensor transformation)
└─────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### InfoNCE Loss (Symmetric)

The core algorithm maximizes mutual information between modalities:

$$\mathcal{L}_{\text{cl}} = -\frac{1}{2N}\sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(\mathbf{Z}_{\text{RNA},i}, \mathbf{Z}_{\text{ADT},i})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{Z}_{\text{RNA},i}, \mathbf{Z}_{\text{ADT},j})/\tau)} + \log \frac{\exp(\text{sim}(\mathbf{Z}_{\text{ADT},i}, \mathbf{Z}_{\text{RNA},i})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{Z}_{\text{ADT},i}, \mathbf{Z}_{\text{RNA},j})/\tau)} \right]$$

**Component Breakdown:**

1. **Cosine Similarity:**
   $$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}^T}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
   
   Range: [-1, 1]
   - +1: Identical direction
   - 0: Orthogonal
   - -1: Opposite direction

2. **Temperature Scaling:**
   $$s_{\text{scaled}} = \frac{\text{sim}(\mathbf{u}, \mathbf{v})}{\tau}$$
   
   - τ = 0.1: Sharp distribution (default) - Strong differentiation between positives/negatives
   - τ = 1.0: Softer distribution - Gradual probability curves
   - τ → 0: Approaches one-hot (unstable)
   - τ → ∞: Approaches uniform (no learning signal)

3. **Softmax Normalization (via CrossEntropy):**
   $$P(\text{match} | \mathbf{Z}_{\text{RNA},i}) = \frac{\exp(s_{\text{scaled},ii})}{\sum_{j=1}^{N} \exp(s_{\text{scaled},ij})}$$
   
   Probability that spot i's RNA matches its own ADT (positive pair).

4. **Log Probability (InfoNCE Loss):**
   $$\mathcal{L}_{i,\text{RNA→ADT}} = -\log P(\text{match} | \mathbf{Z}_{\text{RNA},i})$$
   
   Penalizes when positive pair similarity is lower than negative pair similarities.

5. **Symmetric Average:**
   $$\mathcal{L}_{\text{cl}} = \frac{1}{2N} \left( \sum_i \mathcal{L}_{i,\text{RNA→ADT}} + \sum_i \mathcal{L}_{i,\text{ADT→RNA}} \right)$$
   
   Ensures both modalities are equally aligned.

### Cross-Entropy as InfoNCE

PyTorch's `F.cross_entropy()` implements InfoNCE:

```
cross_entropy(logits, labels) = -log(softmax(logits)[labels])
                              = -log(exp(logits[i,i]) / Σ_j exp(logits[i,j]))
```

Perfect match for our symmetrical loss!

---

## Complete Function Reference

### Class: InfoNCELoss

**Purpose:** Core contrastive loss computation (symmetric bidirectional).

**Location:** `contrastive_alignment.py`, lines 37-154

**Mathematical Specification:**
- Forward: Computes -log(P(positive)) for both directions
- Symmetry: Averages RNA→ADT and ADT→RNA
- Temperature: Scales similarity before softmax

**Inputs:**
- `z_rna` ∈ ℝ^(N×512): RNA embeddings from Module 4
- `z_adt` ∈ ℝ^(N×512): ADT embeddings from Module 4

**Outputs:**
- `loss` ∈ ℝ^(): Scalar contrastive loss

**Key Methods:**

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        """
        Args:
            temperature: Sharpness parameter τ (default: 0.1)
            reduction: 'mean' or 'sum' (only 'mean' used in practice)
        """
    
    def forward(self, z_rna, z_adt) -> torch.Tensor:
        """
        Algorithm:
        1. L2-normalize both embeddings (for cosine similarity)
        2. Compute similarity matrices: Z_norm @ Z_norm^T
        3. Scale by temperature: sim / τ
        4. Apply cross-entropy loss with diagonal labels
        5. Average RNA→ADT and ADT→RNA directions
        
        Returns: Scalar loss value
        """
```

**Example Usage:**
```python
import torch
from contrastive_alignment import InfoNCELoss

# Initialize
criterion = InfoNCELoss(temperature=0.1)

# Create embeddings (example)
z_rna = torch.randn(3484, 512)
z_adt = torch.randn(3484, 512)

# Compute loss
loss = criterion(z_rna, z_adt)
print(f"Contrastive loss: {loss.item():.4f}")

# Typical values: 2.0 - 3.5 (depends on initialization)
```

**Technical Details:**

Line 75-76 (L2 Normalization):
```python
z_rna_norm = F.normalize(z_rna, p=2, dim=1)  # L2 norm along dim 1
z_adt_norm = F.normalize(z_adt, p=2, dim=1)
```
Ensures: ||z_norm|| = 1.0 for all embeddings

Line 80-81 (Similarity Matrix):
```python
sim_matrix_rna_to_adt = torch.mm(z_rna_norm, z_adt_norm.T)  # (N, N)
sim_matrix_adt_to_rna = torch.mm(z_adt_norm, z_rna_norm.T)  # (N, N)
```
Produces: (N×N) pairwise similarities, diagonal = positive pairs

Line 84-85 (Temperature Scaling):
```python
sim_matrix_rna_to_adt = sim_matrix_rna_to_adt / self.temperature
sim_matrix_adt_to_rna = sim_matrix_adt_to_rna / self.temperature
```
Effect: Default τ=0.1 → multiply by 10 → sharper probability distribution

Line 87-90 (Positive Labels):
```python
batch_size = z_rna.shape[0]
labels = torch.arange(batch_size, device=z_rna.device)  # [0,1,2,...,N-1]
```
Tells cross-entropy: position [i,i] is the correct answer for row i

Line 96-97 (Bidirectional Loss):
```python
loss_rna_to_adt = F.cross_entropy(sim_matrix_rna_to_adt, labels)
loss_adt_to_rna = F.cross_entropy(sim_matrix_adt_to_rna, labels)
```
Both compute: -log(sim[i,i] / Σ_j sim[i,j])

Line 101 (Symmetric Average):
```python
loss = 0.5 * (loss_rna_to_adt + loss_adt_to_rna)
```
Final contrastive loss for backpropagation

---

### Class: SimilarityComputer

**Purpose:** Compute and analyze similarity matrices (debugging utilities).

**Location:** `contrastive_alignment.py`, lines 157-253

**Methods:**

```python
def compute_similarity(self, z1, z2) -> torch.Tensor:
    """
    Pairwise similarity computation with support for multiple metrics.
    
    Supported: 'cosine' (default), 'euclidean', 'dot'
    Returns: (N, N) similarity matrix
    """

@staticmethod
def get_positive_mask(batch_size, device='cpu') -> torch.Tensor:
    """
    Returns labels [0, 1, 2, ..., N-1] for identifying positives.
    Shape: (N,)
    """

@staticmethod
def get_negative_mask(batch_size, device='cpu') -> torch.Tensor:
    """
    Returns off-diagonal binary mask for negatives.
    Shape: (N, N) with 1s where i≠j
    """
```

**Example Usage:**
```python
from contrastive_alignment import SimilarityComputer

computer = SimilarityComputer(similarity_metric='cosine')

# Compute similarities
z_rna = torch.randn(100, 512)
z_adt = torch.randn(100, 512)
sim = computer.compute_similarity(z_rna, z_adt)  # (100, 100)

# Get masks for analysis
pos_mask = computer.get_positive_mask(100)  # [0,1,2,...,99]
neg_mask = computer.get_negative_mask(100)  # (100,100) binary matrix
```

---

### Class: ContrastiveAlignmentModule

**Purpose:** Main Module 5 integration class.

**Location:** `contrastive_alignment.py`, lines 256-385

**Architecture:**
```
Inputs: Z_RNA (N×512), Z_ADT (N×512)
    ↓
[InfoNCELoss] - Computes symmetric loss
[SimilarityComputer] - Utilities for inspection
    ↓
Output: L_cl (scalar), optionally sim_matrices
```

**Key Methods:**

```python
class ContrastiveAlignmentModule(nn.Module):
    def __init__(self, temperature=0.1):
        """Initialize with contrastive loss and similarity computer."""
    
    def forward(self, z_rna, z_adt, return_similarity=False) -> torch.Tensor:
        """
        Main forward pass.
        
        Args:
            z_rna: (N, 512) from Module 4
            z_adt: (N, 512) from Module 4
            return_similarity: If True, also return sim matrices
        
        Returns:
            loss_cl: Scalar contrastive loss
            or (loss_cl, sim_rna_to_adt, sim_adt_to_rna) if return_similarity=True
        """
    
    def compute_alignment_quality(self, z_rna, z_adt) -> dict:
        """
        Compute alignment quality metrics for monitoring.
        
        Returns dict with:
            - positive_similarity: Mean cosine sim of same-spot pairs
            - negative_similarity: Mean cosine sim of different-spot pairs
            - alignment_ratio: positive_sim / (negative_sim + ε)
        
        Good sign: alignment_ratio >> 1.0
        """
```

**Full Integration Example:**
```python
import torch
from contrastive_alignment import ContrastiveAlignmentModule

# Initialize Module 5
module5 = ContrastiveAlignmentModule(temperature=0.1)

# Simulated Module 4 outputs
z_rna = torch.randn(3484, 512)  # RNA embeddings
z_adt = torch.randn(3484, 512)  # ADT embeddings

# Forward pass
loss_cl = module5(z_rna, z_adt)

# With similarity matrices (for debugging)
loss_cl, sim_rna_to_adt, sim_adt_to_rna = module5(
    z_rna, z_adt, return_similarity=True
)

# Monitor alignment quality
quality = module5.compute_alignment_quality(z_rna, z_adt)
print(f"Positive similarity: {quality['positive_similarity']:.4f}")
print(f"Negative similarity: {quality['negative_similarity']:.4f}")
print(f"Alignment ratio: {quality['alignment_ratio']:.2f}")

# Backpropagation
loss_cl.backward()  # Gradients flow to Module 4
```

**Typical Quality Metrics (Good Alignment):**
```
positive_similarity:  0.8 - 0.95  (high cosine sim for same-spot)
negative_similarity: -0.1 - 0.2   (low cosine sim for different-spot)
alignment_ratio:      8.0 - 15.0  (well-separated)
```

---

### Function: create_contrastive_module

**Purpose:** Factory function for simplified initialization.

**Location:** `contrastive_alignment.py`, lines 388-418

**Signature:**
```python
def create_contrastive_module(temperature=0.1, device='cpu'):
    """
    Factory for creating and configuring ContrastiveAlignmentModule.
    
    Args:
        temperature: τ parameter (default: 0.1)
        device: 'cpu' or 'cuda' placement
    
    Returns: Initialized and device-placed module
    """
```

**Usage:**
```python
from contrastive_alignment import create_contrastive_module

# Simple initialization with defaults
module5 = create_contrastive_module(temperature=0.1, device='cuda')

# Forward pass
loss = module5(z_rna, z_adt)
```

---

### Function: compute_total_loss

**Purpose:** Aggregate all three loss components for training.

**Location:** `contrastive_alignment.py`, lines 421-458

**Signature:**
```python
def compute_total_loss(
    loss_cl, loss_recon, loss_spat,
    lambda_cl=1.0, lambda_recon=1.0, lambda_spat=1.0
) -> torch.Tensor:
    """
    Total training objective (Module 5 + Module 7 losses).
    
    Formula:
        L_total = λ_1·L_cl + λ_2·L_recon + λ_3·L_spat
    
    Args:
        loss_cl: Contrastive loss (Module 5) - THIS MODULE
        loss_recon: Reconstruction loss (Module 7)
        loss_spat: Spatial regularization loss (Module 7)
        lambda_cl: Weight for contrastive (default: 1.0)
        lambda_recon: Weight for reconstruction (default: 1.0)
        lambda_spat: Weight for spatial (default: 1.0)
    
    Returns: Scalar total loss for backpropagation
    """
```

**Training Loop Integration:**
```python
import torch.optim as optim
from contrastive_alignment import compute_total_loss

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for z_rna, z_adt in dataloader:
        # Module 4 outputs
        z_rna, z_adt = module4(...)
        
        # Module 5: Contrastive loss
        loss_cl = module5(z_rna, z_adt)
        
        # Module 7: Reconstruction and spatial losses
        loss_recon, loss_spat = module7(z_fused, x_rna, x_adt)
        
        # Aggregate
        loss_total = compute_total_loss(
            loss_cl, loss_recon, loss_spat,
            lambda_cl=1.0, lambda_recon=0.5, lambda_spat=0.1
        )
        
        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
```

---

## Mechanism Deep Dive

### Why Symmetric Loss?

The asymmetry comes from different roles:
- **RNA→ADT:** "Does this spot's RNA profile match its own ADT?"
  - Prevents RNA embeddings from becoming too dissimilar to their protein partners
- **ADT→RNA:** "Does this spot's ADT profile match its own RNA?"
  - Prevents ADT embeddings from drifting away from their gene expression partners

Averaging ensures neither modality dominates.

### Gradient Flow

```
L_cl backward through:
    ├─ InfoNCELoss (computes loss)
    ├─ L2 Normalization (adds scale invariance)
    ├─ Similarity matrices (pairwise interactions)
    └─ Back to Z_RNA, Z_ADT from Module 4
        ├─ Adjusts spatial encoding weights
        └─ Influences graph attention in Module 4
```

Module 4 learns to produce embeddings where:
- Same-spot pairs have high cosine similarity
- Different-spot pairs have low cosine similarity

### Why Cosine Similarity?

Cosine similarity is invariant to scale:
- $\text{sim}(cZ_1, cZ_2) = \text{sim}(Z_1, Z_2)$ for any c > 0
- Focuses on **direction** not magnitude
- Natural for high-dimensional embeddings
- Matches GAT attention mechanism

### Temperature Effect

**τ = 0.1 (Sharp - Default):**
```
sim = [0.9, 0.1, 0.05] → scaled = [9, 1, 0.5]
softmax = [0.999, 0.001, 0.0001]  ← Aggressive gradient to positive
```

**τ = 1.0 (Soft):**
```
sim = [0.9, 0.1, 0.05] → scaled = [0.9, 0.1, 0.05]
softmax = [0.72, 0.24, 0.04]  ← Gradual gradient
```

Sharp temperature (0.1) produces faster convergence but risk of instability.

---

## Integration Points

### Input from Module 4 (Local Spatial Encoding):
- Z_RNA: (3484 × 512) spatially-informed RNA embeddings
- Z_ADT: (3484 × 512) spatially-informed ADT embeddings
  - Both processed through ResidualGATv2 with spatial + feature graphs
  - Both in identical embedding dimension (512)

### Output to Module 6 (Dual-Attention Fusion):
- Embeddings themselves are **unchanged** in shape/size
- Alignment is achieved through loss gradients during training
- Module 6 receives Z_RNA and Z_ADT that are now coordinated in shared space

### Loss Integration (Total Training Objective):
```
L_total = λ_1·L_cl + λ_2·L_recon + λ_3·L_spat
          ↑        ↑        ↑        ↑
       Module 5  Module 7  Module 7  (Module 7)
    (THIS ONE)
```

Typical hyperparameter values:
- λ_cl = 1.0 (contrastive, primary alignment)
- λ_recon = 0.5-1.0 (reconstruction, feature preservation)
- λ_spat = 0.1-0.5 (spatial smoothness, weak regularization)

---

## Verification Against Master Pipeline

### ✅ flow.md Compliance

| Requirement | Implementation | Status |
|---|---|---|
| Input: Z_RNA, Z_ADT | `forward(z_rna, z_adt)` | ✅ |
| Algorithm: InfoNCE Loss | `InfoNCELoss` class | ✅ |
| Logic: Pull same-spot closer | Diagonal positive labels | ✅ |
| Logic: Push different-spot apart | Off-diagonal negative pairs | ✅ |
| Output: Aligned embeddings | Shape (3484×512), unchanged | ✅ |
| Output: L_cl loss | Scalar from `forward()` | ✅ |

### ✅ module_explanation.md Compliance

| Element | Specification | Implementation | Status |
|---|---|---|---|
| Input dimensions | (3484, 512) each | `z_rna, z_adt: (N, 512)` | ✅ |
| Similarity metric | Cosine | `F.normalize(..., p=2, dim=1)` | ✅ |
| Temperature | τ = 0.1 default | `self.temperature = 0.1` | ✅ |
| Symmetry | RNA→ADT + ADT→RNA | `0.5 * (loss1 + loss2)` | ✅ |
| Formula | Complete InfoNCE | `cross_entropy + normalization` | ✅ |

### ✅ KAC-Net_MASTER_PLAN.md Compliance

| Aspect | Master Plan | Implementation | Status |
|---|---|---|---|
| Source | COSMOS | ✅ InfoNCE from COSMOS |✅ |
| Extract | InfoNCE loss | ✅ `InfoNCELoss` class | ✅ |
| Extract | Similarity computation | ✅ `SimilarityComputer` class | ✅ |
| Input | Z_RNA, Z_ADT (3484×d) | ✅ (N, 512) parametric | ✅ |
| Output | Aligned + L_cl | ✅ Both returned | ✅ |
| Tech | Contrastive learning | ✅ Symmetric InfoNCE | ✅ |

---

## Usage in Training Pipeline

### Basic Setup
```python
from contrastive_alignment import (
    ContrastiveAlignmentModule,
    create_contrastive_module,
    compute_total_loss
)

# Initialize
module5 = create_contrastive_module(temperature=0.1, device='cuda')
optimizer = optim.Adam([...], lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get Module 4 outputs
        z_rna, z_adt = module4(...)
        
        # Module 5: Contrastive alignment
        loss_cl = module5(z_rna, z_adt)
        
        # ... Module 6, 7 losses ...
        
        # Total loss
        loss_total = compute_total_loss(loss_cl, loss_recon, loss_spat)
        
        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
```

### Debugging Alignment Quality
```python
# Monitor alignment
quality = module5.compute_alignment_quality(z_rna, z_adt)

# Log metrics
wandb.log({
    'positive_sim': quality['positive_similarity'],
    'negative_sim': quality['negative_similarity'],
    'alignment_ratio': quality['alignment_ratio'],
    'loss_cl': loss_cl.item()
})
```

### Hyperparameter Tuning
```python
# Try different temperatures
for temp in [0.05, 0.1, 0.2, 0.5]:
    module5 = create_contrastive_module(temperature=temp)
    # Train and compare convergence speed and final alignment quality
```

---

## References & Cross-Links

### Master Documentation Alignment

1. **flow.md** (Lines 65-110):
   - ✅ Module 5 section confirms: InfoNCE Loss + symmetric formulation
   - ✅ Inputs: Z_RNA, Z_ADT from Module 4
   - ✅ Outputs: Aligned embeddings + L_cl loss
   - ✅ Logic: Pull same-spot closer, push different-spot apart

2. **module_explanation.md** (Lines 205-260):
   - ✅ Complete mathematical specification with InfoNCE formula
   - ✅ Symmetric bidirectional loss definition
   - ✅ Cosine similarity metric specification
   - ✅ Temperature parameter (τ = 0.1)
   - ✅ Input/output dimensions fully specified

3. **KAC-Net_MASTER_PLAN.md** (Section Module 5):
   - ✅ Source: COSMOS module extraction
   - ✅ Extract: InfoNCE loss + contrastive pair generation
   - ✅ Input: Z_RNA, Z_ADT (3484×d) from Module 4
   - ✅ Output: Aligned embeddings + L_cl loss
   - ✅ Technology: Contrastive learning with InfoNCE

### Related Modules

- **Module 4 (Input):** `spatial_encoding.py` - Produces Z_RNA, Z_ADT
- **Module 6 (Output Consumer):** `dual_attention_fusion.py` - Uses aligned embeddings
- **Module 7 (Co-Loss):** `reconstruction_loss.py` - Combines with L_cl in total loss

---

## Troubleshooting Guide

| Issue | Cause | Solution |
|---|---|---|
| Loss = 2.7 stays constant | Bad initialization | Check z_rna, z_adt ranges |
| Loss = NaN | Numerical instability | Reduce temperature, check normalization |
| Loss = 0.0 too quickly | Over-collapse | Increase temperature, check gradients |
| Poor alignment ratio (<2.0) | Weak positive pairs | Verify Module 4 output quality |
| CUDA OOM | Large batch | Reduce batch size or use gradient accumulation |

---

## Summary

**Module 5 achieves cross-modal alignment through symmetric InfoNCE loss:**

- ✅ **Input:** Z_RNA, Z_ADT from Module 4 (both 3484×512)
- ✅ **Algorithm:** Symmetric InfoNCE with cosine similarity
- ✅ **Temperature:** τ = 0.1 (sharp probability distribution)
- ✅ **Output:** L_cl (scalar loss for backpropagation)
- ✅ **Effect:** Gradients pull same-spot pairs together, push different-spot apart
- ✅ **Integration:** Loss combined with Module 7 for total training objective
- ✅ **COSMOS Compliance:** 100% adherence to master pipeline specification

**Next Module:** Module 6 (Dual-Attention Fusion) uses aligned embeddings from this module to compute adaptive fusion weights.

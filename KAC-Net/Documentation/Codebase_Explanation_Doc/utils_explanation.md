# utils.py Explanation

**File:** `utils.py`  
**Lines:** 600+  
**Purpose:** Shared utility functions for data loading, preprocessing, visualization, and metrics  
**Status:** ✅ Production-ready

---

## 📌 Why Do You Need utils.py?

### **The Problem (Without utils.py)**

Every script that needs to:
- Load H5AD files → Duplicate code in each file
- Create DataLoaders → Repeated boilerplate
- Make visualizations → Copy-paste matplotlib code
- Compute metrics → Reimplement ARI, NMI, silhouette, etc.
- Normalize data → Same normalization logic everywhere

**Result:** ❌ Code duplication, maintenance nightmare, inconsistency

### **The Solution (utils.py)**

**Centralized Utilities:**
```python
# Before (without utils.py):
# In trainer.py:
adata = sc.read_h5ad('data/adata_RNA.h5ad')  # Duplicate
X_RNA = adata.X.toarray()

# In tutorial.ipynb:
adata = sc.read_h5ad('data/adata_RNA.h5ad')  # Duplicate again!
X_RNA = adata.X.toarray()

# After (with utils.py):
from utils import load_lymph_node_data
X_RNA, X_ADT, coords, labels, meta = load_lymph_node_data()  # Use everywhere!
```

**Benefits:**
✅ **No Code Duplication** - Write once, use everywhere  
✅ **Consistent Behavior** - All loading/plotting uses same logic  
✅ **Easy Maintenance** - Change once, affects all uses  
✅ **Professional** - Separates utilities from business logic  
✅ **Reusable** - Import in any script/notebook  
✅ **Tested** - Single implementation to test  

---

## 🏗️ Function Organization

### **4 Main Categories**

```
utils.py
├── DATA LOADING (Load H5AD, create DataLoaders)
├── VISUALIZATION (UMAP, heatmaps, spatial plots)
├── METRICS (ARI, NMI, silhouette, modularity)
└── HELPERS (Normalization, top genes, AnnData prep)
```

---

## 📂 Data Loading Functions

### **1. `load_lymph_node_data()` - Load 10X Lymph Node Dataset**

**What it does:**
- Loads RNA H5AD file
- Loads ADT H5AD file
- Extracts spatial coordinates
- Loads ground truth annotations
- Returns everything ready to use

**Returns:**
```python
X_RNA           # (3484, 18085) - RNA expression
X_ADT           # (3484, 31)    - Protein expression
coords          # (3484, 2)     - Spatial coordinates
gt_labels       # (3484,)       - Ground truth domains
metadata        # DataFrame     - Metadata
```

**Example:**
```python
from utils import load_lymph_node_data

X_RNA, X_ADT, coords, gt_labels, meta = load_lymph_node_data()
print(f"Data loaded! RNA shape: {X_RNA.shape}, ADT shape: {X_ADT.shape}")
# Output: Data loaded! RNA shape: (3484, 18085), ADT shape: (3484, 31)
```

**Why separate this?**
- Handles all the messy H5AD loading
- Automatically extracts spatial coords from obsm['spatial']
- Matches all coordinate systems (no index mismatches)
- Reproducible across all scripts

---

### **2. `load_data()` - Generic H5AD Loader**

**What it does:**
- Loads any RNA and ADT H5AD files (not just lymph node)
- Optional annotations
- Optional normalization
- Works with any dataset

**Use when:**
- Testing on different datasets
- Using custom data
- Extending to new modalities

**Example:**
```python
from utils import load_data

X_RNA, X_ADT, coords, labels = load_data(
    rna_path='data/my_rna.h5ad',
    adt_path='data/my_adt.h5ad',
    annotation_path='data/my_annotations.csv',
    normalize=True
)
```

---

### **3. `create_data_loaders()` - PyTorch DataLoaders**

**What it does:**
- Creates `MultimodalDataset` (custom PyTorch Dataset)
- Splits into train/val/test (80/10/10 by default)
- Creates DataLoaders with batch size, shuffling, workers
- Handles normalization automatically
- Reproducible splits with seed

**Why needed?**
- ❌ Without: Manual splitting, manual tensor conversion, lots of boilerplate
- ✅ With: One function call, everything ready for training

**Example:**
```python
from utils import create_data_loaders

train_loader, val_loader, test_loader = create_data_loaders(
    X_RNA=X_RNA,
    X_ADT=X_ADT,
    coords=coords,
    batch_size=256,
    num_workers=4,
    normalize=True
)

# Now ready to train:
for X_rna_batch, X_adt_batch, coords_batch in train_loader:
    print(X_rna_batch.shape)  # [256, 18085]
    # Pass to model...
```

**Default Split:**
- Train: 80% (2787 spots)
- Val: 10% (348 spots)
- Test: 10% (349 spots)

**Customization:**
```python
train_loader, val_loader, test_loader = create_data_loaders(
    X_RNA, X_ADT, coords,
    train_ratio=0.7,    # 70% train
    val_ratio=0.15,     # 15% validation
    batch_size=128,     # Smaller batches
    num_workers=8,      # More workers = faster loading
    seed=123            # Reproducible split
)
```

---

## 📊 Visualization Functions

### **1. `plot_umap()` - UMAP of Embeddings**

**What it does:**
- Computes 2D UMAP from embeddings
- Colors by cluster labels
- Visualizes learned representation

**Good for:**
- Seeing if clusters are separated
- Qualitative evaluation of model
- Publication figures

**Example:**
```python
from utils import plot_umap

# After training, get Z_Fused embeddings
Z_fused = model.get_embeddings(full_loader)  # [3484, 64]
pred_domains = leiden_clustering(Z_fused)    # [3484,]

# Plot
fig = plot_umap(Z_fused, labels=pred_domains, title="KAC-Net UMAP")
plt.savefig('results/umap.png', dpi=300)
```

**Output:**
- 2D scatter plot with colors for each domain
- Shows if learned embeddings separate domains well

---

### **2. `plot_genes_heatmap()` - Top Gene Heatmap**

**What it does:**
- Computes mean gene expression per domain
- Selects top N genes by variance
- Creates heatmap (domains × genes)

**Good for:**
- Finding domain-specific genes
- Biological validation
- Marker gene discovery

**Example:**
```python
from utils import plot_genes_heatmap

fig = plot_genes_heatmap(
    X_RNA=X_RNA,
    labels=pred_domains,
    gene_names=gene_list,  # Optional
    n_top_genes=20,
    save_path='results/top_genes.png'
)
```

**Output:**
- Rows: Top 20 genes
- Columns: Each domain
- Colors: Expression level (red=high, blue=low)

---

### **3. `plot_spatial_distribution()` - Spatial Domain Map**

**What it does:**
- Plots spots on spatial coordinates
- Colors by domain labels
- Shows spatial organization

**Good for:**
- Validating spatial structure
- Comparing ground truth vs predicted
- Understanding domain boundaries

**Example:**
```python
from utils import plot_spatial_distribution

# Ground truth
fig1 = plot_spatial_distribution(
    coords, gt_labels, 
    title="Ground Truth Domains",
    save_path='results/gt_spatial.png'
)

# Predicted
fig2 = plot_spatial_distribution(
    coords, pred_domains,
    title="Predicted Domains",
    save_path='results/pred_spatial.png'
)
```

**Output:**
- 2D scatter plot on spatial coordinates
- Each spot colored by domain

---

### **4. `plot_confusion_matrix()` - Clustering Comparison**

**What it does:**
- Compares ground truth vs predicted labels
- Creates confusion matrix heatmap
- Shows which domains are confused

**Example:**
```python
from utils import plot_confusion_matrix

fig = plot_confusion_matrix(
    gt_labels=gt_labels,
    pred_labels=pred_domains,
    save_path='results/confusion_matrix.png'
)
```

**Output:**
- Rows: Ground truth domains
- Columns: Predicted domains
- Values: Number of spots in each combination
- Shows which predictions are correct (diagonal) vs confused

---

## 📈 Evaluation Metrics

All metrics compute in ~1 second and log results automatically.

### **1. `compute_ari()` - Adjusted Rand Index**

**What it measures:**
- How well predicted domains match ground truth
- Scale: [-1, 1]
  - 1.0 = Perfect agreement ✓
  - 0.0 = Random chance
  - <0.0 = Worse than random ✗

**Example:**
```python
from utils import compute_ari

ari = compute_ari(gt_labels, pred_domains)
print(f"ARI: {ari:.4f}")
# Output: Adjusted Rand Index (ARI): 0.7234
```

**Interpretation:**
- ARI > 0.7: Good clustering ✓
- ARI 0.5-0.7: Reasonable
- ARI < 0.5: Poor clustering ✗

---

### **2. `compute_nmi()` - Normalized Mutual Information**

**What it measures:**
- Mutual information between two labelings
- Scale: [0, 1]
  - 1.0 = Perfect agreement ✓
  - 0.0 = Independent
  - Symmetric (unlike purity)

**Example:**
```python
from utils import compute_nmi

nmi = compute_nmi(gt_labels, pred_domains)
# Output: Normalized Mutual Information (NMI): 0.8123
```

---

### **3. `compute_silhouette()` - Silhouette Score**

**What it measures:**
- How well separated clusters are in embedding space
- Scale: [-1, 1]
  - 1.0 = Well separated ✓
  - 0.0 = Overlapping
  - -1.0 = Wrong assignments ✗

**Why important?**
- ARI/NMI measure label agreement (external)
- Silhouette measures embedding quality (internal)

**Example:**
```python
from utils import compute_silhouette

silhouette = compute_silhouette(
    Z=Z_fused,
    labels=pred_domains,
    sample_size=1000  # Use subset for speed
)
# Output: Silhouette Score: 0.6234
```

---

### **4. `compute_modularity()` - Graph Modularity**

**What it measures:**
- Strength of community structure in graph
- How much edges stay within domains vs between
- Scale: [-0.5, 1]
  - Higher = stronger community separation ✓

**Example:**
```python
from utils import compute_modularity

modularity = compute_modularity(
    adj_matrix=adj_spatial,  # From graph_construction
    labels=pred_domains
)
# Output: Graph Modularity: 0.4567
```

---

## 🔧 Helper Functions

### **1. `prepare_adata()` - Create AnnData Object**

**Why needed?**
- Many scanpy functions need AnnData format
- Integrates with scanpy pipeline

**Example:**
```python
from utils import prepare_adata

adata = prepare_adata(
    X=Z_fused,
    coords=coords,
    gene_names=[f"PC_{i}" for i in range(64)],
    spot_names=[f"Spot_{i}" for i in range(3484)]
)

# Now use with scanpy:
import scanpy as sc
sc.pp.neighbors(adata, use_rep='X')
sc.tl.leiden(adata)
```

---

### **2. `get_top_genes()` - Extract Top Genes per Domain**

**What it does:**
- Computes mean expression per domain
- Returns top N genes for each domain

**Good for:**
- Biological validation
- Marker gene discovery

**Example:**
```python
from utils import get_top_genes

top_genes = get_top_genes(
    X=X_RNA,
    labels=pred_domains,
    n_genes=50
)

for domain, genes in top_genes.items():
    print(f"Domain {domain} top genes: {genes}")
    # genes is a list of 50 gene indices
```

---

### **3. `normalize_data()` - Data Normalization**

**Supports:**
- `minmax`: Scale to [0, 1] (preserves 0)
- `zscore`: Standardize (mean=0, std=1)

**Example:**
```python
from utils import normalize_data

X_norm_minmax = normalize_data(X_RNA, method='minmax')
X_norm_zscore = normalize_data(X_RNA, method='zscore')
```

---

## 🎯 Complete Workflow Example

### **Full End-to-End Pipeline**

```python
import torch
from utils import (
    load_lymph_node_data,
    create_data_loaders,
    compute_ari, compute_nmi, compute_silhouette,
    plot_umap, plot_spatial_distribution, plot_genes_heatmap,
    plot_confusion_matrix
)
from config import get_config
from kac_net_main import create_kac_net
from trainer import KACNetTrainer

# ========== 1. LOAD DATA ==========
print("Loading data...")
X_RNA, X_ADT, coords, gt_labels, meta = load_lymph_node_data()
# Output: RNA shape: (3484, 18085), ADT shape: (3484, 31)

# ========== 2. CREATE DATALOADERS ==========
print("Creating dataloaders...")
train_loader, val_loader, test_loader = create_data_loaders(
    X_RNA, X_ADT, coords,
    batch_size=256,
    num_workers=4
)
# Output: Train: 2787 samples, Val: 348 samples, Test: 349 samples

# ========== 3. SETUP MODEL & TRAINING ==========
config = get_config('lymph_node')
model = create_kac_net(config, 'cuda')
trainer = KACNetTrainer(model, config, device='cuda')

# ========== 4. TRAIN ==========
print("Training...")
history = trainer.train(train_loader, val_loader, epochs=50)

# ========== 5. GET EMBEDDINGS & CLUSTER ==========
print("Clustering...")
from modules.clustering import leiden_clustering_with_sweep
Z_fused = model.get_embeddings(test_loader)
pred_domains = leiden_clustering_with_sweep(Z_fused)['labels']

# ========== 6. EVALUATE ==========
print("Evaluating...")
ari = compute_ari(gt_labels, pred_domains)          # 0.7234
nmi = compute_nmi(gt_labels, pred_domains)          # 0.8123
silhouette = compute_silhouette(Z_fused, pred_domains)  # 0.6234

print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette: {silhouette:.4f}")

# ========== 7. VISUALIZE ==========
print("Visualizing...")
plot_umap(Z_fused, labels=pred_domains, save_path='umap.png')
plot_spatial_distribution(coords, pred_domains, save_path='spatial.png')
plot_spatial_distribution(coords, gt_labels, title="Ground Truth", save_path='gt_spatial.png')
plot_genes_heatmap(X_RNA, pred_domains, n_top_genes=20, save_path='genes.png')
plot_confusion_matrix(gt_labels, pred_domains, save_path='confusion.png')

print("✓ Complete pipeline finished!")
```

---

## 📋 Complete Function Reference

### **Data Loading**
| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `load_lymph_node_data()` | data_dir | X_RNA, X_ADT, coords, labels, meta | Load 10X data |
| `load_data()` | paths | X_RNA, X_ADT, coords, labels | Generic loader |
| `create_data_loaders()` | X, batch_size | train_loader, val_loader, test_loader | PyTorch training |

### **Visualization**
| Function | Input | Output | Good For |
|----------|-------|--------|----------|
| `plot_umap()` | Z_fused, labels | figure | Embedding quality |
| `plot_genes_heatmap()` | X_RNA, labels | figure | Marker genes |
| `plot_spatial_distribution()` | coords, labels | figure | Spatial structure |
| `plot_confusion_matrix()` | gt_labels, pred_labels | figure | Comparison |

### **Metrics**
| Function | Input | Output | Scale | Interpretation |
|----------|-------|--------|-------|-----------------|
| `compute_ari()` | gt, pred | float | [-1, 1] | 1 = perfect |
| `compute_nmi()` | gt, pred | float | [0, 1] | 1 = perfect |
| `compute_silhouette()` | Z, labels | float | [-1, 1] | 1 = separated |
| `compute_modularity()` | adj, labels | float | [-0.5, 1] | Higher = better |

### **Helpers**
| Function | Purpose |
|----------|---------|
| `prepare_adata()` | Create AnnData for scanpy |
| `get_top_genes()` | Extract marker genes |
| `normalize_data()` | Min-max or z-score normalization |

---

## ✅ Design Principles

### **Why This Organization?**

1. **Single Responsibility**
   - Each function does ONE thing well
   - `load_lymph_node_data()` = loads data (nothing else)

2. **No Side Effects**
   - Functions don't modify global state
   - Deterministic and testable

3. **Clear Naming**
   - `plot_*` functions visualize
   - `compute_*` functions calculate metrics
   - `load_*` functions load data

4. **Consistent Interfaces**
   - All plotting functions: (data, labels, save_path, figsize)
   - All metrics: (gt, pred) or (Z, labels)
   - Logging at end of each operation

5. **Reproducibility**
   - Seeds for data splitting
   - Consistent defaults
   - Logged parameters

---

## 🔗 Integration with Other Modules

### **Data Flow**

```
utils.py (load data)
    ↓
config.py (hyperparameters)
    ↓
trainer.py (training loop)
    ↓
utils.py (visualize & evaluate)
```

### **Import Examples**

**In trainer:**
```python
from utils import create_data_loaders
train_loader, val_loader, test_loader = create_data_loaders(...)
```

**In notebook:**
```python
from utils import load_lymph_node_data, plot_umap, compute_ari
X_RNA, X_ADT, coords, labels, meta = load_lymph_node_data()
```

**In analysis scripts:**
```python
from utils import compute_nmi, get_top_genes, plot_confusion_matrix
nmi = compute_nmi(gt, pred)
```

---

## ✅ Summary

**utils.py provides:**
- ✅ Standardized data loading (H5AD support)
- ✅ PyTorch DataLoader creation
- ✅ 4 types of visualization
- ✅ 4 evaluation metrics
- ✅ Helper functions for common tasks
- ✅ Zero code duplication
- ✅ Professional, maintainable code

**Why it matters:**
- Enables reproducible pipelines
- Saves development time
- Reduces bugs (single implementation)
- Makes code readable and maintainable
- Separates utilities from business logic

---

## 📚 Related Files

- **config.py** - Provides data paths
- **trainer.py** - Uses `create_data_loaders()`
- **kac_net_main.py** - Model being trained
- **clustering.py** - Uses embeddings from trained model
- **tutorial_kac_net.ipynb** - Example usage (next to create)


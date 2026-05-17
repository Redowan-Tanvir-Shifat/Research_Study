# COSMOS: Cooperative Integration of Spatially Resolved Multi-Omics Data

## Overview

COSMOS is an advanced computational framework designed to integrate multiple omics datasets (e.g., RNA and ATAC data) from spatially resolved experiments. It uses a graph neural network approach with weighted nearest neighbors (WNN) analysis to create a unified representation of spatial multi-omics data.

**Paper**: "Cooperative Integration of Spatially Resolved Multi-Omics Data with COSMOS"

---

## Folder Structure & File Architecture

This folder contains 4 core Python modules that work together in a coordinated pipeline:

```
COSMOS/
├── cosmos.py              # Main orchestrator class - entry point for users
├── modulesWNN.py          # Deep Graph Infomax model architecture
├── pyWNN.py               # Weighted Nearest Neighbors analysis
└── util.py                # Utility functions
```

---

## File Relationships & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    cosmos.py (Main Class)                    │
│         Cosmos: High-level API for integration               │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┬─────────────────────┐
         │                   │                     │
         ▼                   ▼                     ▼
    ┌──────────┐      ┌──────────────┐      ┌──────────────┐
    │ util.py  │      │ modulesWNN.py│      │ pyWNN.py     │
    │ (Helper  │      │ (Neural      │      │ (Weighted    │
    │ Tools)   │      │  Network)    │      │  Nearest     │
    │          │      │              │      │  Neighbors)  │
    └──────────┘      └──────────────┘      └──────────────┘
```

---

## Detailed File Descriptions

### 1. **cosmos.py** - Main Integration Engine

#### Purpose
- Entry point for users; provides the high-level `Cosmos` class
- Orchestrates the entire analysis pipeline
- Manages data preprocessing and model training

#### Key Components

**Class: `Cosmos`**
- Accepts two omics datasets (adata1, adata2) or count matrices
- Stores data in AnnData format (standardized single-cell analysis object)

**Main Methods:**

##### `__init__(adata1, adata2, count_matrix1, count_matrix2, spatial_locs, ...)`
- **Input**: Two omics datasets and their spatial coordinates
- **Output**: Initialized Cosmos object
- **Process**: 
  - Accepts either AnnData objects or raw count matrices + locations
  - Creates AnnData objects if needed
  - Stores spatial coordinates in `adata.obsm['spatial']`

##### `preprocessing_data(do_norm, do_log, n_top_genes, do_pca, n_neighbors)`
- **Input**: Preprocessing parameters
- **Output**: Preprocessed data stored in `self.adata1_preprocessed`, `self.adata2_preprocessed`
- **Process**:
  1. Optional normalization (library size correction)
  2. Optional log transformation
  3. Optional selection of highly variable genes
  4. Optional PCA dimensionality reduction
  5. **Spatial graph construction**: Creates k-nearest neighbor graph based on spatial locations
  
  ```
  Spatial coordinates (x,y) → KNN graph → Sparse adjacency matrix
  ```

##### `train(spatial_regularization_strength, z_dim, lr, wnn_epoch, total_epoch, ...)`
- **Input**: Training hyperparameters
- **Output**: 
  - `self.embedding`: Low-dimensional integrated representation (n_cells × z_dim)
  - `self.weights`: Modality weights showing contribution of each omics (n_cells × 2)
- **Process Flow**:
  ```
  Step 1: Prepare data
  ├─ Convert omics matrices to tensors
  ├─ Create spatial graph edges
  └─ Move data to GPU/CPU device
  
  Step 2: Model training loop
  ├─ Initialize DeepGraphInfomaxWNN model (from modulesWNN.py)
  ├─ For each epoch:
  │  ├─ Forward pass through encoder → z (embeddings)
  │  ├─ Corruption function → corrupted embeddings
  │  ├─ Calculate loss (mutual information + spatial regularization)
  │  ├─ Weighted Nearest Neighbors (pyWNN.py) applied at wnn_epoch
  │  └─ Backpropagation & parameter update
  │
  Step 3: Extract final embeddings and weights
  └─ Return integrated representation with modality contributions
  ```

**Key Features:**
- **Spatial Regularization**: Penalty term encouraging nearby cells in space to have similar embeddings
- **WNN Integration**: At a specific epoch, switches to weighted nearest neighbors for adaptive modality integration
- **Early Stopping**: Uses patience mechanism to prevent overfitting
- **GPU Support**: Automatic GPU detection and usage

---

### 2. **modulesWNN.py** - Neural Network Architecture

#### Purpose
- Implements the Deep Graph Infomax model adapted for multi-omics
- Contains the neural network encoder architecture

#### Key Components

**Class: `DeepGraphInfomaxWNN`** (Inherits from `torch.nn.Module`)
- **Purpose**: Implements Deep Graph Infomax framework for learning graph representations
- **Core Concepts**:
  - **Encoder**: Learns latent representations from data
  - **Summary**: Computes a summary vector of the entire graph
  - **Corruption**: Creates corrupted versions of data for contrastive learning
  - **Discriminator**: Uses mutual information to assess quality of embeddings

**Key Methods:**

##### `forward(x1, x2, edge_index, adata, w, w1, w2)`
- **Input**: 
  - x1, x2: Two omics data tensors
  - edge_index: Spatial graph edges
  - adata: AnnData object
  - w: Flag (0 or 1) for WNN computation
  - w1, w2: Current modality weights
- **Output**: (pos_z, neg_z, summary, w1, w2)
  - pos_z: Embeddings of original data
  - neg_z: Embeddings of corrupted data
  - summary: Global graph summary
  - w1, w2: Updated modality weights (or same if not updating)

##### `discriminate(z, summary, sigmoid)`
- Scores patch-summary pairs using a learnable weight matrix
- Higher scores = better alignment between patch and summary

##### `loss(pos_z, neg_z, summary)`
- Implements mutual information maximization
- Positive samples (original) should have high discriminator score
- Negative samples (corrupted) should have low discriminator score

**Class: `GraphEncoderWNNit`** (Inherits from `nn.Module`)
- **Purpose**: Specific encoder for COSMOS that processes both omics simultaneously

**Architecture**:
```
Input Data (x1, x2)
    │
    ├─→ [GCN Conv Layer 1] → ReLU → [GCN Conv Layer 2] → L2 Norm → x1_normalized
    │
    ├─→ [GCN Conv Layer 3] → ReLU → [GCN Conv Layer 4] → L2 Norm → x2_normalized
    │
    ├─→ [If WNN Flag=1: Apply Weighted Nearest Neighbors (pyWNN.py)]
    │        └─→ Compute w1, w2 (weight for each modality per cell)
    │
    └─→ x = (x1_normalized × w1) + (x2_normalized × w2)
           └─→ Weighted combination of normalized embeddings
```

---

### 3. **pyWNN.py** - Weighted Nearest Neighbors Analysis

#### Purpose
- Implements Weighted Nearest Neighbors (WNN) for multi-omics integration
- Based on Hao et al. 2021 methodology
- Computes adaptive weights for each modality per cell

#### Key Concepts

**Weighted Nearest Neighbors (WNN) Principle:**
- For each cell, WNN integrates information from multiple modalities
- Weights reflect the reliability/informativeness of each modality for that cell
- Cells with stronger signal in one modality get higher weight for that modality

**Class: `pyWNN`**

**Initialization: `__init__(adata, reps, n_neighbors, npcs, seed, distances)`**
- **Input**:
  - adata: AnnData object with embeddings in `adata.obsm`
  - reps: List of representation keys (typically ['Omics1_PCA', 'Omics2_PCA'])
  - n_neighbors: Number of neighbors to consider (typically 20)
  - npcs: Number of principal components per modality
- **Output**: Initialized WNN object with precomputed neighbor structures
- **Process**:
  1. Normalize embeddings for both modalities
  2. Compute KNN graphs for each modality independently (20 and 200 neighbors)
  3. Find nearest neighbor for each cell in each modality
  4. Compute bandwidth for kernel density estimation

**Key Method: `compute_weights()`**
- **Purpose**: Calculate adaptive weights w1 and w2 for each cell
- **Algorithm**:
  ```
  For each modality (i):
    1. Predict cell positions using within-modality neighbors
    2. Predict cell positions using cross-modality neighbors
    3. Compute prediction error (distance) for within vs. cross
    
    4. Compute affinity using kernel:
       affinity = exp(-(prediction_error) / bandwidth)
    
    5. Calculate affinity ratio:
       ratio_i = within_affinity / cross_affinity
  
  Final weights (sigmoid transformation):
    w1 = 1 / (1 + exp(ratio_2 - ratio_1))
    w2 = 1 - w1
  
  Interpretation:
  - If modality 1 predicts cell positions better → w1 ≈ 1, w2 ≈ 0
  - If modality 2 predicts cell positions better → w1 ≈ 0, w2 ≈ 1
  - If both equally informative → w1 ≈ w2 ≈ 0.5
  ```

**Key Method: `compute_wnn(adata)`**
- **Purpose**: Construct final WNN graph using weighted distances
- **Process**:
  1. Create union of 200-neighbor graphs from both modalities
  2. Compute distances in each modality's embedding space
  3. Apply modality weights to distances:
     ```
     weighted_distance = w1 × exp(-dist_modality1/bw1) 
                       + w2 × exp(-dist_modality2/bw2)
     ```
  4. Select top K neighbors (20) based on weighted distances
  5. Normalize distances for final KNN graph
- **Output**: 
  - `adata.obsp['WNN']`: Final weighted neighbor adjacency matrix
  - `adata.obsm['Weights']`: Per-cell modality weights (n_cells × 2)

#### Helper Functions

**`get_nearestneighbor(knn, neighbor=1)`**
- Returns nearest neighbor index for each cell

**`compute_bw(knn_adj, embedding, n_neighbors=20)`**
- Computes Jaccard similarity-based bandwidth for kernel density

**`compute_affinity(dist_to_predict, dist_to_nn, bw)`**
- Gaussian kernel affinity based on distance and bandwidth

**`dist_from_adj(adjacency, embed1, embed2, nndist1, nndist2)`**
- Computes distance from adjacency matrix
- Calculates within vs. cross-modality prediction errors

**`select_topK(dist, n_neighbors=20)`**
- Selects top K neighbors by distance

---

### 4. **util.py** - Utility Functions

#### Purpose
- Provides helper functions used by other modules
- Handles data format conversions

#### Functions

**`sparse_mx_to_torch_edge_list(sparse_mx)`**
- **Input**: Scipy sparse matrix (adjacency matrix)
- **Output**: PyTorch edge list tensor (2 × n_edges)
- **Purpose**: Converts sparse spatial graph to PyTorch geometric format
- **Example**:
  ```
  Sparse matrix:     Edge list tensor:
  [0 1 0]   →        [[0, 1, 0, 1],
   [1 0 1]            [1, 0, 1, 2]]
   [0 1 0]
  ```

**`corruptionWNNit(x1, x2, edge_index, adata, w, w1, w2)`**
- **Input**: Data and edge information
- **Output**: Randomly shuffled (corrupted) versions of data
- **Purpose**: Creates negative samples for contrastive learning in Deep Graph Infomax
- **Process**:
  ```
  x1_corrupted = x1[random_permutation]
  x2_corrupted = x2[random_permutation]
  edge_index remains unchanged
  ```

**`corruption(x, edge_index)` (Alternative)**
- Simpler corruption function for basic graph data

---

## Complete Analysis Pipeline

### Step-by-Step Workflow

```
START
  ↓
1. USER INSTANTIATION
   └─ cosmos = Cosmos(adata1=rna_data, adata2=atac_data)
  ↓
2. PREPROCESSING (cosmos.preprocessing_data())
   ├─ Load and validate two omics datasets
   ├─ Optional: Normalization, log-transformation, HVG selection, PCA
   ├─ Extract spatial coordinates
   └─ Build spatial graph (KNN in 2D space)
  ↓
3. MODEL TRAINING (cosmos.train())
   │
   ├─ Initialize DeepGraphInfomaxWNN model (modulesWNN.py)
   │  └─ GraphEncoderWNNit with two parallel GCN pathways
   │
   ├─ Training Loop (multiple epochs):
   │  │
   │  ├─ PRE-WNN PHASE (Epochs 1 to wnn_epoch)
   │  │  ├─ Forward pass through encoder
   │  │  │  ├─ Modality 1: GCN → ReLU → GCN → L2 Norm → x1
   │  │  │  └─ Modality 2: GCN → ReLU → GCN → L2 Norm → x2
   │  │  ├─ Weights: w1=0.5, w2=0.5 (equal initially)
   │  │  ├─ Combined embedding: z = x1×w1 + x2×w2
   │  │  ├─ Compute loss:
   │  │  │  ├─ Deep Graph Infomax loss (modulesWNN.py)
   │  │  │  └─ Spatial regularization penalty
   │  │  └─ Backpropagation
   │  │
   │  ├─ WNN TRANSITION (Epoch = wnn_epoch)
   │  │  ├─ Extract encodings: pc1, pc2
   │  │  ├─ Apply pyWNN.compute_weights() (pyWNN.py)
   │  │  │  ├─ Build KNN graphs for both modalities
   │  │  │  ├─ Compute prediction errors (within vs. cross)
   │  │  │  └─ Calculate adaptive weights w1, w2
   │  │  └─ Reset patience and learning dynamics
   │  │
   │  └─ POST-WNN PHASE (Epochs after wnn_epoch)
   │     ├─ Forward pass with computed weights w1, w2
   │     ├─ z = x1×w1 + x2×w2 (weighted by cell-specific importance)
   │     ├─ Same loss computation
   │     └─ Backpropagation continues with better weights
   │
   ├─ Early Stopping:
   │  ├─ If loss doesn't improve → increment patience counter
   │  ├─ If patience > threshold → stop training
   │  └─ Load best model parameters
   │
   └─ Extract final results:
      ├─ Final embeddings: z (n_cells × z_dim)
      ├─ Modality weights: ww (n_cells × 2)
      └─ Store in self.embedding, self.weights
  ↓
4. ANALYSIS & VISUALIZATION (User code)
   ├─ Clustering on embeddings (Louvain, KMeans, etc.)
   ├─ UMAP visualization
   ├─ Pseudo-spatiotemporal mapping (diffusion pseudotime)
   └─ Interpretation of modality weights
  ↓
END
```

---

## Data Flow Illustration

### Single Cell Processing Through the Pipeline

```
Cell i with two omics:
├─ Modality 1 (RNA): gene expression vector (g1_1, g1_2, ..., g1_n)
├─ Modality 2 (ATAC): chromatin accessibility vector (g2_1, g2_2, ..., g2_m)
└─ Spatial location: (x_i, y_i)

COSMOS.preprocessing_data():
├─ RNA vector → log + normalize → RNA embedding
├─ ATAC vector → log + normalize → ATAC embedding
└─ Location → Connect to k nearest neighbors in 2D space

COSMOS.train():
├─ [Epochs 1-499] Pre-WNN phase
│  ├─ RNA embedding → GCN layers → Normalized RNA encoding (128-d)
│  ├─ ATAC embedding → GCN layers → Normalized ATAC encoding (128-d)
│  ├─ Weight (equally): w1=0.5, w2=0.5
│  └─ Combined: z_i = 0.5 × RNA_encoding + 0.5 × ATAC_encoding
│
├─ [Epoch 500] WNN transition
│  ├─ Collect all cells' encodings: RNA_matrix (n×128), ATAC_matrix (n×128)
│  ├─ PyWNN analysis:
│  │  ├─ Cell i's neighborhood in RNA space: N_RNA(i)
│  │  ├─ Cell i's neighborhood in ATAC space: N_ATAC(i)
│  │  ├─ Prediction error_RNA = ||cell_i - mean(N_RNA(i))||
│  │  ├─ Prediction error_ATAC = ||cell_i - mean(N_ATAC(i))||
│  │  ├─ Affinity_RNA ∝ exp(-error_RNA/bandwidth_RNA)
│  │  ├─ Affinity_ATAC ∝ exp(-error_ATAC/bandwidth_ATAC)
│  │  ├─ w_i = sigmoid(cross_affinity_ATAC - cross_affinity_RNA)
│  │  └─ Result: w1_i=0.7, w2_i=0.3 (RNA more informative for cell i)
│  │
│  └─ Update: w1_i and w2_i stored for future epochs
│
├─ [Epochs 501+] Post-WNN phase
│  ├─ RNA encoding → same as before
│  ├─ ATAC encoding → same as before
│  ├─ Weight (adaptively): w1_i=0.7, w2_i=0.3
│  └─ Combined: z_i = 0.7 × RNA_encoding + 0.3 × ATAC_encoding
│
└─ Final result:
   ├─ Integrated embedding: z_i (128-dimensional)
   ├─ Weight contributions: (0.7, 0.3)
   └─ Both available in cosmos.embedding and cosmos.weights
```

---

## Key Innovation: Why This Matters

### Traditional Multi-Omics Integration Issues
- ❌ Equal weighting of all modalities (ignores cell-type specific informativeness)
- ❌ No spatial context (ignores tissue architecture)
- ❌ Inflexible integration (one-size-fits-all approach)

### COSMOS Solutions
- ✅ **Adaptive Weighting** (pyWNN): Each cell gets custom weights based on local neighborhood quality
- ✅ **Spatial Regularization**: Encourages biologically meaningful representations (neighbors = neighbors)
- ✅ **Deep Learning** (modulesWNN): Learns non-linear relationships
- ✅ **Contrastive Learning**: Uses positive/negative samples for robust representations

---

## Dependencies & Imports

### External Libraries Used
- **torch, torch_geometric**: Deep learning framework
- **scanpy, anndata**: Single-cell analysis standard
- **numpy, scipy**: Numerical computing
- **sklearn**: Machine learning utilities
- **gudhi**: Topological data analysis
- **matplotlib, cmcrameri**: Visualization

### Internal Module Dependencies
```
cosmos.py
├─ imports modulesWNN.DeepGraphInfomaxWNN
├─ imports pyWNN.pyWNN
├─ imports util functions (indirectly)
└─ uses GraphEncoderWNNit class

modulesWNN.py
└─ Standalone (no internal imports)

pyWNN.py
└─ Standalone (no internal imports)

util.py
└─ Standalone (no internal imports)
```

---

## Usage Example

```python
# 1. Initialize
cosmos_obj = Cosmos(adata1=rna_adata, adata2=atac_adata)

# 2. Preprocess
cosmos_obj.preprocessing_data(n_neighbors=10, do_log=True)

# 3. Train
embedding = cosmos_obj.train(
    spatial_regularization_strength=0.01,
    z_dim=50,
    wnn_epoch=500,
    total_epoch=1000,
    gpu=0
)

# 4. Results
integrated_representation = cosmos_obj.embedding  # (n_cells × 50)
modality_weights = cosmos_obj.weights             # (n_cells × 2)
```

---

## Key Parameters Explained

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| `z_dim` | Latent embedding dimension | 50-100 |
| `spatial_regularization_strength` | Weight of spatial smoothness penalty | 0.01-0.1 |
| `wnn_epoch` | When to apply WNN | 500-1000 |
| `lr` | Learning rate for optimizer | 1e-3 |
| `n_neighbors` | KNN graph size | 10-50 |
| `max_patience_aft` | Early stopping threshold | 20-30 |

---

## References & Attribution

This COSMOS software is built upon:

1. **SpaceFlow** (Honglei et al.)
   - Spatial graph construction & regularization ideas
   
2. **PyWNN** (Dylan Kotliar et al.)
   - Weighted nearest neighbors mathematics
   
3. **PyTorch Geometric**
   - Deep Graph Infomax implementation
   
4. **Scanpy / AnnData**
   - Single-cell data standards

---

**For questions about the code**: Contact Xue Xiao (Xiao.Xue@UTSouthwestern.edu)

**Version**: 10/10/2024

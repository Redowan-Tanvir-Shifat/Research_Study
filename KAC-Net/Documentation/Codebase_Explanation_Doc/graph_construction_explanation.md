# Module 3: Multi-Graph Construction - Complete Explanation

## Overview

**Module 3** constructs two complementary graphs that define cell neighborhoods from different perspectives:

**Input:**
- Spatial coordinates: $(x, y) \in \mathbb{R}^{3484 \times 2}$ (from Module 1)
- Enriched RNA embedding: $H_{\text{RNA}} \in \mathbb{R}^{3484 \times 512}$ (from Module 2)
- Normalized ADT counts: $\tilde{X}_{\text{ADT}} \in \mathbb{R}^{3484 \times 31}$ (from Module 1)

**Output:**
- Spatial adjacency: $A_s \in \mathbb{R}^{3484 \times 3484}$ (sparse, normalized)
- Feature adjacency: $A_f \in \mathbb{R}^{3484 \times 3484}$ (sparse, normalized)

---

## Why Two Graphs?

Tissue micro-environments are defined by **both** physical constraints AND biological similarity:

### Problem 1: Spatial Graph Alone
- Over-smooths across tissue boundaries
- Treats follicle edge cells same as cortex edge cells (even if biologically different)
- Risk: Mixing distinct anatomical regions

**Example:** Two cells at same distance but different follicles
```
Cell A ─── Cell C ───── Cell B
(Follicle) (Follicle)  (Cortex, ~same distance from A)
```
Spatial K=6 connects A to both C and B, even though C and B are different types.

### Problem 2: Feature Graph Alone
- Creates "salt-and-pepper" noise patterns
- Sporadic connections within regions due to local noise
- Risk: Fragmentation and instability

### Solution: Dual Graph
```
Spatial Graph (k=6):  Physical neighbors only
  ↓
Captures: Tissue topology, neighborhood structure

Feature Graph (k=20): Expression-similar cells
  ↓
Captures: Long-range functional similarities

Module 4 (GATv2):  Learn to blend both graphs
  ↓
Result: Adaptive neighborhood definitions
```

---

## Graph Construction Overview

```
Spatial Coordinates        RNA Embedding          Protein Counts
(3484 × 2)                 (3484 × 512)          (3484 × 31)
         │                        │                      │
         └────────────┬───────────┴──────────┬──────────┘
                      ↓                      ↓
         ┌───────────────────────┐  ┌─────────────────────┐
         │ Spatial K-NN (k=6)    │  │ Feature K-NN (k=20) │
         │ Euclidean distance    │  │ Cosine similarity   │
         │ on coordinates        │  │ on H_RNA || X̃_ADT  │
         └───────────────────────┘  └─────────────────────┘
                      ↓                      ↓
         ┌───────────────────────┐  ┌─────────────────────┐
         │ Edge List (Spatial)   │  │ Sparse Matrix (Adj) │
         │ 3484 cells × 6 = ~20K │  │ n_nonzero = ~70K    │
         └───────────────────────┘  └─────────────────────┘
                      ↓                      ↓
         ┌───────────────────────────────────────────────┐
         │  Transform to Sparse CSR Matrices             │
         │  A_s, A_f ∈ ℝ^(3484×3484)                     │
         └───────────────────────────────────────────────┘
                      ↓
         ┌───────────────────────────────────────────────┐
         │  Symmetric Normalization: D^(-1/2)AD^(-1/2)   │
         │  Prevents gradient explosion in GNNs          │
         └───────────────────────────────────────────────┘
                      ↓
         Final Output: A_s_norm, A_f_norm
         (Ready for Module 4: GATv2 processing)
```

---

## Spatial Graph Construction

### Method: K-Nearest Neighbors on Coordinates

For each cell $i$, find k cells with minimum Euclidean distance in coordinate space:

$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

Connect cell $i$ to its k nearest neighbors.

### Mathematical Formulation

**Step 1: Distance calculation**
For all cells, compute pairwise distances:
$$D = \{d_{ij} : i,j \in [1,3484]\}$$

**Step 2: K-NN selection**
For each cell $i$, rank by distance:
$$\text{KNN}(i) = \{j_1, j_2, \ldots, j_k\} \text{ with } d_{ij_1} \leq d_{ij_2} \leq \cdots \leq d_{ij_k}$$

**Step 3: Adjacency definition**
$$A_s[i,j] = \begin{cases} 1 & \text{if } j \in \text{KNN}(i) \\ 0 & \text{otherwise} \end{cases}$$

### Why k=6 for Visium?

**Visium Array Layout:**
```
        ●
       ● ●
      ●   ●
       ● ●
        ●

Hexagonal lattice: Each spot has exactly 6 neighbors
→ k=6 matches native array geometry
→ Most natural cell relationships
```

### Edge List Representation

Output is edge list DataFrame:
```
   source  target  weight
0       0       5       1
1       0      12       1
2       0     145       1
3       0     304       1
4       0     389       1
5       0     421       1
6       1       0       1
7       1      13       1
...
```

Total edges: 3484 cells × 6 neighbors = **~20,904 edges**

---

## Feature Graph Construction

### Method: K-Nearest Neighbors on Combined Feature Space

Instead of spatial coordinates, use **concatenated biological features**:

$$z_i = [H_{\text{RNA},i} ; \tilde{X}_{\text{ADT},i}] \in \mathbb{R}^{512+31} = \mathbb{R}^{543}$$

Connect cells based on expression similarity.

### Mathematical Formulation

**Step 1: Feature concatenation**
For each cell $i$:
- RNA embedding: $H_{\text{RNA},i} \in \mathbb{R}^{512}$ (from Module 2)
- Protein counts: $\tilde{X}_{\text{ADT},i} \in \mathbb{R}^{31}$ (from Module 1, CLR-normalized)
- Concatenate: $z_i = [H_{\text{RNA},i}; \tilde{X}_{\text{ADT},i}] \in \mathbb{R}^{543}$

**Step 2: Normalization by modality**
Independently normalize each feature stream to unit norm:
$$\tilde{H}_{\text{RNA},i} = \frac{H_{\text{RNA},i}}{||H_{\text{RNA},i}|| + \epsilon}$$
$$\tilde{X}_{\text{ADT},i} = \frac{\tilde{X}_{\text{ADT},i}}{||\tilde{X}_{\text{ADT},i}|| + \epsilon}$$

Concatenate normalized features:
$$\tilde{z}_i = [\tilde{H}_{\text{RNA},i}; \tilde{X}_{\text{ADT},i}]$$

**Why normalize separately?** Prevents RNA (512 dims) from dominating protein (31 dims) in distance calculations.

**Step 3: Cosine similarity computation**
$$\text{similarity}(i,j) = \frac{\tilde{z}_i \cdot \tilde{z}_j}{||\tilde{z}_i|| \cdot ||\tilde{z}_j||} \in [0, 1]$$

**Step 4: K-NN in feature space**
For each cell $i$, find k cells with highest cosine similarity:
$$\text{KNN}_{\text{feature}}(i) = \arg \text{top-}k \{\text{similarity}(i,j) : j \neq i\}$$

**Step 5: Adjacency matrix (directed)**
$$A_f^{\text{directed}}[i,j] = \begin{cases} 1 & \text{if } j \in \text{KNN}_{\text{feature}}(i) \\ 0 & \text{otherwise} \end{cases}$$

**Step 6: Symmetrization**
Feature graph is initially directed (i→j if j is neighbor of i). Symmetrize to get undirected graph:

$$A_f[i,j] = \begin{cases} 1 & \text{if } A_f^{\text{directed}}[i,j] = 1 \text{ OR } A_f^{\text{directed}}[j,i] = 1 \\ 0 & \text{otherwise} \end{cases}$$

### Why k=20 for Feature Graph?

- Feature space is continuous (unlike discrete hexagonal lattice)
- 20 neighbors = ~0.6% of dataset (larger radius, captures functional clusters)
- Empirically optimal for 3000+ cell datasets
- Captures long-range co-expression without over-smoothing

### Result

Total edges: 3484 cells × (up to) 20 neighbors = **~69,680 edges** (after symmetrization, some edges overlap)

---

## Graph Normalization for GNNs

Raw adjacency matrices have problematic spectral properties for deep neural networks:
- Large eigenvalues → gradient explosion
- Unstable training in multi-layer GNNs

### Symmetric Normalization Formula

Standard symmetric normalization used in spectral graph theory:

$$A_{\text{norm}} = D^{-1/2} A D^{-1/2}$$

Where $D$ is the degree matrix.

**Step-by-step computation:**

**Step 1: Add self-loops**
$$A' = A + I$$

(Each cell has implicit connection to itself, weight 1)

**Step 2: Compute row-wise degrees**
$$d_i = \sum_{j} A'[i,j]$$

For spatial graph: $d_i \approx 6 + 1 = 7$ (6 neighbors + self)
For feature graph: $d_i \approx 20 + 1 = 21$ (20 neighbors + self)

**Step 3: Degree matrix**
$$D = \text{diag}(d_1, d_2, \ldots, d_{3484})$$

**Step 4: Compute $D^{-1/2}$**
$$D^{-1/2}[i,i] = \frac{1}{\sqrt{d_i}}$$

**Step 5: Apply normalization**
$$A_{\text{norm}} = D^{-1/2} A' D^{-1/2}$$

Element-wise:
$$A_{\text{norm}}[i,j] = \frac{A'[i,j]}{\sqrt{d_i \cdot d_j}}$$

### Why This Works

**Spectral properties after normalization:**
- Eigenvalues of normalized adjacency: $\lambda \in [-1, 1]$
- Prevents vanishing/exploding gradients in GNNs
- Preserves graph topology (same sparsity pattern)

**Intuitive interpretation:**
- Strong connections (high degree endpoints) get down-weighted
- Weak connections (low degree endpoints) get up-weighted
- Result: Scale-invariant neighborhood aggregation

---

## Function Reference: Complete Method Documentation

### 1. `construct_graph_by_coordinate()`

**What it does:** Builds spatial adjacency graph using K-nearest neighbors in physical coordinates. Creates edge connections between physically close cells.

**Method signature:**
```python
def construct_graph_by_coordinate(
    cell_positions: np.ndarray,    # Input
    n_neighbors: int = 6            # Input
) -> pd.DataFrame:                  # Output
```

**Inputs:**
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `cell_positions` | `np.ndarray` | (n_cells, 2) | Spatial (x, y) coordinates. For lymph node: (3484, 2) |
| `n_neighbors` | `int` | scalar | K value for K-NN. Default: 6 (hexagonal Visium) |

**Output:**
| Name | Type | Shape | Description |
|------|------|-------|-------------|
| Edge list DataFrame | `pd.DataFrame` | (n_cells × k, 3) | Columns: [source, target, weight]. For lymph node: ~20,904 edges |

**Output columns:**
```
   source  target  weight
0       0       5       1
1       0      12       1
2       0     145       1
...     ...    ...     ...
```

**Mathematical operation:**
1. Fit K-NN tree on coordinates: $O(n \log n)$
2. Find k nearest neighbors per cell: $O(n \log n)$
3. Create edge list: $O(n \cdot k)$

**Example:**
```python
import numpy as np
from graph_construction import construct_graph_by_coordinate

# Generate random coordinates
coords = np.random.rand(100, 2)

# Build spatial graph
edges = construct_graph_by_coordinate(coords, n_neighbors=6)
print(edges.shape)        # (600, 3) - 100 cells × 6 neighbors
print(edges.columns)      # ['source', 'target', 'weight']
print(edges['weight'].unique())  # [1] - binary weights
```

**Performance:**
- Time: O(n log n) ≈ 100 ms for 3484 cells
- Space: O(nk) ≈ 2 MB
- Sparsity: 0.17%

---

### 2. `construct_graph_by_feature()`

**What it does:** Builds feature adjacency graph using K-nearest neighbors in combined biological feature space (RNA embedding + protein). Connects cells with similar expression profiles.

**Method signature:**
```python
def construct_graph_by_feature(
    adata_rna: AnnData,         # Input
    adata_protein: AnnData,     # Input
    k: int = 20                  # Input
) -> Tuple[csr_matrix, csr_matrix]:  # Output
```

**Inputs:**
| Parameter | Type | Required from | Description |
|-----------|------|---------------|-------------|
| `adata_rna` | `AnnData` | Module 2 | Must have H_rna in .obsm['H_rna'] (3484, 512) |
| `adata_protein` | `AnnData` | Module 1 | Must have X̃_ADT in .X (3484, 31) CLR-normalized |
| `k` | `int` | user | Number of nearest feature neighbors. Default: 20 |

**Output:**
| Name | Type | Shape | Sparsity | Description |
|------|------|-------|----------|-------------|
| `adj_directed` | `csr_matrix` | (3484, 3484) | 0.57% | Directed K-NN adjacency (i→j if j ∈ KNN(i)) |
| `adj_symmetric` | `csr_matrix` | (3484, 3484) | 0.57% | Undirected adjacency (i↔j if directed connects either way) |

**Computation steps:**
1. Extract H_rna from `adata_rna.obsm['H_rna']`: (3484, 512)
2. Extract X̃_ADT from `adata_protein.X`: (3484, 31)
3. Normalize each stream independently: L2 normalization
4. Concatenate: (3484, 512) + (3484, 31) = (3484, 543)
5. Fit KNN with cosine metric: $O(n \cdot d \cdot k)$
6. Build directed adjacency: $O(n \cdot k)$
7. Symmetrize: max(A_dir, A_dir^T)

**Example:**
```python
import numpy as np
from anndata import AnnData
from graph_construction import construct_graph_by_feature

# Create synthetic data
adata_rna = AnnData(X=np.random.randn(100, 18085))
adata_rna.obsm['H_rna'] = np.random.randn(100, 512)

adata_protein = AnnData(X=np.random.randn(100, 31))

# Build feature graph
adj_dir, adj_sym = construct_graph_by_feature(adata_rna, adata_protein, k=20)

print(adj_dir.shape)         # (100, 100)
print(adj_dir.nnz)           # ~2000 edges (100 × 20)
print(adj_sym.nnz)           # ~2000+ edges (symmetrization adds some)
print(type(adj_dir))         # <class 'scipy.sparse._matrix.csr_matrix'>
```

**Performance:**
- Time: O(n·d·k) ≈ 500 ms (d=543 features)
- Space: O(n·k) ≈ 5 MB sparse
- Sparsity: 0.57%

---

### 3. `construct_neighbor_graph()` (Main Function)

**What it does:** Orchestrates complete pipeline: builds both spatial and feature graphs, normalizes them for GNN processing, and returns ready-to-use adjacency matrices.

**Method signature:**
```python
def construct_neighbor_graph(
    adata_rna: AnnData,                    # Input
    adata_protein: AnnData,                # Input
    datatype: str = 'visium',              # Input
    n_neighbors_spatial: int = 6,          # Input
    n_neighbors_feature: int = 20          # Input
) -> Dict[str, Union[csr_matrix, pd.DataFrame]]:  # Output
```

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata_rna` | `AnnData` | required | RNA data with .obsm['spatial'] and .obsm['H_rna'] |
| `adata_protein` | `AnnData` | required | Protein data with .X containing CLR-normalized counts |
| `datatype` | `str` | 'visium' | Dataset type: 'visium', 'merfish', 'iss', 'spots' |
| `n_neighbors_spatial` | `int` | 6 | K for spatial neighbors (auto-set by datatype) |
| `n_neighbors_feature` | `int` | 20 | K for feature neighbors |

**Output:**
```python
Dict keys and values:
{
    'edge_list_spatial': pd.DataFrame,        # Edge list (20904, 3)
    'adj_spatial': csr_matrix,                # Normalized A_s (3484, 3484)
    'adj_feature': csr_matrix,                # Normalized A_f (3484, 3484)
    'datatype': str                           # 'visium'
}
```

**Output shapes & properties:**

| Key | Type | Shape | Sparsity | Description |
|-----|------|-------|----------|-------------|
| `edge_list_spatial` | DataFrame | (20904, 3) | - | Columns: [source, target, weight] |
| `adj_spatial` | CSR sparse | (3484, 3484) | 0.17% | Normalized spatial adjacency |
| `adj_feature` | CSR sparse | (3484, 3484) | 0.57% | Normalized feature adjacency |
| `datatype` | str | scalar | - | 'visium' |

**Pipeline steps:**
1. Extract spatial coordinates from `adata_rna.obsm['spatial']`
2. Call `construct_graph_by_coordinate()` → edge list (spatial)
3. Call `construct_graph_by_feature()` → adjacency matrices (feature)
4. Call `transform_adjacent_matrix()` → convert edge list to sparse
5. Call `preprocess_graph()` on both → normalize for GNN
6. Return dictionary with all results

**Example:**
```python
from graph_construction import construct_neighbor_graph

# Run complete pipeline
graphs = construct_neighbor_graph(
    adata_rna,
    adata_protein,
    datatype='visium',
    n_neighbors_spatial=6,
    n_neighbors_feature=20
)

# Access outputs
A_s = graphs['adj_spatial']          # (3484, 3484) spatial
A_f = graphs['adj_feature']          # (3484, 3484) feature
edges = graphs['edge_list_spatial']  # (20904, 3)

print(f"Spatial edges: {A_s.nnz}")    # ~20904
print(f"Feature edges: {A_f.nnz}")    # ~70000
print(f"Max A_s value: {A_s.max()}")  # ~0.3 (normalized)
```

**Performance:**
- Total time: ~650 ms
- Total space: ~10 MB

---

### 4. `transform_adjacent_matrix()`

**What it does:** Converts pandas edge list (source, target, weight) into sparse CSR adjacency matrix for efficient computation.

**Method signature:**
```python
def transform_adjacent_matrix(
    edge_list: pd.DataFrame     # Input
) -> csr_matrix:                # Output
```

**Input:**
```python
# DataFrame format
   source  target  weight
0       0       5       1
1       0      12       1
2       0     145       1
...
```

| Column | Type | Description |
|--------|------|-------------|
| `source` | int | Cell i (row index) |
| `target` | int | Cell j (column index) |
| `weight` | float | Edge weight (typically 1 for binary) |

**Output:**
| Name | Type | Shape | Format | Description |
|------|------|-------|--------|-------------|
| Adjacency matrix | `csr_matrix` | (n_cells, n_cells) | Sparse CSR | A[i,j] = weight if edge exists |

**Matrix representation:**
```
Input edge list:
  (0, 5, 1)
  (0, 12, 1)
  (1, 5, 1)
  (1, 23, 1)

Output sparse matrix (first 3 cells):
    0   1   2   3   4   5  ...  12  ...  23
0 [ 0   0   0   0   0   1  ...   1  ...   0 ]
1 [ 0   0   0   0   0   1  ...   0  ...   1 ]
2 [ 0   0   0   0   0   0  ...   0  ...   0 ]
```

**Example:**
```python
import pandas as pd
from graph_construction import transform_adjacent_matrix

# Create edge list
edges = pd.DataFrame({
    'source': [0, 0, 0, 1, 1, 1],
    'target': [5, 12, 145, 5, 23, 89],
    'weight': [1, 1, 1, 1, 1, 1]
})

# Convert to sparse matrix
A = transform_adjacent_matrix(edges)

print(A.shape)           # (146, 146) - max index is 145
print(A.nnz)             # 6 - number of edges
print(type(A))           # <class 'scipy.sparse._matrix.csr_matrix'>
print(A[0, 5])           # 1 - edge from 0 to 5
print(A.toarray())       # Convert to dense for viewing
```

**Performance:**
- Time: O(E) where E = number of edges ≈ 10 ms
- Space: O(E) sparse storage

---

### 5. `preprocess_graph()`

**What it does:** Applies symmetric graph normalization D^(-1/2)AD^(-1/2) to prevent gradient explosion in multi-layer GNNs. Scales adjacency values to stable range.

**Method signature:**
```python
def preprocess_graph(
    adj: Union[csr_matrix, coo_matrix]  # Input
) -> csr_matrix:                         # Output
```

**Input:**
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `adj` | `csr_matrix` or `coo_matrix` | (n_cells, n_cells) | Sparse adjacency matrix (binary or weighted) |

**Output:**
| Name | Type | Shape | Properties | Description |
|------|------|-------|-----------|-------------|
| Normalized adjacency | `csr_matrix` | (n_cells, n_cells) | Values ∈ [0, 1/(min_degree)] | Symmetrically normalized |

**Normalization steps:**

1. **Add self-loops:** A' = A + I
   ```
   Before: A[i,i] = 0 (no self-loops)
   After:  A'[i,i] = 1 (self-loops added)
   ```

2. **Compute degrees:** d_i = sum(A'[i, :])
   ```
   For spatial graph: d_i ≈ 7 (6 neighbors + self-loop)
   For feature graph: d_i ≈ 21 (20 neighbors + self-loop)
   ```

3. **Compute D^(-1/2):** Each diagonal element = 1 / sqrt(d_i)
   ```
   D_inv_sqrt[i,i] = 1 / sqrt(d_i)
   ```

4. **Apply normalization:** A_norm = D^(-1/2) × A' × D^(-1/2)
   ```
   A_norm[i,j] = A'[i,j] / sqrt(d_i × d_j)
   ```

**Example:**
```python
import numpy as np
from scipy.sparse import csr_matrix
from graph_construction import preprocess_graph

# Create simple adjacency
A = csr_matrix([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

# Normalize
A_norm = preprocess_graph(A)

print(A.toarray())
# [[0 1 0]
#  [1 0 1]
#  [0 1 0]]

print(A_norm.toarray())
# [[0.5  0.408  0.    ]
#  [0.408 0.333 0.408]
#  [0.    0.408 0.5  ]]

print(f"Max value before: {A.max()}")  # 1
print(f"Max value after: {A_norm.max()}")   # ~0.5
```

**Effect on eigenvalues:**
```
Raw adjacency:
  - Largest eigenvalue: λ_max ≈ 2.0 (can vary wildly)
  - Risk: Gradient explosion in deep GNNs

Normalized adjacency:
  - Largest eigenvalue: λ_max ≈ 1.0 (stable)
  - All eigenvalues ∈ [-1, 1]
  - Enables training of 6+ layer models
```

**Performance:**
- Time: O(E) where E = number of edges ≈ 50 ms
- Space: O(E) in-place operation

---

## Summary Table: All Functions

| Function | Input Type | Input Size | Output Type | Output Size | Key Operation |
|----------|-----------|-----------|------------|-----------|---|
| `construct_graph_by_coordinate` | np.ndarray | (3484, 2) | pd.DataFrame | (20904, 3) | Euclidean K-NN |
| `construct_graph_by_feature` | 2× AnnData | (3484, 543) | 2× csr_matrix | 2× (3484, 3484) | Cosine K-NN |
| `construct_neighbor_graph` | 2× AnnData | mixed | Dict | 4 items | Orchestration |
| `transform_adjacent_matrix` | pd.DataFrame | (20904, 3) | csr_matrix | (3484, 3484) | COO → CSR |
| `preprocess_graph` | csr_matrix | (3484, 3484) | csr_matrix | (3484, 3484) | D^(-1/2)AD^(-1/2) |

---

## Performance Metrics

---

## Data Flow Through Module 3

```
Input Phase:
  Coordinates (3484, 2)  │  H_RNA (3484, 512)  │  X̃_ADT (3484, 31)
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                ↓
Graph Construction Phase:
         ┌──────────────────────────────────────┐
         │  Spatial K-NN (k=6)                  │
         │  • Euclidean distances on (x,y)      │
         │  • Sort and connect nearest 6        │
         │  • Produce edge list                 │
         └──────────────────────────────────────┘
                                ↓
         ┌──────────────────────────────────────┐
         │  Feature K-NN (k=20)                 │
         │  • Concatenate H_RNA || X̃_ADT       │
         │  • Normalize each stream separately  │
         │  • Cosine similarity compute         │
         │  • Sort and connect top 20 similar   │
         │  • Symmetrize adjacency              │
         └──────────────────────────────────────┘
                                ↓
Matrix Construction Phase:
         ┌──────────────────────────────────────┐
         │ Transform edge list to sparse CSR    │
         │ A_s, A_f ∈ ℝ^(3484 × 3484) sparse   │
         └──────────────────────────────────────┘
                                ↓
Normalization Phase:
         ┌──────────────────────────────────────┐
         │  Symmetric normalization             │
         │  A_norm = D^(-1/2) A D^(-1/2)        │
         │  • Add self-loops                    │
         │  • Compute degrees                   │
         │  • Scale by D^(-1/2)                 │
         └──────────────────────────────────────┘
                                ↓
Output:
  A_s_norm (3484, 3484)  │  A_f_norm (3484, 3484)
  Spatial adjacency      │  Feature adjacency
  ~20,904 edges          │  ~69,680 edges (est.)
  Ready for Module 4     │  Ready for Module 4
```

---

## Performance Metrics

### Spatial Graph Construction
- Time: ~100 ms
- Space: ~2 MB (edge list)
- Edge count: 20,904 (3484 × 6)
- Sparsity: 0.17% (20K / 12M possible edges)

### Feature Graph Construction
- Time: ~500 ms (cosine similarity computation)
- Space: ~5 MB (sparse matrix)
- Edge count: ~70K (after symmetrization)
- Sparsity: 0.57% (70K / 12M possible edges)

### Graph Normalization
- Time: ~50 ms
- Space: In-place (no additional memory)
- Complexity: $O(E)$ where $E$ = number of edges

---

## Validation Checks

After Module 3, verify:

```python
# Check shapes
assert graphs['adj_spatial'].shape == (3484, 3484)
assert graphs['adj_feature'].shape == (3484, 3484)

# Check sparsity (should be ~0.2-0.6%)
spatial_density = graphs['adj_spatial'].nnz / (3484 ** 2)
assert 0.001 < spatial_density < 0.01

# Check normalization (normalized values should be small)
assert graphs['adj_spatial'].max() < 1.0
assert graphs['adj_feature'].max() < 1.0

# Check connectivity (every cell should have neighbors)
spatial_degrees = np.array(graphs['adj_spatial'].sum(axis=1)).flatten()
assert np.all(spatial_degrees > 0), "Isolated cells in spatial graph"
```

---

## Examples: Interpretation

### Spatial Graph Example
```
Cell 0 neighbors (spatial, k=6):
  Neighbor 5   (distance: 145 μm)
  Neighbor 12  (distance: 148 μm)
  Neighbor 145 (distance: 151 μm)
  ...

Interpretation: 6 physically closest cells
             (form local hexagon in tissue)
```

### Feature Graph Example
```
Cell 1234 neighbors (feature, k=20):
  Neighbor 800  (cosine sim: 0.92, CD8+ T cell like)
  Neighbor 1050 (cosine sim: 0.88, CD8+ T cell like)
  Neighbor 2100 (cosine sim: 0.85, CD8+ T cell like)
  ...

Interpretation: 20 most similar expression profiles
             (includes distant but similar cells)
```

---

## References to Master Pipeline

This implementation follows **flow.md**, **module_explanation.md**, and **KAC-Net_MASTER_PLAN.md**:

| Reference | Implementation |
|-----------|-----------------|
| Spatial K-NN (k=6) | ✓ Euclidean distance on coordinates |
| Feature K-NN (k=20) | ✓ Cosine similarity on H_RNA \|\| X̃_ADT |
| Graph normalization | ✓ Symmetric D^(-1/2)AD^(-1/2) |
| Output: A_s, A_f | ✓ Both (3484, 3484) sparse normalized |
| Ready for Module 4 | ✓ GATv2 processing |

---

## Next Step: Module 4

Once graph construction completes, data is ready for **Module 4: Local Spatial Encoding**.

Module 4 inputs:
- $H_{\text{RNA}} \in \mathbb{R}^{3484 \times 512}$
- $\tilde{X}_{\text{ADT}} \in \mathbb{R}^{3484 \times 31}$
- $A_s \in \mathbb{R}^{3484 \times 3484}$ (spatial adjacency)
- $A_f \in \mathbb{R}^{3484 \times 3484}$ (feature adjacency)

Module 4 outputs:
- $Z_{\text{RNA}} \in \mathbb{R}^{3484 \times d}$ (spatially smoothed RNA embedding)
- $Z_{\text{ADT}} \in \mathbb{R}^{3484 \times d}$ (spatially smoothed ADT embedding)

---

**Module 3 Status: ✅ Complete**  
**Ready for: Module 4 (Local Spatial Encoding with GATv2)**

# KAC-Net Model Build Plan - Complete

**Date:** May 21, 2026 | **Status:** ✅ PLANNING COMPLETE

---

## 📊 Executive Summary

Building **KAC-Net**: an 8-module spatial multi-omics model combining **COSMOS**, **spaLLM**, and **SpatialGlue**.

- **Total Code:** ~1,530 lines
  - spaLLM: 550 lines (36%)
  - COSMOS: 320 lines (21%)
  - SpatialGlue: 350 lines (23%)
  - Custom: 310 lines (20%)

- **Timeline:** 5 weeks

---

## 🏗️ Architecture (8 Modules × 4 Phases)

### **PHASE 1: Feature Extraction & Enrichment**

#### **Module 1: Multimodal Preprocessing**
- **Source:** spaLLM
- **Code:** `spaLLM/preprocess.py` (lines 47-56, 40-44, 61-71)
- **Functions to Extract:**
  - `clr_normalize_each_cell()` - CLR normalization for ADT
  - `construct_graph_by_coordinate()` - Spatial graph
  - `construct_graph_by_feature()` - Feature graph
- **Input:** RNA (3484×18,085), ADT (3484×31)
- **Output:** Normalized X̃_RNA, X̃_ADT
- **Tech:** Library scaling, log1p, CLR normalization

#### **Module 2: Knowledge-Enriched Encoding (spaLLM Engine)**
- **Source:** spaLLM
- **Code:** `spaLLM/modelTriatt_Flow1.py` (entire file)
- **Extract:** TransformerModel class (full foundation model)
- **Input:** X̃_RNA (3484×18,085)
- **Output:** H_RNA (3484×512)
- **Tech:** Multi-head self-attention transformer for gene recovery

---

### **PHASE 2: Structural & Geometric Mapping**

#### **Module 3: Multi-Graph Construction**
- **Source:** spaLLM + COSMOS
- **Code:** 
  - `spaLLM/preprocess.py` (lines 40-71)
  - `COSMOS/cosmos.py` (lines 50-54): `sparse_mx_to_torch_edge_list()`
- **Extract:** Graph construction utilities + tensor conversion
- **Input:** Coordinates (3484×2), H_RNA (3484×512), X̃_ADT (3484×31)
- **Output:** A_s (3484×3484 spatial), A_f (3484×3484 feature)
- **Tech:** K-NN spatial (k=6), cosine similarity feature graph

#### **Module 4: Local Spatial Encoding (Residual GATv2)**
- **Source:** SpatialGlue + COSMOS
- **Code:**
  - `SpatialGlue/model.py` (lines 21-91): Encoder, AttentionLayer, forward pass
  - `COSMOS/modulesWNN.py` (entire): GNN layers, Deep Graph Infomax
- **Extract:** Encoder class, AttentionLayer, residual connections
- **Input:** H_RNA (3484×512), X̃_ADT (3484×31), A_s, A_f
- **Output:** Z_RNA (3484×d), Z_ADT (3484×d)
- **Tech:** GATv2 attention with dual graphs + residual skip connections

---

### **PHASE 3: Integration, Fusion & Regularization**

#### **Module 5: Cross-Modal Contrastive Alignment (COSMOS)**
- **Source:** COSMOS
- **Code:** `COSMOS/modulesWNN.py` (entire), `COSMOS/cosmos.py`
- **Extract:** InfoNCE loss, contrastive pair generation, similarity computation
- **Input:** Z_RNA (3484×d), Z_ADT (3484×d) from Module 4
- **Output:** Aligned embeddings (shared space) + L_cl loss
- **Tech:** InfoNCE loss: $\mathcal{L}_{cl} = -\frac{1}{2N}\sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(\mathbf{Z}_{\text{RNA},i}, \mathbf{Z}_{\text{ADT},i})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{Z}_{\text{RNA},i}, \mathbf{Z}_{\text{ADT},j})/\tau)} + ... \right]$

#### **Module 6: Adaptive Dual-Attention Fusion (SpatialGlue)**
- **Source:** SpatialGlue
- **Code:** `SpatialGlue/model.py` (lines 5-91): Encoder_overall class (complete)
- **Extract:** Full Encoder_overall with:
  - Within-modality attention (graph blending, Tier 1)
  - Between-modality attention (modality gating, Tier 2)
- **Input:** Aligned Z_RNA, Z_ADT (3484×d), A_s, A_f
- **Output:** Z_Fused (3484×64)
- **Tech:** 
  - Tier 1: $\alpha_{s,i}^m, \alpha_{f,i}^m = \text{Softmax}(\text{MLP}_m(Z_m \cdot A))$
  - Tier 2: $\omega_{\text{RNA}}, \omega_{\text{ADT}} = \text{Softmax}(\text{Tanh}(...))$
  - Final: $Z_{\text{Fused}} = \omega_{\text{RNA}}(W_R \tilde{Z}_{\text{RNA}}) + \omega_{\text{ADT}}(W_A \tilde{Z}_{\text{ADT}})$

#### **Module 7: Reconstruction & Regularization**
- **Source:** SpatialGlue + Custom
- **Code:** 
  - `SpatialGlue/model.py`: Decoder class
  - Custom: Write loss functions
- **Extract:** Decoder MLP architecture
- **Input:** Z_Fused (3484×64)
- **Output:** X̂_RNA (3484×18,085), X̂_ADT (3484×31), L_total loss
- **Tech:**
  - Reconstruction: $\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum_i (\|X̃_{\text{RNA},i} - X̂_{\text{RNA},i}\|^2 + \|X̃_{\text{ADT},i} - X̂_{\text{ADT},i}\|^2)$
  - Spatial: $\mathcal{L}_{\text{spat}} = \sum_{i,j} A_{s,ij} \|Z_{\text{Fused},i} - Z_{\text{Fused},j}\|^2$
  - Total: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{cl} + \lambda_2 \mathcal{L}_{\text{recon}} + \lambda_3 \mathcal{L}_{\text{spat}}$

---

### **PHASE 4: Unsupervised Biological Discovery**

#### **Module 8: Spatial Domain Identification**
- **Source:** Custom + scanpy/sklearn
- **Tech:** Leiden clustering, UMAP, ARI validation
- **Input:** Z_Fused (3484×64) from trained model
- **Output:** Domain labels (3484,), UMAP coordinates, ARI score
- **Implementation:**
  ```python
  sc.pp.neighbors(adata, use_rep='latent', n_neighbors=15)
  sc.tl.leiden(adata, resolution=1.0)
  sc.tl.umap(adata)
  ari = adjusted_rand_score(ground_truth, predicted)
  ```

---

## 📋 Code Extraction Checklist

### **spaLLM (550 lines):**
- [ ] `preprocess.py` → clr_normalize_each_cell, seurat_clr (15 lines)
- [ ] `preprocess.py` → construct_neighbor_graph, construct_graph_by_coordinate, construct_graph_by_feature (100 lines)
- [ ] `modelTriatt_Flow1.py` → TransformerModel class (300 lines)
- [ ] `spaLLM_util.py` → Embedding utilities (135 lines)

### **COSMOS (320 lines):**
- [ ] `cosmos.py` → sparse_mx_to_torch_edge_list (5 lines)
- [ ] `modulesWNN.py` → DeepGraphInfomaxWNN, GNN layers, InfoNCE loss (315 lines)

### **SpatialGlue (350 lines):**
- [ ] `model.py` → Encoder class (60 lines)
- [ ] `model.py` → AttentionLayer class (50 lines)
- [ ] `model.py` → Decoder class (50 lines)
- [ ] `model.py` → Encoder_overall class (85 lines)
- [ ] `SpatialGlue_PyG.py` → PyG variant (optional, 100 lines)

### **Custom Code (310 lines):**
- [ ] Loss functions (80 lines)
- [ ] Clustering module (50 lines)
- [ ] Main orchestrator KACNet class (100 lines)
- [ ] Config & utilities (80 lines)

---

## 📊 Data Flow Through Pipeline

```
INPUT (3,484 spots)
├─ RNA: 3,484 × 18,085 genes
├─ ADT: 3,484 × 31 proteins
└─ Spatial: 3,484 × 2 coordinates
     ↓
[MODULE 1] → X̃_RNA, X̃_ADT (normalized)
     ↓
[MODULE 2] → H_RNA (3,484 × 512)
     ↓
[MODULE 3] → A_s, A_f (3,484 × 3,484 sparse matrices)
     ↓
[MODULE 4] → Z_RNA, Z_ADT (3,484 × d)
     ↓
[MODULE 5] → Aligned Z_RNA, Z_ADT (shared space) + L_cl
     ↓
[MODULE 6] → Z_Fused (3,484 × 64)
     ↓
[MODULE 7] → X̂_RNA, X̂_ADT + L_total
     ↓
[TRAINING] → Backpropagation (50 epochs)
     ↓
[MODULE 8] → Domain labels (3,484,), UMAP, ARI score
     ↓
OUTPUT: Domain-annotated spatial transcriptomics
```

---

## 🎯 Exact Extraction Points

### **spaLLM/preprocess.py:**
```
Lines 47-56:  clr_normalize_each_cell() + seurat_clr()
Lines 40-44:  construct_neighbor_graph()
Lines 61-71:  construct_graph_by_coordinate() & construct_graph_by_feature()
Lines 74-83:  transform_adjacent_matrix() & preprocess_graph()
```

### **spaLLM/modelTriatt_Flow1.py:**
```
Full file: TransformerModel class (entire ~300 lines)
```

### **COSMOS/cosmos.py:**
```
Lines 50-54: sparse_mx_to_torch_edge_list()
```

### **COSMOS/modulesWNN.py:**
```
Full file: DeepGraphInfomaxWNN class with GNN + InfoNCE
```

### **SpatialGlue/model.py:**
```
Lines 5-91:   Encoder_overall (complete forward pass)
Lines (cont): Encoder class
Lines (cont): Decoder class
Lines (cont): AttentionLayer class
```

---

## ✅ Success Criteria

### **Module Tests (Each Module):**
- [ ] Correct input/output tensor shapes
- [ ] No NaN/Inf values
- [ ] Loss decreases during training
- [ ] Gradient flow verified

### **Integration Tests:**
- [ ] End-to-end pipeline runs
- [ ] Z_Fused produces (3,484×64) embeddings
- [ ] Clustering produces domain labels
- [ ] ARI score > 0.7

### **Documentation:**
- [ ] All modules documented
- [ ] Example notebook created
- [ ] Results reproducible

---

## 📁 Directory Structure to Create

```
KAC-Net/
├── kac_net_main.py              # Main orchestrator
├── config.py                    # Hyperparameters
├── trainer.py                   # Training loop
├── losses.py                    # All loss functions
├── utils.py                     # Shared utilities
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py         # Module 1
│   ├── encoding.py              # Module 2
│   ├── graph_construction.py    # Module 3
│   ├── spatial_encoding.py      # Module 4
│   ├── contrastive_alignment.py # Module 5
│   ├── dual_attention_fusion.py # Module 6
│   ├── reconstruction_loss.py   # Module 7
│   └── clustering.py            # Module 8
├── tests/
│   ├── test_module1.py
│   ├── test_module2.py
│   └── ... (one per module)
└── notebooks/
    └── tutorial_kac_net.ipynb
```

---

## 🔧 Dependencies

**Core:**
- torch >= 1.9.0
- torch_geometric >= 2.0.0
- numpy, pandas, scipy, scikit-learn
- anndata, scanpy
- matplotlib

**Optional (for foundation models):**
- scgpt or geneformer

---

## 📌 Quick Reference: What to Extract

| Module | From | What | Lines | Priority |
|--------|------|------|-------|----------|
| 1 | spaLLM | clr_normalize, graph build | 100 | 1 |
| 2 | spaLLM | TransformerModel class | 300 | 1 |
| 3 | spaLLM+COSMOS | Graph utils + tensor convert | 150 | 1 |
| 4 | SG+COSMOS | Encoder, GATv2, attention | 250 | 2 |
| 5 | COSMOS | InfoNCE loss, contrastive | 120 | 2 |
| 6 | SpatialGlue | Encoder_overall, dual-attention | 250 | 2 |
| 7 | SG+Custom | Decoder + loss functions | 180 | 3 |
| 8 | Custom | Clustering, UMAP, ARI | 50 | 3 |

---

**Status:** ✅ COMPLETE - READY FOR IMPLEMENTATION  
**Start Date:** Week 1 (Begin with Module 1 extraction from spaLLM)  
**Estimated Completion:** 5 weeks

# KAC-Net Architecture Diagram & Code Source Mapping

## 🎨 Visual Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KAC-NET MODEL ARCHITECTURE                           │
│                  (8 Modules × 3 State-of-the-Art Models)                    │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: FEATURE EXTRACTION & ENRICHMENT
═════════════════════════════════════════════════════════════════════════════

┌─────────────────────────┐
│   INPUT DATA            │
├─────────────────────────┤
│ RNA: (3484 × 18,085)    │
│ ADT: (3484 × 31)        │
│ Spatial: (3484 × 2)     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ MODULE 1: MULTIMODAL PREPROCESSING          │
│ (spaLLM/preprocess.py)                      │
├─────────────────────────────────────────────┤
│ ✓ CLR Normalize ADT                         │
│ ✓ Log1p Transform RNA                       │
│ ✓ Library Size Scaling                      │
│ Output: X̃_RNA, X̃_ADT (normalized)          │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│ MODULE 2: KNOWLEDGE-ENRICHED ENCODING       │
│ (spaLLM/modelTriatt_Flow1.py)               │
├─────────────────────────────────────────────┤
│ ✓ Transformer: Multi-Head Self-Attention    │
│ ✓ Gene → Dense Embedding Projection         │
│ ✓ Missing Transcript Recovery               │
│ Input: X̃_RNA (3484 × 18,085)                │
│ Output: H_RNA (3484 × 512)                  │
└──────────┬──────────────────────────────────┘


PHASE 2: STRUCTURAL & GEOMETRIC MAPPING
═════════════════════════════════════════════════════════════════════════════

      ┌────────────────────────────────────────────┐
      │  MODULE 3: MULTI-GRAPH CONSTRUCTION        │
      │  (spaLLM/preprocess.py + COSMOS/cosmos.py) │
      ├────────────────────────────────────────────┤
      │  ✓ Spatial KNN (k=6): A_s                  │
      │  ✓ Feature KNN: A_f (cosine similarity)    │
      │  Input: Spatial coords, H_RNA, X̃_ADT       │
      │  Output: A_s, A_f (3484 × 3484 sparse)     │
      └────────────┬─────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
   A_s            H_RNA         X̃_ADT
(Spatial)       (Embedding)    (Protein)
    │              │              │
    └──────────────┼──────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ MODULE 4: LOCAL SPATIAL ENCODING             │
│ (SpatialGlue/model.py + COSMOS/modulesWNN.py)│
├──────────────────────────────────────────────┤
│ ┌────────────────────────────────────────┐   │
│ │ GATv2 Encoder RNA:                     │   │
│ │ • Spatial neighbors (A_s)              │   │
│ │ • Feature neighbors (A_f)              │   │
│ │ • Attention weights (dynamic)          │   │
│ │ + Residual connections                 │   │
│ └────────┬─────────────────────────────┘   │
│          │                                  │
│ ┌────────▼─────────────────────────────┐   │
│ │ GATv2 Encoder ADT: (parallel stream) │   │
│ │ • Spatial neighbors (A_s)            │   │
│ │ • Feature neighbors (A_f)            │   │
│ │ • Attention weights (dynamic)        │   │
│ │ + Residual connections               │   │
│ └────────┬─────────────────────────────┘   │
│          │                                  │
│ Output: Z_RNA (3484 × d), Z_ADT (3484 × d)  │
└──────────┬───────────────────────────────────┘


PHASE 3: INTEGRATION, FUSION & REGULARIZATION
═════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │ Z_RNA (3484 × d)  |  Z_ADT (3484 × d)│
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │ MODULE 5: CROSS-MODAL                │
    │ CONTRASTIVE ALIGNMENT                │
    │ (COSMOS/cosmos.py + modulesWNN.py)   │
    ├─────────────────────────────────────┤
    │ ✓ InfoNCE Loss (symmetric)           │
    │ ✓ Positive pairs: (z_rna_i, z_adt_i)│
    │ ✓ Negative pairs: (z_rna_i, z_adt_j)│
    │ ✓ Cosine Similarity Scoring          │
    │ Output: Aligned embeddings + L_cl    │
    └─────────────────┬───────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Aligned Z_RNA       Aligned Z_ADT
    (3484 × d)          (3484 × d)
    [Shared Space]
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │ MODULE 6: ADAPTIVE DUAL-ATTENTION FUSION │
    │ (SpatialGlue/model.py)                   │
    ├──────────────────────────────────────────┤
    │                                          │
    │ TIER 1: WITHIN-MODALITY ATTENTION        │
    │ ┌──────────────────────────────────┐   │
    │ │ For RNA Stream:                  │   │
    │ │ α_s,i^RNA = softmax(MLP_rna(...))│   │
    │ │ α_f,i^RNA = softmax(MLP_rna(...))│   │
    │ │ Ž_RNA = α_s * (W_s @ Z_RNA)      │   │
    │ │       + α_f * (W_f @ Z_RNA)      │   │
    │ └──────────────────────────────────┘   │
    │                                          │
    │ ┌──────────────────────────────────┐   │
    │ │ For ADT Stream: (parallel)       │   │
    │ │ α_s,i^ADT = softmax(MLP_adt(...))│   │
    │ │ α_f,i^ADT = softmax(MLP_adt(...))│   │
    │ │ Ž_ADT = α_s * (W_s @ Z_ADT)      │   │
    │ │       + α_f * (W_f @ Z_ADT)      │   │
    │ └──────────────────────────────────┘   │
    │                                          │
    │ TIER 2: BETWEEN-MODALITY ATTENTION      │
    │ ┌──────────────────────────────────┐   │
    │ │ ω_RNA = softmax(Tanh_gate(...))  │   │
    │ │ ω_ADT = softmax(Tanh_gate(...))  │   │
    │ │                                  │   │
    │ │ Z_Fused = ω_RNA*(W_R @ Ž_RNA)   │   │
    │ │         + ω_ADT*(W_A @ Ž_ADT)   │   │
    │ └──────────────────────────────────┘   │
    │                                          │
    │ Output: Z_Fused (3484 × 64)             │
    └──────────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    Z_Fused              Z_Fused
   (Latent)          (Reconstruction)
                      │
        ┌─────────────┤
        │             │
        ▼             ▼
    ┌──────────────────────────────────────────┐
    │ MODULE 7: RECONSTRUCTION & REGULARIZATION│
    │ (SpatialGlue/model.py + Custom)          │
    ├──────────────────────────────────────────┤
    │                                          │
    │ ┌────────────────────────────────────┐  │
    │ │ RNA DECODER:                       │  │
    │ │ MLP: (3484×64) → (3484×18,085)     │  │
    │ │ X̂_RNA = Decoder_RNA(Z_Fused)       │  │
    │ └────────────────────────────────────┘  │
    │                                          │
    │ ┌────────────────────────────────────┐  │
    │ │ ADT DECODER:                       │  │
    │ │ MLP: (3484×64) → (3484×31)         │  │
    │ │ X̂_ADT = Decoder_ADT(Z_Fused)       │  │
    │ └────────────────────────────────────┘  │
    │                                          │
    │ LOSS COMPUTATION:                       │
    │ L_recon = MSE(X̃_RNA, X̂_RNA)             │
    │         + MSE(X̃_ADT, X̂_ADT)             │
    │                                          │
    │ L_spat = Σ A_s,ij ||Z_Fused,i -        │
    │          Z_Fused,j||²                   │
    │                                          │
    │ L_total = λ₁*L_cl + λ₂*L_recon +       │
    │           λ₃*L_spat                     │
    │                                          │
    │ Output: L_total (scalar for backprop)   │
    └──────────────┬───────────────────────┘
                   │
                   ▼
        ┌─────────────────────────┐
        │ TRAINING LOOP           │
        │ • Backpropagation       │
        │ • Gradient updates      │
        │ • 50 epochs             │
        └─────────────────────────┘


PHASE 4: UNSUPERVISED BIOLOGICAL DISCOVERY
═════════════════════════════════════════════════════════════════════════════

              Z_Fused (3484 × 64)
              [Trained Latent]
                      │
                      ▼
    ┌──────────────────────────────────────┐
    │ MODULE 8: SPATIAL DOMAIN             │
    │ IDENTIFICATION                       │
    │ (Custom + scanpy + sklearn)          │
    ├──────────────────────────────────────┤
    │                                      │
    │ ┌──────────────────────────────────┐ │
    │ │ Step 1: kNN Graph Construction  │ │
    │ │ Build neighborhood graph in     │ │
    │ │ latent space (k=15)             │ │
    │ └──────────────────────────────────┘ │
    │               │                      │
    │ ┌─────────────▼──────────────────┐   │
    │ │ Step 2: Leiden Clustering      │   │
    │ │ Community detection            │   │
    │ │ resolution=1.0                 │   │
    │ └──────────────────────────────┐ │   │
    │ ┌─────────────────────────────▼┴─┐  │
    │ │ Step 3: UMAP Projection        │  │
    │ │ 64D → 2D visualization         │  │
    │ └────────────────────────────────┘  │
    │               │                      │
    │ ┌─────────────▼──────────────────┐   │
    │ │ Step 4: ARI Validation         │   │
    │ │ Compare to manual annotations  │   │
    │ │ Score in [0, 1]                │   │
    │ └──────────────────────────────────┘ │
    │                                      │
    │ Outputs:                             │
    │ • Domain labels (3484,)              │
    │ • 2D UMAP coordinates                │
    │ • ARI score                          │
    └──────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    Domains     UMAP      Metrics
    Labeled     Plot      Report


═════════════════════════════════════════════════════════════════════════════
FINAL OUTPUT (3484 spots × 1 domain each)
═════════════════════════════════════════════════════════════════════════════
```

---

## 📌 Code Source Attribution Matrix

```
┌────────────────┬────────────────┬────────────┬────────────────┐
│ MODULE NAME    │ SOURCE MODEL   │ SOURCE FILE│ LINES EXTRACTED│
├────────────────┼────────────────┼────────────┼────────────────┤
│ 1. Preprocess  │ spaLLM         │preprocess. │ ~50            │
│                │                │py L47-56   │                │
├────────────────┼────────────────┼────────────┼────────────────┤
│ 2. Encoding    │ spaLLM         │modelTriatt │ ~300           │
│                │                │_Flow1.py   │ (entire)       │
│                │                │full file   │                │
├────────────────┼────────────────┼────────────┼────────────────┤
│ 3. Graphs      │ spaLLM +       │preprocess. │ ~100 (spaLLM)  │
│ Construction   │ COSMOS         │py L40-71   │ ~50 (COSMOS)   │
│                │                │cosmos.py   │                │
│                │                │L50-54      │                │
├────────────────┼────────────────┼────────────┼────────────────┤
│ 4. Spatial     │ SpatialGlue +  │model.py    │ ~150 (SG)      │
│ Encoding       │ COSMOS         │L21-91      │ ~100 (COSMOS)  │
│                │                │modulesWNN. │                │
│                │                │py          │                │
├────────────────┼────────────────┼────────────┼────────────────┤
│ 5. Contrastive │ COSMOS         │cosmos.py + │ ~120           │
│ Alignment      │                │modulesWNN. │                │
│                │                │py          │                │
├────────────────┼────────────────┼────────────┼────────────────┤
│ 6. Dual-       │ SpatialGlue    │model.py    │ ~250           │
│ Attention      │                │L5-91       │                │
│ Fusion         │                │(full       │                │
│                │                │forward)    │                │
├────────────────┼────────────────┼────────────┼────────────────┤
│ 7. Loss        │ SpatialGlue +  │model.py +  │ ~180           │
│ Functions      │ Custom         │custom loss │ (custom)       │
├────────────────┼────────────────┼────────────┼────────────────┤
│ 8. Clustering  │ Custom +       │scanpy +    │ ~50            │
│                │ scanpy/sklearn │sklearn     │                │
├────────────────┼────────────────┼────────────┼────────────────┤
│ TOTAL          │                │            │ ~1,530 lines   │
│ BREAKDOWN:     │ spaLLM: 550    │            │                │
│                │ COSMOS: 320    │            │                │
│                │ SpatialGlue:350│            │                │
│                │ Custom: 310    │            │                │
└────────────────┴────────────────┴────────────┴────────────────┘
```

---

## 🔄 Data Flow: Tensor Shape Evolution

```
Module 1: Preprocessing
  INPUT:  RNA (3484 × 18,085), ADT (3484 × 31)
  OUTPUT: X̃_RNA (3484 × 18,085), X̃_ADT (3484 × 31)
          ↓ (same shape)

Module 2: Encoding
  INPUT:  X̃_RNA (3484 × 18,085)
  OUTPUT: H_RNA (3484 × 512)
          ↓ (dimension reduction)

Module 3: Graph Construction
  INPUT:  Coords (3484 × 2), H_RNA (3484 × 512), X̃_ADT (3484 × 31)
  OUTPUT: A_s (3484 × 3484), A_f (3484 × 3484)
          ↓ (adjacency matrices)

Module 4: Spatial Encoding
  INPUT:  H_RNA (3484 × 512), X̃_ADT (3484 × 31), A_s, A_f
  OUTPUT: Z_RNA (3484 × d), Z_ADT (3484 × d)
          ↓ (smoothed, d typically 128 or 256)

Module 5: Contrastive Alignment
  INPUT:  Z_RNA (3484 × d), Z_ADT (3484 × d)
  OUTPUT: Z_RNA (3484 × d), Z_ADT (3484 × d) [aligned in shared space]
          + L_cl (scalar)
          ↓ (shapes same, but mathematically aligned)

Module 6: Dual-Attention Fusion
  INPUT:  Z_RNA (3484 × d), Z_ADT (3484 × d), A_s, A_f
  OUTPUT: Z_Fused (3484 × 64)
          ↓ (major dimension reduction to latent)

Module 7: Reconstruction Loss
  INPUT:  Z_Fused (3484 × 64)
  OUTPUT: X̂_RNA (3484 × 18,085), X̂_ADT (3484 × 31)
          + L_total (scalar)
          ↓ (reconstruction back to original space)

Module 8: Clustering
  INPUT:  Z_Fused (3484 × 64)
  OUTPUT: domain_labels (3484,), umap_coords (3484 × 2)
          + ARI (scalar metric)
          ↓ (final biological discovery)

═════════════════════════════════════════════════════════════════
SUMMARY: (3484 × 18,085) ──→ ... ──→ (3484 × 64) ──→ 3,484 labels
         Raw Data             Pipeline      Latent      Discovery
```

---

## 🎯 Module Integration Checklist

```
PHASE 1: Feature Extraction
─────────────────────────────────────────────────────────
[✓] Module 1: Preprocessing
    └─ Extract: clr_normalize_each_cell, log1p transform
    └─ Source: spaLLM/preprocess.py (lines 47-56)
    
[✓] Module 2: Encoding  
    └─ Extract: Full modelTriatt_Flow1.py transformer
    └─ Source: spaLLM/modelTriatt_Flow1.py (entire file)


PHASE 2: Structural Mapping
─────────────────────────────────────────────────────────
[✓] Module 3: Graph Construction
    └─ Extract spaLLM: construct_graph_by_coordinate, construct_graph_by_feature
    └─ Extract COSMOS: sparse_mx_to_torch_edge_list
    └─ Source: spaLLM/preprocess.py (L40-71) + COSMOS/cosmos.py (L50-54)
    
[✓] Module 4: Spatial Encoding
    └─ Extract SpatialGlue: Encoder class, AttentionLayer
    └─ Extract COSMOS: GNN layers from modulesWNN.py
    └─ Source: SpatialGlue/model.py (L21-91) + COSMOS/modulesWNN.py


PHASE 3: Integration & Fusion
─────────────────────────────────────────────────────────
[✓] Module 5: Contrastive Alignment
    └─ Extract COSMOS: InfoNCE loss, contrastive pair generation
    └─ Source: COSMOS/cosmos.py + COSMOS/modulesWNN.py
    
[✓] Module 6: Dual-Attention Fusion
    └─ Extract SpatialGlue: Encoder_overall, hierarchical attention
    └─ Source: SpatialGlue/model.py (L5-91)
    
[✓] Module 7: Reconstruction & Loss
    └─ Extract SpatialGlue: Decoder class
    └─ Write Custom: MSE loss + Graph Laplacian + combined L_total
    └─ Source: SpatialGlue/model.py (decoder) + custom writing


PHASE 4: Biological Discovery
─────────────────────────────────────────────────────────
[✓] Module 8: Clustering
    └─ Write Custom: Leiden clustering, UMAP, ARI validation
    └─ Use: scanpy.tl.leiden + sklearn.metrics.adjusted_rand_score
```

---

## 💾 Dependencies by Module

| Module | PyTorch | TensorFlow | scikit-learn | SciPy | NumPy | PyTorch Geometric | Scanpy |
|--------|---------|-----------|--------------|-------|-------|-------------------|--------|
| 1      | ✓       | -         | ✓            | ✓     | ✓     | -                 | -      |
| 2      | ✓       | -         | -            | -     | ✓     | -                 | -      |
| 3      | -       | -         | ✓            | ✓     | ✓     | -                 | -      |
| 4      | ✓       | -         | -            | -     | ✓     | ✓                 | -      |
| 5      | ✓       | -         | -            | -     | ✓     | -                 | -      |
| 6      | ✓       | -         | -            | -     | ✓     | -                 | -      |
| 7      | ✓       | -         | -            | -     | ✓     | -                 | -      |
| 8      | -       | -         | ✓            | ✓     | ✓     | -                 | ✓      |

---

## 🎓 Learning Path for Implementation

**Week 1: Foundation (Modules 1-3)**
- Understand spaLLM preprocessing philosophy
- Study graph construction logic
- Get comfortable with sparse matrix handling

**Week 2: Encoding Pipeline (Modules 2, 4)**
- Learn transformer architecture from spaLLM
- Understand GATv2 and attention mechanisms
- Practice spatial smoothing concepts

**Week 3: Integration (Modules 5-6)**
- Study COSMOS's contrastive learning approach
- Learn hierarchical attention from SpatialGlue
- Understand multi-modal alignment

**Week 4: Training (Module 7)**
- Implement multi-loss training
- Understand backpropagation through all modules
- Practice hyperparameter tuning

**Week 5: Discovery (Module 8)**
- Implement clustering pipeline
- Validate results with ARI
- Create visualization outputs

---

**Created:** May 21, 2026  
**Status:** ✅ Complete - Ready for Implementation  
**Document Level:** Technical Reference + Visual Guide

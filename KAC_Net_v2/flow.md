# Figure: Complete Workflow of the Proposed KAC-Net Framework

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: FEATURE EXTRACTION & ENRICHMENT             
└──────────────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────── INPUT DATA ────────────────────────────┐
        │                                                                    │
        │  Raw RNA Counts : X_RNA ∈ R^(3484 × 18085)                         │
        │  Raw ADT Counts : X_ADT ∈ R^(3484 × 31)                            │
        │                                                                   │
        └──────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: MULTIMODAL PREPROCESSING                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ RNA Pipeline:                                                                │
│   • Library-size Normalization                                               │
│   • Log1p Transformation                                                     │
│                                                                              │
│ ADT Pipeline:                                                                │
│   • CLR (Centered Log Ratio) Normalization                                   │
│                                                                              │
│ Purpose:                                                                     │
│   Stabilize variance and align feature scales                                │
│                                                                              │
│ Output:                                                                      │
│   X̃_RNA , X̃_ADT                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: KNOWLEDGE-ENRICHED ENCODING (spaLLM Logic)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ Input: X̃_RNA                                                               │
│                                                                              │
│ Transformer-based Biological Encoder:                                       │
│   • scGPT                                                                    │
│   • Geneformer                                                               │
│                                                                              │
│ Logic:                                                                      │
│   Uses biological prior knowledge from millions of cells                    │
│   to recover signals lost due to technical noise/dropouts                   │
│                                                                              │
│ Output:                                                                     │
│   H_RNA ∈ R^(3484 × 512)                                                    │
└──────────────────────────────────────────────────────────────────────────────┘


                                      │
                                      ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 2: STRUCTURAL MAPPING                        │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: MULTI-GRAPH CONSTRUCTION                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs:                                                                     │
│   • Spatial Coordinates (x, y)                                              │
│   • H_RNA                                                                   │
│   • X̃_ADT                                                                  │
│                                                                              │
│ Spatial Graph (G_s):                                                        │
│   • KNN (k = 6) using Euclidean Distance                                    │
│                                                                              │
│ Feature Graph (G_f):                                                        │
│   • Concatenate RNA + ADT Features                                          │
│   • KNN using Cosine Similarity                                             │
│                                                                              │
│ Outputs:                                                                    │
│   A_s → Physical Roadmap                                                    │
│   A_f → Biological Similarity Roadmap                                       │
│                                                                              │
│ Purpose:                                                                    │
│   Identify physical and molecular neighbors                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: LOCAL SPATIAL ENCODING ("Talking Phase")                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs:                                                                     │
│   • H_RNA                                                                   │
│   • X̃_ADT                                                                  │
│   • A_s , A_f                                                               │
│                                                                              │
│ Algorithm:                                                                  │
│   Residual GATv2 (Graph Attention Network)                                  │
│                                                                              │
│ Mechanism:                                                                  │
│   • Neighbor communication                                                  │
│   • Trustworthy neighbors weighted higher                                   │
│   • Residual skip-connections prevent over-smoothing                        │
│                                                                              │
│ Outputs:                                                                    │
│   Z_RNA , Z_ADT                                                             │
└──────────────────────────────────────────────────────────────────────────────┘


                                      │
                                      ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: INTEGRATION & OPTIMIZATION                     │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 5: CROSS-MODAL CONTRASTIVE ALIGNMENT (COSMOS Logic)                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs:                                                                     │
│   • Z_RNA                                                                   │
│   • Z_ADT                                                                   │
│                                                                              │
│ Algorithm:                                                                  │
│   • InfoNCE Loss                                                            │
│   • Contrastive Learning                                                    │
│                                                                              │
│ Logic:                                                                      │
│   Pull same-spot RNA & Protein embeddings closer                            │
│   Push different spots apart                                                │
│                                                                              │
│ Loss Generated:                                                             │
│   L_cl → Contrastive Loss                                                   │
│                                                                              │
│ Output:                                                                     │
│   Shared Aligned Latent Space                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 6: ADAPTIVE DUAL-ATTENTION FUSION (SpatialGlue Logic)                │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs:                                                                     │
│   • Aligned Z_RNA                                                           │
│   • Aligned Z_ADT
│   • A_s                                                                     │
│   • A_f                                                                     │

│                                                                              │
│ Within-Modality Attention:                                                  │
│   • Learns importance of A_s vs A_f                                         │
│                                                                              │
│ Between-Modality Attention:                                                 │
│   • Learns spot-specific weight ω                                           │
│   • Balances RNA vs Protein contribution                                    │
│                                                                              │
│ Output:                                                                     │
│   Z_Fused ∈ R^(3484 × 64)                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 7: RECONSTRUCTION & REGULARIZATION ("Decoder")                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs:                                                                     │
│   • Z_Fused                                                                 │
│   • A_s                                                                     │
│                                                                              │
│ Decoder:                                                                    │
│   • Parallel MLPs reconstruct RNA & ADT counts                              │
│                                                                              │
│ Regularizer:                                                                │
│   • Graph Laplacian Spatial Smoothness                                      │
│                                                                              │
│ Losses Generated:                                                           │
│   • L_recon → Reconstruction Loss (MSE)                                     │
│   • L_spat  → Spatial Regularization Loss                                   │
│                                                                              │
│ Total Optimization:                                                         │
│   L_total = L_cl + L_recon + L_spat                                         │
│                                                                              │
│ Training:                                                                   │
│   Backpropagation across entire network                                     │
└──────────────────────────────────────────────────────────────────────────────┘


                                      │
                                      ▼


┌──────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 4: BIOLOGICAL DISCOVERY                       │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ MODULE 8: SPATIAL DOMAIN IDENTIFICATION                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Input:                                                                      │
│   Z_Fused                                                                   │
│                                                                              │
│ Algorithms:                                                                 │
│   • Leiden Clustering                                                       │
│   • Louvain Clustering                                                      │
│   • UMAP Visualization                                                      │
│                                                                              │
│ Final Outputs:                                                              │
│   • Anatomical Domain Labels                                                │
│       - Follicle                                                            │
│       - Cortex                                                              │
│       - Medulla Cords                                                       │
│                                                                              │
│ Validation:                                                                 │
│   • ARI Score vs Manual Ground Truth                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```
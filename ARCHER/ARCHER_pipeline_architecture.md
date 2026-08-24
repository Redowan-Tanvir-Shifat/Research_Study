# ARCHER Pipeline Architecture & System Specification
**Adaptive RNA-Anchored Consensus Graph Fusion Framework for Spatial Multi-Omics**

---

## Executive Overview
This document provides a rigorous, step-by-step architectural breakdown of the **ARCHER** framework (`Archer_v1.ipynb`). Each step explicitly details its **Input**, **Processing / Method**, and **Output**.

---

## Step 1: Raw Data Loading & Ground Truth Annotation

* **Input**:
  * `rna_path` (`str`): File path to `adata_RNA.h5ad` (RNA expression matrix & spatial coordinates).
  * `other_path` (`str`): File path to `adata_ADT.h5ad` (10x CITE-seq ADT) or `adata_ATAC.h5ad` (Spatial ATAC).
  * `annotation_path` (`str`): File path to ground truth CSV metadata (`annotation.csv` or `anno.csv`).
  * `gt_column` (`str`): Ground truth column key (`manual-anno` for 10x, `cluster` for ATAC).
* **Processing**:
  1. Load single-cell spatial AnnData objects using `scanpy.read_h5ad()`.
  2. Execute `.var_names_make_unique()` on both AnnData instances.
  3. Load ground truth metadata table via `pandas.read_csv()` and assign ground truth labels to `adata_omics1.obs['ground_truth']` and `adata_omics2.obs['ground_truth']`.
  4. Determine unique class count $K = N_{\text{ground\_truth}}$.
* **Output**:
  * `adata_omics1`: Raw RNA AnnData object containing spatial coordinates $(x, y)$ in `.obsm['spatial']`.
  * `adata_omics2`: Raw ADT/ATAC AnnData object.
  * `n_ground_truth` (`int`): Count of ground-truth clusters.

---

## Step 2: Modality Preprocessing & Feature Extraction

* **Input**:
  * Raw `adata_omics1` and `adata_omics2` AnnData objects.
  * `data_type` (`str`): Modality indicator (`'10x'` or `'Spatial-epigenome-transcriptome'`).
* **Processing**:
  1. **RNA Modality Preprocessing ($X_1$)**:
     * Filter genes present in $<10$ cells via `sc.pp.filter_genes()`.
     * Identify top 3,000 Highly Variable Genes (HVGs) using Seurat v3 flavor (`sc.pp.highly_variable_genes`).
     * Perform total count normalization ($\text{target\_sum} = 10^4$), $\log(1+x)$ transformation, and Z-score scaling (`sc.pp.scale`).
     * Compute Principal Component Analysis (PCA) to extract $X_1$ features ($D_1 = N_{\text{vars, ADT}} - 1$ for 10x, or $D_1 = 50$ for ATAC).
  2. **Proteomic / Epigenomic Preprocessing ($X_2$)**:
     * **10x CITE-seq (ADT)**: Apply Centered Log-Ratio (CLR) normalization per cell, scale features, and compute PCA to obtain $X_2 \in \mathbb{R}^{N \times D_2}$.
     * **Spatial Epigenome (ATAC)**: Align cell barcodes, apply Term Frequency-Inverse Document Frequency (TF-IDF), L1 normalization, $\log(1+x)$ transformation, and randomized Singular Value Decomposition (LSI/SVD, 50 components) to obtain $X_2 \in \mathbb{R}^{N \times 50}$.
* **Output**:
  * `adata_omics1.obsm['feat']`: Matrix $X_1 \in \mathbb{R}^{N \times D_1}$ (RNA feature representations).
  * `adata_omics2.obsm['feat']`: Matrix $X_2 \in \mathbb{R}^{N \times D_2}$ (ADT/ATAC feature representations).

---

## Step 3: Hybrid Topology Construction Engine

* **Input**:
  * Physical spot coordinates: `adata_omics1.obsm['spatial']` $\in \mathbb{R}^{N \times 2}$.
  * Feature matrices: $X_1 \in \mathbb{R}^{N \times D_1}$ and $X_2 \in \mathbb{R}^{N \times D_2}$.
  * `n_neighbors_spatial` (`int`): 6 for 10x/Epigenome datasets, 3 for SPOTS datasets.
  * `n_neighbors_feature` (`int`): 20.
* **Processing**:
  1. **Physical Proximity Graph ($A_s$)**: Compute Spatial $K$-NN graph on physical spot coordinates $(x,y)$, symmetrize $A_s = \max(A_s, A_s^T)$, and zero-out diagonal elements.
  2. **Feature KNN Graphs ($A_{f1}, A_{f2}$)**: Compute correlation-metric $K$-NN graphs ($K=20$) on $X_1$ and $X_2$, symmetrize, and zero-out diagonals.
  3. **ARISE Shared Edge Graph ($A_{\text{com}}$)**: Compute RNA-anchored graph intersection between feature graph $A_{f1}$ and physical spatial graph $A_s$:
     $$A_{\text{com}} = (A_{f1} > 0) \odot (A_s > 0)$$
  4. **SpaFusion Refined Spatial Graphs ($A_{s1}, A_{s2}$)**:
     $$A_{s1} = (A_{f1} > 0) \odot (A_s > 0), \quad A_{s2} = (A_{f2} > 0) \odot (A_s > 0)$$
  5. **SpaFusion 3-Node Motif Graphs ($A_{m1}, A_{m2}$)**: Compute normalized 3-node triangle motif co-occurrence:
     $$M_3^v = \frac{(A_{fv}^2 \odot A_{fv})}{\max(A_{fv}^2 \odot A_{fv})}, \quad A_{mv} = 0.5 \cdot A_{fv} + 0.5 \cdot M_3^v$$
  6. **Symmetric GCN Graph Normalization**: Apply degree normalization to all adjacency matrices:
     $$\tilde{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}$$
* **Output**: Dictionary of PyTorch tensors (`graphs`):
  * `adj_spatial_raw`: Dense float tensor $A_s \in \mathbb{R}^{N \times N}$.
  * `adj_com_norm`: Normalized sparse PyTorch tensor $\tilde{A}_{\text{com}} \in \mathbb{R}^{N \times N}$.
  * `adj_s1_norm`, `adj_s2_norm`: Normalized sparse PyTorch tensors $\tilde{A}_{s1}, \tilde{A}_{s2} \in \mathbb{R}^{N \times N}$.
  * `adj_m1_norm`, `adj_m2_norm`: Normalized sparse PyTorch tensors $\tilde{A}_{m1}, \tilde{A}_{m2} \in \mathbb{R}^{N \times N}$.

---

## Step 4: SMART MNN Triplet Mining Engine

* **Input**:
  * RNA feature matrix $X_1 \in \mathbb{R}^{N \times D_1}$.
  * `top_k` (`int`): 3.
  * `neg_ratio` (`float`): 0.6.
* **Processing**:
  1. **Mutual Nearest Neighbor (MNN) Pair Mining**: Identify reciprocal nearest neighbors in $X_1$. If cell $i$ has neighbor $j$ in top $K$, and cell $j$ has neighbor $i$ in top $K$, add $(i, j)$ as an anchor-positive pair. (Fallback to 1-NN if MNN set is empty).
  2. **Semi-Hard Negative Mining**: Compute pairwise Euclidean distance matrix $D \in \mathbb{R}^{N \times N}$. For each anchor $i$, select a negative cell $n$ from the far quantile threshold index ($N \times \text{neg\_ratio}$).
* **Output**:
  * `triplets`: Tuple of 1D LongTensors `(anchors, positives, negatives)` representing cell index triplets.

---

## Step 5: Multi-Branch GNN & Transformer Encoders

* **Input**:
  * Feature matrices $X_1 \in \mathbb{R}^{N \times D_1}$ and $X_2 \in \mathbb{R}^{N \times D_2}$.
  * Normalized graph dictionary `graphs`.
* **Processing**:
  * **Modality 1 Encoders (RNA)**:
    1. Motif GNN: $z_{m1} = \text{ReLU}(\tilde{A}_{m1} X_1 W_{m1}) \in \mathbb{R}^{N \times 64}$.
    2. Spatial GNN: $z_{s1} = \text{ReLU}(\tilde{A}_{s1} X_1 W_{s1}) \in \mathbb{R}^{N \times 64}$.
    3. Global Context Transformer: Linear projection $X_1 W_{g1} \to \text{TransformerEncoder} \to z_{g1} \in \mathbb{R}^{N \times 64}$.
  * **Modality 2 Encoders (ADT / ATAC)**:
    1. Motif GNN: $z_{m2} = \text{ReLU}(\tilde{A}_{m2} X_2 W_{m2}) \in \mathbb{R}^{N \times 64}$.
    2. Spatial GNN: $z_{s2} = \text{ReLU}(\tilde{A}_{\text{com}} X_2 W_{s2}) \in \mathbb{R}^{N \times 64}$ (Anchored on $A_{\text{com}}$ for spatial stability).
    3. Global Context Transformer: Linear projection $X_2 W_{g2} \to \text{TransformerEncoder} \to z_{g2} \in \mathbb{R}^{N \times 64}$.
* **Output**:
  * Sub-branch embeddings ($z_{m1}, z_{s1}, z_{g1}$) and ($z_{m2}, z_{s2}, z_{g2}$) of shape $\mathbb{R}^{N \times 64}$.

---

## Step 6: Intra-Omic Self-Correlation Fusion

* **Input**:
  * Sub-branch embeddings ($z_m, z_s, z_g \in \mathbb{R}^{N \times 64}$).
  * Normalized motif graph $\tilde{A}_m$.
* **Processing**:
  1. **Learnable Linear Softmax Combination**:
     $$\mathbf{w} = \text{Softmax}([w_1, w_2, w_3]), \quad z_{\text{linear}} = w_1 z_m + w_2 z_s + w_3 z_g$$
  2. **Graph Structure Propagation**: $\mathcal{L}_v = \text{spmm}(\tilde{A}_m, z_{\text{linear}})$.
  3. **Self-Correlation Matrix Attention**:
     $$S = \text{Softmax}\left(\frac{\mathcal{L}_v \mathcal{L}_v^T}{\sqrt{64}}\right), \quad H_v = S \cdot \mathcal{L}_v$$
  4. **Residual Addition & Refinement MLP**:
     $$\tilde{Z}_v = \mathcal{L}_v + \alpha H_v, \quad z_v = \text{MLP}_v(\tilde{Z}_v) \in \mathbb{R}^{N \times 64}$$
* **Output**:
  * `z1`: Modality 1 refined embedding tensor $z_1 \in \mathbb{R}^{N \times 64}$.
  * `z2`: Modality 2 refined embedding tensor $z_2 \in \mathbb{R}^{N \times 64}$.

---

## Step 7: Variance-Adaptive Inter-Omic Aggregation

* **Input**:
  * Modality 1 embedding $z_1 \in \mathbb{R}^{N \times 64}$.
  * Modality 2 embedding $z_2 \in \mathbb{R}^{N \times 64}$.
* **Processing**:
  1. Compute scalar embedding feature variances $\text{Var}(z_1)$ and $\text{Var}(z_2)$.
  2. Compute variance balance weights:
     $$\beta_1 = \frac{\text{Var}(z_1)}{\text{Var}(z_1) + \text{Var}(z_2) + 10^{-12}}, \quad \beta_2 = 1.0 - \beta_1$$
  3. Compute variance-weighted joint aggregation:
     $$z_{\text{fused}} = \beta_1 z_1 + \beta_2 z_2$$
* **Output**:
  * `z_fused`: Joint inter-omic embedding tensor $z_{\text{fused}} \in \mathbb{R}^{N \times 64}$.

---

## Step 8: Multi-Scale Decoders & Reconstructions

* **Input**:
  * Bottleneck embeddings $z_1, z_2, z_{\text{fused}} \in \mathbb{R}^{N \times 64}$.
  * Normalized spatial adjacencies $\tilde{A}_{s1}, \tilde{A}_{s2}$.
* **Processing**:
  1. **Modality GNN Decoders**:
     $$\hat{X}_1 = \tilde{A}_{s1} z_1 W_{\text{dec1}}, \quad \hat{X}_2 = \tilde{A}_{s2} z_2 W_{\text{dec2}}$$
  2. **Joint Linear Decoders**:
     $$\hat{X}_1^{\text{joint}} = z_{\text{fused}} W_{\text{joint1}}, \quad \hat{X}_2^{\text{joint}} = z_{\text{fused}} W_{\text{joint2}}$$
* **Output**:
  * Reconstructed feature matrices `rec_x1`, `rec_x2`, `rec_joint_x1`, `rec_joint_x2`.

---

## Step 9: Stage 1 Warmup Pre-training Optimization (5,000 Epochs)

* **Input**:
  * Feature matrices $X_1, X_2$.
  * Decoder outputs ($\hat{X}_1, \hat{X}_2, \hat{X}_1^{\text{joint}}, \hat{X}_2^{\text{joint}}$).
  * Raw physical spatial graph $A_s \in \mathbb{R}^{N \times N}$.
  * MNN triplets $(a, p, n)$.
  * Loss weight schedule: `recon: 1.0, triplet: 0.5, spatial: 5.0, aux: 0.1`.
* **Processing**:
  Execute 5,000 warmup epochs using Adam optimizer ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$):
  1. **Multi-Scale Reconstruction MSE Loss**:
     $$\mathcal{L}_{\text{recon}} = \text{MSE}(X_1, \hat{X}_1) + 5.0 \cdot \text{MSE}(X_2, \hat{X}_2) + \text{MSE}(X_1, \hat{X}_1^{\text{joint}}) + 5.0 \cdot \text{MSE}(X_2, \hat{X}_2^{\text{joint}})$$
  2. **SMART MNN Triplet Margin Loss**:
     $$\mathcal{L}_{\text{triplet}} = \frac{1}{|T|} \sum_{(a,p,n)} \max\left(0, \|z_a - z_p\|^2 - \|z_a - z_n\|^2 + 0.5\right)$$
  3. **ARISE Spatial Coherence BCE Loss**:
     $$\mathcal{L}_{\text{spatial}} = -\frac{1}{N^2} \sum_{i,j} \left[ A_{s,ij} \log \sigma(C_{ij}) + (1 - A_{s,ij}) \log(1 - \sigma(C_{ij})) \right]$$
     where $C_{ij} = \frac{z_i \cdot z_j}{\|z_i\| \|z_j\|}$.
  4. **Auxiliary Alignment Loss**: $\mathcal{L}_{\text{aux}} = \text{MSE}(z_{\text{fused}}, z_1) + \text{MSE}(z_{\text{fused}}, z_2)$.
  5. **Total Stage 1 Loss**:
     $$\mathcal{L}_{\text{Stage1}} = 1.0 \cdot \mathcal{L}_{\text{recon}} + 0.5 \cdot \mathcal{L}_{\text{triplet}} + 5.0 \cdot \mathcal{L}_{\text{spatial}} + 0.1 \cdot \mathcal{L}_{\text{aux}}$$
* **Output**:
  * Warmup pre-trained model weights.
  * Warmup joint embedding $z_{\text{warmup}} \in \mathbb{R}^{N \times 64}$.

---

## Step 10: Cluster Centroid Initialization

* **Input**:
  * Warmup joint embedding $z_{\text{warmup}} \in \mathbb{R}^{N \times 64}$.
  * Ground truth cluster count $K = N_{\text{clusters}}$.
* **Processing**:
  * Execute K-Means clustering (`sklearn.cluster.KMeans`, $\text{n\_init}=20$) on $z_{\text{warmup}}$.
  * Copy K-Means cluster center matrix into trainable parameter tensor `model.cluster_layer.data`.
* **Output**:
  * Initialized cluster centroid parameters $\mathbf{\mu} \in \mathbb{R}^{K \times 64}$.

---

## Step 11: Stage 2 Consensus Self-Training Fine-Tuning (2,500 Epochs)

* **Input**:
  * Pre-trained model parameters and cluster centroids $\mathbf{\mu} \in \mathbb{R}^{K \times 64}$.
  * Loss weight schedule: `recon: 1.0, triplet: 0.5, spatial: 10.0, consensus: 1.0, aux: 0.1`.
* **Processing**:
  Execute 2,500 fine-tuning epochs:
  1. **Student's t-Distribution Soft Assignment**:
     $$q_{ij} = \frac{\left(1 + \|z_i - \mu_j\|^2\right)^{-1}}{\sum_{j'} \left(1 + \|z_i - \mu_{j'}\|^2\right)^{-1}}$$
     Compute soft assignment matrices $q_{\text{joint}}, q_1, q_2$ for $z_{\text{fused}}, z_1, z_2$.
  2. **Sharpened Target Distribution ($P$)**:
     $$p_{ij} = \frac{q_{ij}^2 / \sum_i q_{ij}}{\sum_{j'} \left(q_{ij'}^2 / \sum_i q_{ij'}\right)}$$
  3. **Consensus KL Divergence Loss**:
     $$\bar{Q} = \frac{q_{\text{joint}} + q_1 + q_2}{3}, \quad \mathcal{L}_{\text{consensus}} = D_{\text{KL}}(\bar{Q} \parallel P) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{\bar{Q}_{ij}}$$
  4. **Total Stage 2 Loss**:
     $$\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{Stage1}} + 1.0 \cdot \mathcal{L}_{\text{consensus}}$$
* **Output**:
  * Fine-tuned ARCHER model weights.
  * Inference fused embedding $z_{\text{fused}} \in \mathbb{R}^{N \times 64}$.
  * Soft assignment probabilities $q_{\text{final}} \in \mathbb{R}^{N \times K}$ and initial cluster predictions $y_{\text{pred\_consensus}} = \arg\max_j (q_{\text{final}})$.

---

## Step 12: Embedding Normalization & Dimensionality Reduction

* **Input**:
  * Inference joint embedding $z_{\text{fused}} \in \mathbb{R}^{N \times 64}$.
* **Processing**:
  1. Compute L2 normalized embedding: $z_{\text{norm}} = \frac{z_{\text{fused}}}{\|z_{\text{fused}}\|_2}$.
  2. Perform Principal Component Analysis (PCA) to extract 20 principal components.
* **Output**:
  * `adata.obsm['ARCHER_emb']`: L2-normalized embedding matrix $z_{\text{norm}} \in \mathbb{R}^{N \times 64}$.
  * `adata.obsm['ARCHER_emb_pca']`: PCA-reduced matrix $\in \mathbb{R}^{N \times 20}$.

---

## Step 13: Downstream Clustering & Metric Evaluation Benchmark

* **Input**:
  * `adata.obsm['ARCHER_emb_pca']` $\in \mathbb{R}^{N \times 20}$.
  * Ground truth cluster array $y_{\text{true}} \in \mathbb{R}^N$.
  * `n_clusters` (`int`).
* **Processing**:
  1. **R `mclust` Gaussian Mixture Model**: Call R function `Mclust(df, G=n_clusters, modelNames="EEE")` via `rpy2` to obtain spatial domain predictions $y_{\text{pred}}$.
  2. **Metrics Calculation**:
     * **Adjusted Rand Index (ARI)**: Metric comparing partition similarity.
     * **Normalized Mutual Information (NMI)**: Shared entropy-normalized mutual information.
     * **Adjusted Mutual Information (AMI)**: Adjusted mutual information score.
     * **Homogeneity & V-Measure**: Clustering purity metrics.
     * **Silhouette Score**: Unsupervised cluster compactness score computed on $z_{\text{fused}}$.
  3. Export result summary metrics to CSV (`ARCHER_all_results.csv`).
* **Output**:
  * `adata.obs['ARCHER']`: Final predicted spatial cluster labels.
  * CSV result files (`results/ARCHER_{dataset_name}_results.csv` and `results/ARCHER_all_results.csv`).

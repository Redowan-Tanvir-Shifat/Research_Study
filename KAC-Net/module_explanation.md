# Phase 1: Feature Extraction & Enrichment

## Module 1: Multimodal Preprocessing

### 1. The Core Architectural Problem Solved
Raw spatial multi-omics sequencing output displays stark dynamic range imbalances and technical artifacts. Intracellular mRNA captures are inherently subject to sequencing depth variations, while Antibody-Derived Tags (ADTs) suffer from non-specific background binding and amplification biases. Standard raw values cannot be used directly because the structural optimization layers would focus exclusively on dominant, highly-expressed molecules. Module 1 conditions both signal tracks to bring them into a stable numerical range without sacrificing molecular sensitivity.

### 2. The Step-by-Step Mechanism & Internal Flow
Transcriptomic Stream (RNA): For each coordinate spot, the cell library profile is scaled by total sequencing depth across all 18,085 target transcripts, forcing a uniform scale. Following library scaling, data elements undergo a logarithmic transform to counter high-expression skew.

Proteomic Stream (ADT): Surface protein markers (31 target antibodies) are processed utilizing a compositionally-aware logarithmic adjustment relative to the geometric mean of the protein panel for that specific micro-environment spot.

### 3. Algorithms & Deep Mathematical Logic
Library-Size Scaling: Normalizes capture variation across spots.

Log1p Shift: Evaluated as:
$$\tilde{X}_{RNA} = \ln(X_{scaled} + 1)$$
This transformation compresses large values while stabilizing variance across low-expression ranges, allowing minor cell-state markers to remain influential.

Centered Log Ratio (CLR): Computed independently for each spot spot $i$ across $M$ protein channels:
$$\tilde{X}_{ADT,i} = \left[ \ln\frac{x_{i,1}}{g(x_i)}, \ln\frac{x_{i,2}}{g(x_i)}, \dots, \ln\frac{x_{i,M}}{g(x_i)} \right]$$
Where $g(x_i) = \left(\prod_{m=1}^M x_{i,m}\right)^{1/M}$ represents the geometric mean of the raw count vectors. This normalizes for antibody background binding and amplification disparities.

### 4. Inputs & Dimensionalities
Raw Inputs: * $X_{RNA} \in \mathbb{R}^{3484 \times 18085}$ (Raw transcript reads across spots and genes)
$X_{ADT} \in \mathbb{R}^{3484 \times 31}$ (Raw target antibody count strings)

### 5. Outputs & Dimensionalities
Conditioned Matrices:
$\tilde{X}_{RNA} \in \mathbb{R}^{3484 \times 18085}$ (Variance-stabilized log gene counts)
$\tilde{X}_{ADT} \in \mathbb{R}^{3484 \times 31}$ (CLR-normalized protein markers)

## Module 2: Knowledge-Enriched Encoding (spaLLM Engine)

### 1. The Core Architectural Problem Solved
Spatial transcriptomic captures typically map 10–30 cells per capture spot, introducing severe sparsity issues and technical dropout (where true biological transcripts are unrecorded due to detection limits). Traditional pipelines use arbitrary hard-threshold filtering or Highly Variable Gene (HVG) reductions, which discard subtle biological signals. Module 2 addresses this by utilizing a pre-trained single-cell foundation model to infer missing transcriptomic context based on co-expression rules learned from millions of cell profiles.

### 2. The Step-by-Step Mechanism & Internal Flow
The full 18,085 normalized gene count array $\tilde{X}_{RNA}$ is passed directly to the foundation model without subsetting.
The model maps individual gene identities and values to token embeddings.
Multi-head self-attention mechanisms evaluate gene-to-gene interactions, reconstructing missing cellular features by predicting expected expression patterns.
The high-dimensional feature space is projected into a dense, non-sparse biological embedding.

### 3. Algorithms & Deep Mathematical Logic
Foundation Transformer Model (scGPT / Geneformer): Utilizes stacked Multi-Head Self-Attention layers:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where Queries ($Q$), Keys ($K$), and Values ($V$) represent linear transformations of the input transcript tokens. The self-attention matrix captures non-linear gene regulatory networks, using learned co-expression priors to infer missing data values.

### 4. Inputs & Dimensionalities
Input Tensor: $\tilde{X}_{RNA} \in \mathbb{R}^{3484 \times 18085}$

### 5. Outputs & Dimensionalities
Enriched Embedding Matrix: $H_{RNA} \in \mathbb{R}^{3484 \times 512}$ (Dense, biologically-smoothed transcriptomic representation)

# Phase 2: Structural & Geometric Mapping

## Module 3: Multi-Graph Construction

### 1. The Core Architectural Problem Solved
Tissue micro-environments are defined by both physical constraints and functional, long-range cellular similarities. Relying solely on physical coordinates can cause over-smoothing across structural boundaries (e.g., blending follicle edges into the cortex). Conversely, relying only on molecular features can create a fragmented "salt-and-pepper" pattern due to local noise. Module 3 solves this by constructing two separate graphs: one for physical contact networks and another for molecular profile similarities.

### 2. The Step-by-Step Mechanism & Internal Flow
Spatial Graph ($G_s$) Pipeline: Evaluates the Euclidean distance between spot centroids based on slide coordinate registers ($x, y$), connecting each spot to its nearest neighbors.
Feature Graph ($G_f$) Pipeline: Concatenates the enriched transcriptomic vectors $H_{RNA}$ and normalized proteomic arrays $\tilde{X}_{ADT}$ into a single joint feature profile ($512 + 31 = 543$ dimensions) for each spot. It then computes pairwise cosine distances to connect molecularly similar spots across the tissue.

### 3. Algorithms & Deep Mathematical Logic
Spatial K-Nearest Neighbors Lattice ($k=6$): Matches the hexagonal grid layout of 10x Visium arrays, restricting connections to the immediate physical neighborhood.
Feature K-Nearest Neighbors Network: Connects spots based on Cosine Similarity:
$$\text{Sim}(i, j) = \frac{z_i \cdot z_j}{\|z_i\| \|z_j\|}$$
This captures long-range functional similarities between identical, non-contiguous cell types across the tissue section.
Adjacency Formulations: Connections are stored as binary indicators:
$$A_{ij} = \begin{cases} 1, & \text{if } j \in \text{KNN}(i) \\ 0, & \text{otherwise} \end{cases}$$

### 4. Inputs & Dimensionalities
Coordinate Register: Layout arrays $C_{spatial} \in \mathbb{R}^{3484 \times 2}$ (containing $x, y$ coordinates)
Feature Elements: $H_{RNA} \in \mathbb{R}^{3484 \times 512}$ and $\tilde{X}_{ADT} \in \mathbb{R}^{3484 \times 31}$

### 5. Outputs & Dimensionalities
Structural Graphs:
$A_s \in \mathbb{R}^{3484 \times 3484}$ (Physical Adjacency Matrix, sparse layout)
$A_f \in \mathbb{R}^{3484 \times 3484}$ (Functional Similarity Adjacency Matrix)

## Module 4: Local Spatial Encoding (Residual GATv2 Encoders)

### 1. The Core Architectural Problem Solved
Individual spatial profiles are highly susceptible to localized capture dropouts. To resolve this, features must be smoothed using neighborhood context. However, standard graph convolutional networks use uniform averaging, which can smooth away sharp structural boundaries and unique cell types. Module 4 prevents this by using an attention-guided message passing scheme that dynamically weights neighbors, allowing the model to adaptively filter out irrelevant or noisy signals.

### 2. The Step-by-Step Mechanism & Internal Flow
Parallel Graph Attention pipelines are established for the RNA and ADT data streams.
For each modality, spots inspect their spatial ($A_s$) and functional ($A_f$) neighbors.
Every neighbor's features are evaluated to compute a dynamic attention weight, determining its influence.
Neighbor features are aggregated based on these learned weights.
Residual skip connections add the original input back to the aggregated features, preventing over-smoothing and maintaining local identity.

### 3. Algorithms & Deep Mathematical Logic
GATv2 Architecture: Dynamically computes attention weights $\alpha_{ij}$ between spot $i$ and neighbor $j$:
$$\alpha_{ij} = \frac{\exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}[z_i \,\|\, z_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}[z_i \,\|\, z_k]\right)\right)}$$
Where $\mathbf{W}$ is a shared linear projection matrix, $\mathbf{a}$ is the attention vector, and $\|$ denotes concatenation. GATv2 introduces dynamic attention scaling by applying the attention projection after the non-linear activation function, preventing attention collapse.
Residual Mapping Layer: Re-introduces the original feature vector to prevent information loss:
$$Z_{out} = \text{LayerNorm}\left( \text{GATv2}(Z_{in}, A) + \mathbf{W}_{res}Z_{in} \right)$$

### 4. Inputs & Dimensionalities
Feature Matrices: $H_{RNA} \in \mathbb{R}^{3484 \times 512}$ and $\tilde{X}_{ADT} \in \mathbb{R}^{3484 \times 31}$
Graph Lattices: $A_s \in \mathbb{R}^{3484 \times 3484}$ and $A_f \in \mathbb{R}^{3484 \times 3484}$

### 5. Outputs & Dimensionalities
Spatially Embedded Arrays:
$Z_{RNA} \in \mathbb{R}^{3484 \times d}$ (Spatially smoothed transcriptomic embedding)
$Z_{ADT} \in \mathbb{R}^{3484 \times d}$ (Spatially smoothed proteomic embedding)

# Phase 3: Integration, Fusion & Regularization

## Module 5: Cross-Modal Contrastive Alignment (COSMOS Logic)

### 1. The Core Architectural Problem Solved
Even after spatial encoding, the transcriptomic ($Z_{RNA}$) and proteomic ($Z_{ADT}$) embeddings exist in separate feature spaces, making direct integration difficult. Simple concatenation or correlation methods often fail to align these spaces effectively, which can introduce artifact mismatches during clustering. Module 5 uses contrastive learning to project both modalities into a shared coordinate system where matching biological signals overlap perfectly.

### 2. The Step-by-Step Mechanism & Internal Flow
Modality-specific embeddings ($Z_{RNA}$ and $Z_{ADT}$) are projected into a common latent space.
The model forms positive pairs using the RNA and protein data from the same spatial spot.
Negative pairs are formed by pairing the RNA profile of a spot with the protein profiles of all other spots on the slide.
An optimization loss functions like a mathematical magnet, pulling positive pairs together while pushing negative pairs apart.

### 3. Algorithms & Deep Mathematical Logic
InfoNCE Objective Maximization: The contrastive alignment loss $\mathcal{L}_{cl}$ is defined as:
$$\mathcal{L}_{cl} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp\left(\text{sim}(Z_{RNA,i}, Z_{ADT,i}) / \tau\right)}{\sum_{j=1}^N \exp\left(\text{sim}(Z_{RNA,i}, Z_{ADT,j}) / \tau\right)}$$
Where $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ measures cosine similarity, $N=3484$, and $\tau$ is a temperature tuning parameter. Minimizing this objective maximizes the mutual information between the two modalities, synchronizing them into a shared coordinate system.

### 4. Inputs & Dimensionalities
Latent Components: $Z_{RNA} \in \mathbb{R}^{3484 \times d}$ and $Z_{ADT} \in \mathbb{R}^{3484 \times d}$

### 5. Outputs & Dimensionalities
Aligned Manifolds: Synchronized representations of $Z_{RNA}$ and $Z_{ADT}$, preparing them for multi-modal fusion.
Loss Value: $\mathcal{L}_{cl} \in \mathbb{R}^1$ (Fed into the Optimization Hub)

## Module 6: Adaptive Dual-Attention Fusion (SpatialGlue Logic)

### 1. The Core Architectural Problem Solved
Biological signals are not uniformly reliable across all tissue regions. For instance, in a lymph node, the RNA profile might be noisy or drop out in the medulla, while specific protein markers remain clear. Conversely, transcriptomics might offer better resolution for cell sub-populations in the cortex than a limited protein panel can provide. Module 6 addresses this by using a hierarchical attention mechanism that dynamically weights and prioritizes the most reliable data type at each spot.

### 2. The Step-by-Step Mechanism & Internal Flow
Within-Modality Evaluation: For each modality, the model learns attention weights to balance the contributions of the spatial physical layout ($A_s$) and the molecular similarity graph ($A_f$).
Between-Modality Evaluation: The model calculates an attention weight ($\omega$) for each spot to balance the importance of the RNA vs. Protein data stream.
Adaptive Fusion: The final integrated embedding is constructed as a weighted sum of the two modalities based on these spot-level attention values.

### 3. Algorithms & Deep Mathematical Logic
Hierarchical Modality Gating: For each spot $i$, attention coefficients ($\omega_{RNA,i}, \omega_{ADT,i}$) are calculated using a softmax gating network based on feature stability and entropy:
$$\omega_{RNA,i}, \omega_{ADT,i} = \text{Softmax}\left( \mathbf{v}^T \tanh\left( \mathbf{W}_g Z_{RNA,i} \right), \mathbf{v}^T \tanh\left( \mathbf{W}_g Z_{ADT,i} \right) \right)$$
Where $\mathbf{W}_g$ is a weight matrix and $\mathbf{v}$ is an attention vector.
Linear Fusion Projection: The unified latent space is constructed as:
$$Z_{Fused,i} = \left(\omega_{RNA,i} \cdot \mathbf{W}_R Z_{RNA,i}\right) + \left(\omega_{ADT,i} \cdot \mathbf{W}_A Z_{ADT,i}\right)$$
This allows the model to dynamically prioritize the more stable and high-signal modality at each spatial coordinate.

### 4. Inputs & Dimensionalities
Aligned Input Elements: Synchronized matrices $Z_{RNA}$ and $Z_{ADT}$ from Module 5.
Graph Matrices: $A_s \in \mathbb{R}^{3484 \times 3484}$ and $A_f \in \mathbb{R}^{3484 \times 3484}$

### 5. Outputs & Dimensionalities
Unified Master Embedding: $Z_{Fused} \in \mathbb{R}^{3484 \times 64}$ (A clean, low-dimensional summary of the integrated data)

## Module 7: Reconstruction & Regularization (The Optimization Hub)

### 1. The Core Architectural Problem Solved
Unsupervised deep models can suffer from trivial solutions or over-smoothing, where the latent space discards critical biological details or blurs spatial boundaries. Module 7 addresses this by combining all loss signals into a joint optimization framework. It features a reconstruction decoder that forces the model to retain original biological details, and a spatial regularization term that penalizes un-biologically sudden changes between adjacent spots.

### 2. The Step-by-Step Mechanism & Internal Flow
Decoding Path: Parallel Multi-Layer Perceptrons (MLPs) project the 64-dimensional $Z_{Fused}$ embedding back to the original dimensions of both modalities, checking for information loss.
Spatial Regularization Path: The model evaluates the differences between adjacent spots based on the spatial adjacency matrix ($A_s$), penalizing irregular variations.
Joint Optimization Hub: The system calculates Mean Squared Error (MSE) for data reconstruction, combines it with the contrastive loss ($\mathcal{L}_{cl}$) and spatial loss ($\mathcal{L}_{spat}$), and updates the entire network's weights via backpropagation.

### 3. Algorithms & Deep Mathematical Logic
Reconstruction Loss calculation (MSE): Measures how well the model reconstructs the normalized inputs:
$$\mathcal{L}_{recon} = \frac{1}{N}\sum_{i=1}^N \|\hat{X}_{RNA,i} - \tilde{X}_{RNA,i}\|^2 + \frac{1}{N}\sum_{i=1}^N \|\hat{X}_{ADT,i} - \tilde{X}_{ADT,i}\|^2$$
Spatial Smoothness Regularization: Implemented using a Graph Laplacian penalty to enforce spatial continuity:
$$\mathcal{L}_{spat} = \sum_{i=1}^N \sum_{j=1}^N A_{s,ij} \|Z_{Fused,i} - Z_{Fused,j}\|^2$$
This minimizes the distance between physical neighbors in the latent space, smoothing out technical noise while keeping structural boundaries intact.
Joint Objective Equation: The total optimization loss is calculated as:
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{cl} + \lambda_2 \mathcal{L}_{recon} + \lambda_3 \mathcal{L}_{spat}$$

### 4. Inputs & Dimensionalities
Latent Driver: $Z_{Fused} \in \mathbb{R}^{3484 \times 64}$
Spatial Roadmap: $A_s \in \mathbb{R}^{3484 \times 3484}$
Ground Truth Checkpoints: $\tilde{X}_{RNA} \in \mathbb{R}^{3484 \times 18085}$ and $\tilde{X}_{ADT} \in \mathbb{R}^{3484 \times 31}$

### 5. Outputs & Dimensionalities
Reconstructed Guess Vectors:
$\hat{X}_{RNA} \in \mathbb{R}^{3484 \times 18085}$ (Decoded transcript projections)
$\hat{X}_{ADT} \in \mathbb{R}^{3484 \times 31}$ (Decoded protein projections)
Optimization Signal: $\mathcal{L}_{total} \in \mathbb{R}^1$ (Triggers the parameter gradient updates across all modules)

# Phase 4: Unsupervised Biological Discovery

## Module 8: Spatial Domain Identification

### 1. The Core Architectural Problem Solved
The final goal of spatial omics is to discover distinct anatomical regions and cellular niches without manual bias. Performing clustering directly on high-dimensional raw data can suffer from noise and the "curse of dimensionality." Module 8 resolves this by performing community detection within the denoised, aligned, and optimized 64-dimensional $Z_{Fused}$ latent space, producing clear and accurate tissue domains.

### 2. The Step-by-Step Mechanism & Internal Flow
The trained model freezes its weights, and the decoding module is turned off.
The model builds a neighborhood graph in the $Z_{Fused}$ latent space.
Graph partitioning algorithms group similar spots into distinct biological communities.
Uniform Manifold Approximation and Projection (UMAP) compresses the 64-dimensional space into 2D for visual review.
The identified clusters are validated against manual annotations using the Adjusted Rand Index (ARI).

### 3. Algorithms & Deep Mathematical Logic
Leiden / Louvain Community Detection: Optimizes network modularity to find stable clusters:
$$Q = \frac{1}{2m} \sum_{ij} \left[ B_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
Where $B_{ij}$ is the latent space neighbor graph, $k$ represents node degrees, $m$ is total edge volume, and $\delta(c_i,c_j)$ ensures modularity is summed only for spots within the same cluster.
Adjusted Rand Index (ARI) Metric Verification: Computes clustering accuracy relative to ground truth annotations:
$$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}{\frac{1}{2} \left[ \sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} \right] - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}$$
An ARI of 1 indicates perfect match with manual annotations, confirming the model's accuracy.

### 4. Inputs & Dimensionalities
Final Stable Latent: $Z_{Fused} \in \mathbb{R}^{3484 \times 64}$
Validation Register: manual-anno arrays from annotation.csv

### 5. Outputs & Dimensionalities
Domain Array: Vector containing 3,484 discrete categorical domain labels (e.g., Follicle, Cortex, Medulla Cords).
Dimensionality Reduction Plot: 2D spatial coordinate coordinates and alternative 2D UMAP scatter visualizations.
Performance Metric Score: Real value ARI score assessing cluster quality.

# Technical Summary for Thesis Defense

## Core Operations Matrix
| Module | Key Algorithm Architecture | Primary Input Vector | Core Output Vector | Primary Engineering Advantage |
|--------|----------------------------|----------------------|--------------------|-------------------------------|
| 1. Preprocessing | Log1p Scale / CLR Matrix | Raw Counts | $\tilde{X}_{RNA}$ / $\tilde{X}_{ADT}$ | Stabilizes data scale and handles background antibody binding. |
| 2. Encoding | Multi-Head Self-Attention | $\tilde{X}_{RNA}$ | $H_{RNA} \in \mathbb{R}^{3484 \times 512}$ | Recovers missing signals and transcript dropouts using biological priors. |
| 3. Graph Build | Euclidean / Cosine KNN | Coordinates, $H_{RNA}$ | Adjacencies $A_s, A_f$ | Captures both physical structures and long-range cell similarities. |
| 4. Spatial Smooth | Dynamic GATv2 Encoders | Enrichment Vectors, $A$ | Latents $Z_{RNA}, Z_{ADT}$ | Smooths data while keeping sharp domain boundaries using attention. |
| 5. Alignment | InfoNCE Minimization | Modality Latents $Z$ | Aligned Coordinates | Synchronizes different data modalities into a shared coordinate space. |
| 6. Fusion | Hierarchical Gating Engine | Aligned Spaces, $A$ | $Z_{Fused} \in \mathbb{R}^{3484 \times 64}$ | Dynamically prioritizes the more reliable modality at each spot. |
| 7. Hub Loss | Parallel MLP / Graph Laplacian | $Z_{Fused}$ Manifold | Total Loss Scalar $\mathcal{L}_{total}$ | Balances biological detail with spatial continuity during training. |
| 8. Partition | Leiden / Louvain Cluster | $Z_{Fused}$ Manifold | Categorical Labels | Automatically identifies clear tissue domains with high ARI scores. |
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
When the features leave the Graph Attention Layers (Module 4), the transcriptomic representation ($\mathbf{Z}_{\text{RNA}}$) and the proteomic representation ($\mathbf{Z}_{\text{ADT}}$) are decoupled. Because they derive from different feature types (18,085 genes vs. 31 proteins), they exist in completely separate mathematical coordinate systems. If you fuse them without alignment, the model cannot calculate cross-modal relationships accurately.
Module 5 solves this by acting as a cross-modal synchronization bridge, mapping both paths into a shared coordinate space where a gene expression profile can directly "speak" to a surface protein marker.

### 2. The Step-by-Step Mechanism & Internal Flow
Batch Slicing & Pairing: For each spot $i$ out of the 3,484 spots, the model isolates its dual representations. It creates a Positive Pair by matching a spot's RNA vector with its own Protein vector ($\mathbf{Z}_{\text{RNA}, i}$ and $\mathbf{Z}_{\text{ADT}, i}$).
Negative Matrix Generation: The model pairs that same spot $i$'s RNA vector with the Protein vectors of all other spots in the training batch ($\mathbf{Z}_{\text{RNA}, i}$ and $\mathbf{Z}_{\text{ADT}, j \neq i}$), treating them as Negative Pairs.
Similarity Maximization (The Contrastive Tug-of-War): The module calculates a similarity score for all pairs. It shifts the weights of the encoders to pull positive pairs close together while aggressively pushing all negative pairs apart.

### 3. Algorithms & Deep Mathematical Logic
InfoNCE (Information Noise-Contrastive Estimation) Loss: The alignment is governed by a symmetric InfoNCE loss function. It calculates the probability that the network can correctly match a spot's RNA profile to its true protein profile out of a crowd of mismatched negative profiles:
$$\mathcal{L}_{\text{cl}} = -\frac{1}{2N}\sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(\mathbf{Z}_{\text{RNA},i}, \mathbf{Z}_{\text{ADT},i})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{Z}_{\text{RNA},i}, \mathbf{Z}_{\text{ADT},j})/\tau)} + \log \frac{\exp(\text{sim}(\mathbf{Z}_{\text{ADT},i}, \mathbf{Z}_{\text{RNA},i})/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{Z}_{\text{ADT},i}, \mathbf{Z}_{\text{RNA},j})/\tau)} \right]$$
Where $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}^T}{\|\mathbf{u}\| \|\mathbf{v}\|}$ (Cosine Similarity) and $\tau$ is a temperature parameter that scales the sharpness of the penalties.

### 4. Inputs & Dimensionalities
Uncalibrated Embeddings: $\mathbf{Z}_{\text{RNA}} \in \mathbb{R}^{3484 \times d}$ and $\mathbf{Z}_{\text{ADT}} \in \mathbb{R}^{3484 \times d}$ (Spatially aware outputs directly from Module 4).

### 5. Outputs & Dimensionalities
Aligned Manifold Matrices: Separated but synchronized matrices $\mathbf{Z}_{\text{RNA}}$ and $\mathbf{Z}_{\text{ADT}}$, sharing identical coordinate fields, alongside the scalar error matrix $\mathcal{L}_{\text{cl}}$ sent to the total training loop.

## Module 6: Adaptive Dual-Attention Fusion (SpatialGlue Logic)

### 1. The Core Architectural Problem Solved
Biological signals are not uniformly reliable across all tissue regions. In a human lymph node, the transcriptomic profile might suffer from high dropout rates in the Medulla, while specific proteomic surface antibody markers remain stable. Conversely, in the Cortex, transcriptomics offers a deeper view of cell sub-populations than a limited 31-protein panel can capture.
Module 6 resolves this by implementing a Hierarchical (Two-Tier) Attention Mechanism. Instead of treating all data equally everywhere, it dynamically evaluates data quality spot-by-spot to prioritize the highest-signal modality at each physical coordinate.

### 2. Tier 1: Within-Modality Attention (Graph Blending)
Before mixing genes and proteins together, the model must optimize the representation inside each individual modality. Each modality has two competing neighborhood "opinions" from Module 4: the Physical Layout ($A_s$) and the Biological Similarity Graph ($A_f$).

The Specific Mechanism:
For the RNA Stream: The model looks at a spot and decides: "In this specific tissue pocket, should I trust physical grid structures ($A_s$) or non-local molecular similarities ($A_f$) more?" * For the ADT Stream: Simultaneously and independently, the model performs the same evaluation for the protein graph structures.
The Operation: It calculates two attention weights ($\alpha_s, \alpha_f$) for each modality. The output is a blended, optimized representation for each stream.

Mathematical Logic (Graph Blending):
For each spot $i$ in a specific modality $m$ (where $m \in \{\text{RNA}, \text{ADT}\}$):
$$\alpha_{s, i}^m, \alpha_{f, i}^m = \text{Softmax}\left( \text{MLP}_m(Z_{m, i} \cdot A_s), \text{MLP}_m(Z_{m, i} \cdot A_f) \right)$$
The updated, within-modality optimized embeddings are formed by weighting the graph behaviors:
$$\tilde{Z}_{\text{RNA}, i} = \alpha_{s, i}^{\text{RNA}} (\mathbf{W}_{s}^{\text{RNA}} Z_{\text{RNA}, i}) + \alpha_{f, i}^{\text{RNA}} (\mathbf{W}_{f}^{\text{RNA}} Z_{\text{RNA}, i})$$
$$\tilde{Z}_{\text{ADT}, i} = \alpha_{s, i}^{\text{ADT}} (\mathbf{W}_{s}^{\text{ADT}} Z_{\text{ADT}, i}) + \alpha_{f, i}^{\text{ADT}} (\mathbf{W}_{f}^{\text{ADT}} Z_{\text{ADT}, i})$$

### 3. Tier 2: Between-Modality Attention (Modality Gating)
Now that the model has the "best possible version" of the RNA stream ($\tilde{Z}_{\text{RNA}}$) and the ADT stream ($\tilde{Z}_{\text{ADT}}$), it finally merges the two distinct modalities.

The Specific Mechanism:
The Operation: The model calculates a dynamic, spot-specific gate weight called $\omega$ (omega).
The Logic: For every single one of your 3,484 spots, the network measures feature stability and entropy across modalities. If a spot is deep within a B-cell Follicle where proteomic markers like CD19 are exceptionally clean, $\omega_{\text{ADT}}$ is automatically increased, and $\omega_{\text{RNA}}$ is decreased for that specific spot.

Mathematical Logic (Modality Gating & Linear Projection):
The spot-specific modal coefficients are calculated using a Tanh-activated gating network:
$$\omega_{\text{RNA}, i}, \omega_{\text{ADT}, i} = \text{Softmax}\left( \mathbf{v}^T \tanh\left( \mathbf{W}_g \tilde{Z}_{\text{RNA}, i} \right), \mathbf{v}^T \tanh\left( \mathbf{W}_g \tilde{Z}_{\text{ADT}, i} \right) \right)$$
Where $\mathbf{W}_g$ is a shared gate weight matrix and $\mathbf{v}$ is an attention vector.
The final Unified Latent Space Matrix ($Z_{\text{Fused}}$) is constructed as the definitive cross-modal summation:
$$Z_{\text{Fused}, i} = \left(\omega_{\text{RNA}, i} \cdot \mathbf{W}_R \tilde{Z}_{\text{RNA}, i}\right) + \left(\omega_{\text{ADT}, i} \cdot \mathbf{W}_A \tilde{Z}_{\text{ADT}, i}\right)$$
Where $\mathbf{W}_R$ and $\mathbf{W}_A$ are linear projection matrices that compress the dimensions down to the final target width.

### 4. Inputs & Dimensionalities
Aligned Input Tensors: Synchronized matrices $Z_{\text{RNA}} \in \mathbb{R}^{3484 \times d}$ and $Z_{\text{ADT}} \in \mathbb{R}^{3484 \times d}$ from Module 5.
Topological Graph Tensors: Physical Grid Matrix $A_s \in \mathbb{R}^{3484 \times 3484}$ and Cellular Similarity Matrix $A_f \in \mathbb{R}^{3484 \times 3484}$ from Module 3.

### 5. Outputs & Dimensionalities
Unified Master Embedding: $Z_{\text{Fused}} \in \mathbb{R}^{3484 \times 64}$

## Module 7: Reconstruction & Regularization ("Decoder" & Hub)

### 1. The Core Architectural Problem Solved
If a model only compresses data down to 64 dimensions ($\mathbf{Z}_{\text{Fused}}$), it can easily drop vital biological signatures or over-smooth small spatial structures to make the mathematical optimization easier.
Module 7 addresses this by running a Dual-Verification System. It forces the model to prove it hasn't forgotten the original data by attempting to rebuild it completely (Reconstruction), while simultaneously checking that the physical layout is smooth (Regularization).

### 2. The Step-by-Step Mechanism & Internal Flow
Feature Stretching (Decoding): The module passes the 64-dimensional $\mathbf{Z}_{\text{Fused}}$ latent embedding into two separate, parallel Multi-Layer Perceptrons (MLPs). The RNA Decoder stretches the data back out to 18,085 dimensions ($\hat{\mathbf{X}}_{\text{RNA}}$), and the ADT Decoder stretches it to 31 dimensions ($\hat{\mathbf{X}}_{\text{ADT}}$).
Fidelity Cross-Check (MSE Loss): The model calculates the Mean Squared Error (MSE) by directly comparing these high-dimensional reconstructions against the actual baseline normalized matrices ($\tilde{\mathbf{X}}_{\text{RNA}}, \tilde{\mathbf{X}}_{\text{ADT}}$) obtained back in Module 1.
Neighborhood Evaluation (Spatial Loss): Simultaneously, the module uses the physical grid roadmap ($\mathbf{A}_s$) to check the latent space. It measures the distance between the 64-dimensional vectors of spots that are physically touching. If neighbors look completely different, it triggers a penalty.
Loss Summation: All errors are compiled into $\mathcal{L}_{\text{total}}$ to train the entire network via backpropagation.

### 3. Algorithms & Deep Mathematical Logic
Reconstruction Loss (MSE): Quantifies how much original biological information was preserved during compression:
$$\mathcal{L}_{\text{recon}} = \frac{1}{N}\sum_{i=1}^{N} \|\tilde{\mathbf{X}}_{\text{RNA}, i} - \hat{\mathbf{X}}_{\text{RNA}, i}\|^2 + \frac{1}{N}\sum_{i=1}^{N} \|\tilde{\mathbf{X}}_{\text{ADT}, i} - \hat{\mathbf{X}}_{\text{ADT}, i}\|^2$$
Spatial Regularization Loss (Graph Laplacian Smoothing): Penalizes sudden molecular jumps between immediate physical neighbors to eliminate technical "salt-and-pepper" noise:
$$\mathcal{L}_{\text{spat}} = \sum_{i,j} \mathbf{A}_{s, ij} \|\mathbf{Z}_{\text{Fused}, i} - \mathbf{Z}_{\text{Fused}, j}\|^2$$
Joint Objective Optimization Function: The definitive optimization formula that trains KAC-Net:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{cl}} + \lambda_2 \mathcal{L}_{\text{recon}} + \lambda_3 \mathcal{L}_{\text{spat}}$$
Where $\lambda_{1,2,3}$ are balancing hyperparameters.

### 4. Inputs & Dimensionalities
Latent Feature Vector: $\mathbf{Z}_{\text{Fused}} \in \mathbb{R}^{3484 \times 64}$ from Module 6.
Physical Topology Grid: $\mathbf{A}_s \in \mathbb{R}^{3484 \times 3484}$ from Module 3.
Ground Truth Checkpoints: Normalized inputs $\tilde{\mathbf{X}}_{\text{RNA}} \in \mathbb{R}^{3484 \times 18085}$ and $\tilde{\mathbf{X}}_{\text{ADT}} \in \mathbb{R}^{3484 \times 31}$ from Module 1.

### 5. Outputs & Dimensionalities
Reconstructed Matrices: $\hat{\mathbf{X}}_{\text{RNA}} \in \mathbb{R}^{3484 \times 18085}$ and $\hat{\mathbf{X}}_{\text{ADT}} \in \mathbb{R}^{3484 \times 31}$ (Used only during training to compute the loss value).
Scalar Total Loss ($\mathcal{L}_{\text{total}}$): The ultimate error metric driving the gradient updates across the entire neural network.

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
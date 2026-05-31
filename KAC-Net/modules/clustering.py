"""
Module 8: Spatial Domain Identification (Clustering & Discovery)

Core Purpose:
    Identify distinct anatomical regions and cellular niches in spatial tissue
    by performing unsupervised community detection within the denoised,
    aligned, and optimized 64-dimensional Z_Fused latent space from Module 7.

Key Architecture:
    • Leiden Community Detection: Optimizes modularity to find stable clusters
    • UMAP Dimensionality Reduction: Compresses 64-dim to 2D for visualization
    • ARI Validation: Compares predicted domains against manual annotations
    • Domain Discovery: Output categorical labels for anatomical regions
    • Resolution Sweep: Finds optimal resolution that maximizes ARI

Inputs:
    • Z_Fused ∈ R^(3484 × 64) - Learned latent embeddings from Module 7 training
    • Ground Truth (optional): Manual annotations from annotation.csv

Outputs:
    • domain_labels ∈ Z^(3484,) - Categorical domain assignments (0, 1, 2, ...)
    • umap_coords ∈ R^(3484 × 2) - 2D UMAP coordinates for visualization
    • ari_score ∈ [-1, 1] - Adjusted Rand Index (1 = perfect match with ground truth)
    • optimal_resolution - Best resolution from sweep (if ground truth provided)

Mathematical Foundation:
    Leiden Modularity Optimization:
        Q = (1/2m) * Σ_ij [B_ij - (k_i * k_j) / 2m] * δ(c_i, c_j)
        where B_ij is quality matrix, k_i is node degree, δ is cluster indicator

    Adjusted Rand Index (ARI):
        ARI = [Σ_ij C(n_ij, 2) - [Σ_i C(a_i, 2) * Σ_j C(b_j, 2)] / C(n, 2)]
              / {0.5 * [Σ_i C(a_i, 2) + Σ_j C(b_j, 2)] - [Σ_i C(a_i, 2) * Σ_j C(b_j, 2)] / C(n, 2)}

References:
    • Leiden: Community detection with modularity optimization
    • UMAP: Uniform Manifold Approximation and Projection
    • module_explanation.md: Complete mathematical specification
    • flow.md: Algorithm, inputs, outputs, mechanisms
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import scanpy as sc
import anndata as ad


class SpatialDomainIdentifier(nn.Module):
    """
    Main Module 8 class for spatial domain identification.

    Performs unsupervised community detection on the learned latent space
    to identify anatomically/functionally distinct tissue domains.

    Purpose:
        Partition 3,484 spots into distinct biological communities (domains)
        based on their learned representations without manual intervention.

    Args:
        latent_dim (int): Latent space dimension (64 from Module 7)
        n_neighbors (int): Neighbors for neighborhood graph (default: 15)
        leiden_resolution (float): Resolution for Leiden clustering (default: 1.0)
        umap_min_dist (float): UMAP minimum distance (default: 0.1)

    Mathematical Concept:
        Leiden algorithm optimizes a quality function Q over all partitions C:
        Q(C) = (1/2m) * Σ_{i,j} [B_ij - (k_i * k_j)/(2m)] * δ(C_i, C_j)
        
        Higher Q indicates better cluster structure. Resolution parameter
        controls cluster size: larger values → more, smaller clusters.

    Inputs:
        z_fused (torch.Tensor): Shape (N, 64) - Learned embeddings from training

    Outputs:
        domain_labels (np.ndarray): Shape (N,) - Cluster assignment per spot
        umap_coords (np.ndarray): Shape (N, 2) - 2D visualization coordinates
        metrics (dict): Clustering quality metrics (ARI, NMI, modularity, etc.)
    """

    def __init__(self, latent_dim=64, n_neighbors=15, leiden_resolution=1.0,
                 umap_min_dist=0.1, umap_n_components=2):
        """Initialize spatial domain identifier with Leiden + UMAP."""
        super(SpatialDomainIdentifier, self).__init__()
        self.latent_dim = latent_dim
        self.n_neighbors = n_neighbors
        self.leiden_resolution = leiden_resolution
        self.umap_min_dist = umap_min_dist
        self.umap_n_components = umap_n_components

    def forward(self, z_fused, ground_truth_labels=None):
        """
        Identify spatial domains from learned embeddings.

        Args:
            z_fused (torch.Tensor): Shape (N, 64) - Learned latent embeddings
            ground_truth_labels (np.ndarray, optional): Shape (N,) - Manual annotations

        Returns:
            domain_labels (np.ndarray): Shape (N,) - Predicted cluster labels
            umap_coords (np.ndarray): Shape (N, 2) - UMAP coordinates
            metrics (dict): Quality metrics (ari_score, nmi_score, modularity, n_clusters)

        Algorithm Flow:
            1. Convert torch tensor to numpy
            2. Create AnnData object (required by scanpy)
            3. Compute neighborhood graph (k=15)
            4. Apply Leiden clustering (resolution=1.0)
            5. Compute UMAP projection (2D visualization)
            6. Calculate ARI score (if ground truth provided)
            7. Return all results
        """
        # Convert torch tensor to numpy and standardize
        if isinstance(z_fused, torch.Tensor):
            z_fused_np = z_fused.detach().cpu().numpy()
        else:
            z_fused_np = z_fused

        # Standardize for better neighbor computation
        scaler = StandardScaler()
        z_scaled = scaler.fit_transform(z_fused_np)

        # Create AnnData object (scanpy's native format)
        adata = ad.AnnData(X=z_scaled)
        adata.obsm['latent'] = z_scaled

        # Step 1: Compute k-NN neighborhood graph in latent space
        sc.pp.neighbors(
            adata,
            use_rep='latent',
            n_neighbors=self.n_neighbors,
            metric='euclidean'
        )

        # Step 2: Leiden clustering (modularity optimization)
        sc.tl.leiden(
            adata,
            resolution=self.leiden_resolution,
            key_added='leiden'
        )

        # Step 3: UMAP dimensionality reduction (for visualization)
        sc.tl.umap(
            adata,
            min_dist=self.umap_min_dist,
            n_components=self.umap_n_components
        )

        # Extract results
        domain_labels = np.array(adata.obs['leiden']).astype(int)
        umap_coords = adata.obsm['X_umap']

        # Step 4: Compute ARI score if ground truth provided
        ari_score = None
        nmi_score = None
        if ground_truth_labels is not None:
            # Convert to numpy if needed
            if isinstance(ground_truth_labels, torch.Tensor):
                gt = ground_truth_labels.detach().cpu().numpy()
            else:
                gt = ground_truth_labels

            # Compute ARI and NMI
            ari_score = adjusted_rand_score(gt, domain_labels)
            nmi_score = normalized_mutual_info_score(gt, domain_labels)

        # Compute modularity (quality of clustering)
        modularity = self._compute_modularity(adata)

        # Compile metrics
        metrics = {
            'n_clusters': len(np.unique(domain_labels)),
            'modularity': modularity,
            'ari_score': ari_score,
            'nmi_score': nmi_score,
            'leiden_resolution': self.leiden_resolution,
            'n_neighbors': self.n_neighbors
        }

        return domain_labels, umap_coords, metrics

    def _compute_modularity(self, adata):
        """
        Compute modularity score of Leiden clustering.

        Formula: Q = (1/2m) * Σ_ij [A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)

        Args:
            adata (anndata.AnnData): AnnData object with computed clustering

        Returns:
            modularity (float): Modularity score [0, 1]
        """
        # Get adjacency matrix and cluster labels
        adj_matrix = adata.obsp['distances'].tocsr()
        adj_binary = (adj_matrix > 0).astype(float)  # Binary adjacency
        cluster_labels = np.array(adata.obs['leiden']).astype(int)

        # Compute modularity
        n_nodes = adj_binary.shape[0]
        n_edges = adj_binary.nnz / 2.0  # Undirected graph
        degrees = np.array(adj_binary.sum(axis=1)).flatten()

        modularity = 0.0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj_binary[i, j] > 0:  # If connected
                    if cluster_labels[i] == cluster_labels[j]:  # Same cluster
                        modularity += adj_binary[i, j] - (degrees[i] * degrees[j]) / (2 * n_edges)

        if n_edges > 0:
            modularity /= (2 * n_edges)
        else:
            modularity = 0.0

        return float(modularity)

    def compute_domain_statistics(self, domain_labels, metadata=None):
        """
        Compute statistics about discovered domains.

        Args:
            domain_labels (np.ndarray): Shape (N,) - Domain assignments
            metadata (pd.DataFrame, optional): Metadata for each spot

        Returns:
            stats (dict): Domain sizes, composition, etc.
        """
        unique_domains, counts = np.unique(domain_labels, return_counts=True)

        stats = {
            'n_domains': len(unique_domains),
            'domain_sizes': dict(zip(unique_domains.astype(str), counts)),
            'mean_domain_size': float(np.mean(counts)),
            'std_domain_size': float(np.std(counts)),
            'domain_range': (int(np.min(counts)), int(np.max(counts)))
        }

        return stats


def load_ground_truth_annotations(annotation_csv_path):
    """
    Load and encode ground truth annotations from annotation.csv.

    The annotation file should have columns: Barcode, manual-anno
    This function converts string domain names to numeric labels.

    Args:
        annotation_csv_path (str): Path to annotation.csv file

    Returns:
        gt_labels (np.ndarray): Shape (N,) - Numeric domain labels
        label_mapping (dict): String → int mapping
        inv_label_mapping (dict): Int → string mapping
        n_domains (int): Total number of unique domains

    Example:
        >>> gt_labels, mapping, inv_mapping, n_domains = load_ground_truth_annotations(
        ...     'data/10x_human_lymph_node_A1/annotation.csv'
        ... )
        >>> print(f"Domains found: {n_domains}")
        >>> print(f"Mapping: {inv_mapping}")
    """
    # Load CSV
    df = pd.read_csv(annotation_csv_path)

    # Extract manual annotations column
    annotations = df['manual-anno'].values

    # Create unique label mapping
    unique_domains = np.unique(annotations)
    label_mapping = {domain: idx for idx, domain in enumerate(unique_domains)}
    inv_label_mapping = {idx: domain for domain, idx in label_mapping.items()}

    # Convert string labels to integers
    gt_labels = np.array([label_mapping[anno] for anno in annotations], dtype=int)

    n_domains = len(unique_domains)

    if len(unique_domains) > 0:
        print(f"\n✅ Loaded {len(gt_labels)} annotations with {n_domains} unique domains:")
        for domain_name, domain_idx in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = np.sum(gt_labels == domain_idx)
            print(f"   [{domain_idx}] {domain_name:<30} : {count:>5} spots")

    return gt_labels, label_mapping, inv_label_mapping, n_domains


def leiden_clustering_with_sweep(z_fused, ground_truth_labels=None, n_neighbors=15,
                                 res_start=0.2, res_end=2.0, n_steps=15,
                                 verbose=True):
    """
    Perform Leiden clustering with resolution sweep to find optimal configuration.

    This function sweeps through resolution values and finds the resolution
    that produces clustering with maximum Adjusted Rand Index (ARI) against
    ground truth annotations. If no ground truth provided, uses fixed resolution.

    Purpose:
        Discover optimal community structure in latent space that best matches
        biological ground truth domains.

    Args:
        z_fused (np.ndarray or torch.Tensor): Shape (N, 64) - Latent embeddings
        ground_truth_labels (np.ndarray, optional): Shape (N,) - Manual domain annotations
        n_neighbors (int): Neighbors for graph construction (default: 15)
        res_start (float): Resolution sweep start (default: 0.2)
        res_end (float): Resolution sweep end (default: 2.0)
        n_steps (int): Number of resolution values to test (default: 15)
        verbose (bool): Print results during sweep (default: True)

    Returns:
        results (dict): Dictionary containing:
            - 'domain_labels': Best predicted cluster labels (N,)
            - 'umap_coords': UMAP 2D coordinates (N, 2)
            - 'optimal_resolution': Best resolution if ground truth provided, else fixed
            - 'best_ari_score': Maximum ARI achieved (None if no ground truth)
            - 'best_nmi_score': NMI at optimal resolution (None if no ground truth)
            - 'n_clusters': Number of clusters at optimal
            - 'modularity': Modularity score at optimal
            - 'sweep_results': All resolution tests (if ground truth provided)

    Example:
        >>> results = leiden_clustering_with_sweep(
        ...     z_fused,
        ...     ground_truth_labels=gt_labels,
        ...     res_start=0.2,
        ...     res_end=2.0,
        ...     n_steps=15
        ... )
        >>> print(f"✅ Optimal resolution: {results['optimal_resolution']:.3f}")
        >>> print(f"   ARI Score: {results['best_ari_score']:.4f}")
    """
    # If no ground truth, use fixed resolution
    if ground_truth_labels is None:
        if verbose:
            print("\n⚠️  No ground truth provided. Using fixed resolution = 1.0")
        
        identifier = SpatialDomainIdentifier(
            n_neighbors=n_neighbors,
            leiden_resolution=1.0
        )
        
        domain_labels, umap_coords, metrics = identifier(z_fused, ground_truth_labels=None)
        
        return {
            'domain_labels': domain_labels,
            'umap_coords': umap_coords,
            'optimal_resolution': 1.0,
            'best_ari_score': None,
            'best_nmi_score': None,
            'n_clusters': metrics['n_clusters'],
            'modularity': metrics['modularity'],
            'sweep_results': None
        }

    # Otherwise, perform sweep to find optimal resolution
    resolutions = np.linspace(res_start, res_end, n_steps)
    sweep_results = []

    best_ari = -1
    best_resolution = 1.0
    best_result_data = None

    if verbose:
        print(f"\n{'Resolution':<12} {'ARI':<10} {'N_Clust':<10} {'Modularity':<12}")
        print("-" * 48)

    for res in resolutions:
        # Create identifier with specific resolution
        identifier = SpatialDomainIdentifier(
            n_neighbors=n_neighbors,
            leiden_resolution=res
        )

        # Run clustering
        domain_labels, umap_coords, metrics = identifier(
            z_fused,
            ground_truth_labels=ground_truth_labels
        )

        # Extract ARI (critical metric)
        ari = metrics['ari_score'] if metrics['ari_score'] is not None else -1

        # Record result
        result = {
            'resolution': res,
            'ari_score': ari,
            'nmi_score': metrics['nmi_score'],
            'n_clusters': metrics['n_clusters'],
            'modularity': metrics['modularity'],
            'domain_labels': domain_labels,
            'umap_coords': umap_coords
        }
        sweep_results.append(result)

        # Track best
        if ari > best_ari:
            best_ari = ari
            best_resolution = res
            best_result_data = result

        if verbose:
            print(f"{res:<12.3f} {ari:<10.4f} {metrics['n_clusters']:<10} {metrics['modularity']:<12.4f}")

    if verbose:
        print("-" * 48)
        print(f"\n🎯 OPTIMAL: Resolution = {best_resolution:.3f}, ARI = {best_ari:.4f}\n")

    # Compile final result
    return {
        'domain_labels': best_result_data['domain_labels'],
        'umap_coords': best_result_data['umap_coords'],
        'optimal_resolution': float(best_resolution),
        'best_ari_score': float(best_ari),
        'best_nmi_score': float(best_result_data['nmi_score']) if best_result_data['nmi_score'] is not None else None,
        'n_clusters': int(best_result_data['n_clusters']),
        'modularity': float(best_result_data['modularity']),
        'sweep_results': sweep_results
    }


def compute_umap_projection(z_fused, n_components=2, min_dist=0.1, n_neighbors=15):
    """
    Compute UMAP 2D projection of latent space for visualization.

    Args:
        z_fused (np.ndarray or torch.Tensor): Shape (N, 64) - Latent embeddings
        n_components (int): Target dimensionality (default: 2)
        min_dist (float): UMAP minimum distance parameter
        n_neighbors (int): UMAP neighbors parameter

    Returns:
        umap_coords (np.ndarray): Shape (N, n_components) - Reduced coordinates
    """
    # Convert to numpy if needed
    if isinstance(z_fused, torch.Tensor):
        z_np = z_fused.detach().cpu().numpy()
    else:
        z_np = z_fused

    # Standardize
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_np)

    # Create AnnData for scanpy
    adata = ad.AnnData(X=z_scaled)
    adata.obsm['latent'] = z_scaled

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata, use_rep='latent', n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist, n_components=n_components)

    return adata.obsm['X_umap']


def compute_ari_score(predicted_labels, ground_truth_labels):
    """
    Compute Adjusted Rand Index (ARI) between predicted and true labels.

    Mathematical Formula:
        ARI = [Σ_ij C(n_ij, 2) - [Σ_i C(a_i, 2) * Σ_j C(b_j, 2)] / C(n, 2)]
              / {0.5 * [Σ_i C(a_i, 2) + Σ_j C(b_j, 2)] - [Σ_i C(a_i, 2) * Σ_j C(b_j, 2)] / C(n, 2)}

    Where C(n, k) is the binomial coefficient.

    Interpretation:
        ARI = 1: Perfect agreement (predicted = true)
        ARI = 0: Random partitioning
        ARI < 0: Worse than random

    Args:
        predicted_labels (np.ndarray or torch.Tensor): Shape (N,) - Predicted clusters
        ground_truth_labels (np.ndarray or torch.Tensor): Shape (N,) - True clusters

    Returns:
        ari_score (float): ARI in range [-1, 1]
    """
    # Convert to numpy if needed
    if isinstance(predicted_labels, torch.Tensor):
        pred = predicted_labels.detach().cpu().numpy()
    else:
        pred = predicted_labels

    if isinstance(ground_truth_labels, torch.Tensor):
        true = ground_truth_labels.detach().cpu().numpy()
    else:
        true = ground_truth_labels

    # Compute ARI using sklearn
    ari = adjusted_rand_score(true, pred)

    return float(ari)


class DomainVisualizationUtils:
    """Utility class for domain visualization and analysis."""

    @staticmethod
    def prepare_visualization_data(z_fused, domain_labels, umap_coords, metadata=None):
        """
        Prepare comprehensive visualization dataset.

        Args:
            z_fused (np.ndarray): Latent embeddings (N, 64)
            domain_labels (np.ndarray): Domain assignments (N,)
            umap_coords (np.ndarray): UMAP coordinates (N, 2)
            metadata (pd.DataFrame, optional): Spot metadata

        Returns:
            viz_data (dict): Organized data for plotting
        """
        viz_data = {
            'z_fused': z_fused,
            'domain_labels': domain_labels,
            'umap_coords': umap_coords,
            'metadata': metadata,
            'n_domains': len(np.unique(domain_labels)),
            'domain_colors': DomainVisualizationUtils._generate_colors(
                len(np.unique(domain_labels))
            )
        }
        return viz_data

    @staticmethod
    def _generate_colors(n_clusters, colormap='tab20'):
        """Generate distinct colors for clusters."""
        import matplotlib.pyplot as plt
        if n_clusters <= 20:
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i) for i in range(n_clusters)]
        else:
            cmap = plt.get_cmap('hsv')
            colors = [cmap(i / n_clusters) for i in range(n_clusters)]
        return colors

    @staticmethod
    def get_domain_names(domain_labels, manual_annotations=None):
        """
        Map domain indices to biological names.

        Args:
            domain_labels (np.ndarray): Domain indices
            manual_annotations (dict, optional): Index → name mapping

        Returns:
            domain_names (np.ndarray): Named domains
        """
        if manual_annotations is None:
            # Default biological domain names for lymph node
            default_names = {
                0: 'B_Follicles',
                1: 'T_Zones',
                2: 'Germinal_Centers',
                3: 'Marginal_Zone',
                4: 'Medulla',
                5: 'Cortex'
            }
            manual_annotations = default_names

        domain_names = np.array([
            manual_annotations.get(int(d), f'Domain_{d}')
            for d in domain_labels
        ])

        return domain_names

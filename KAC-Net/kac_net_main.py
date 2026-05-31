"""
KAC-Net: Knowledge-Aware Cascaded Network for Spatial Multi-Omics Integration

Main Orchestrator - Integrates all 8 modules into a unified pipeline

Architecture:
    PHASE 1: Feature Extraction
        Module 1: Multimodal Preprocessing (RNA, ADT normalization)
        Module 2: Knowledge-Enriched Encoding (spaLLM transformer)
    
    PHASE 2: Structural Mapping
        Module 3: Multi-Graph Construction (spatial + feature graphs)
        Module 4: Local Spatial Encoding (Residual GATv2)
    
    PHASE 3: Integration & Optimization
        Module 5: Cross-Modal Contrastive Alignment (InfoNCE loss)
        Module 6: Adaptive Dual-Attention Fusion (hierarchical gating)
        Module 7: Reconstruction & Regularization (decoders + losses)
    
    PHASE 4: Unsupervised Discovery
        Module 8: Spatial Domain Identification (Leiden clustering)

Data Flow:
    Input (3,484 spots × 18,085 RNA genes × 31 ADT proteins)
        ↓
    [Module 1] Preprocessing → X̃_RNA, X̃_ADT
        ↓
    [Module 2] Encoding → H_RNA (3,484 × 512)
        ↓
    [Module 3] Graph Construction → A_s, A_f
        ↓
    [Module 4] Spatial Encoding → Z_RNA, Z_ADT (3,484 × d)
        ↓
    [Module 5] Contrastive Alignment → Aligned embeddings
        ↓
    [Module 6] Dual Attention Fusion → Z_Fused (3,484 × 64)
        ↓
    [Module 7] Reconstruction Loss → Training (50 epochs)
        ↓
    [Module 8] Domain Identification → Domain labels + UMAP + ARI
        ↓
    Output: 7 lymph node anatomical domains with validation scores

References:
    • KAC-Net_MASTER_PLAN.md: Complete architecture specification
    • module_explanation.md: Individual module mathematics
    • flow.md: Data flow and algorithm details
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Dict, Tuple, Optional
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# Import all 8 modules
from modules.preprocessing import PreprocessingModule
from modules.encoding import EncodingModule
from modules.graph_construction import GraphConstructionModule
from modules.spatial_encoding import SpatialEncodingModule
from modules.contrastive_alignment import ContrastiveAlignmentModule
from modules.dual_attention_fusion import DualAttentionFusionModule
from modules.reconstruction_loss import ReconstructionModule
from modules.clustering import leiden_clustering_with_sweep, load_ground_truth_annotations


class KACNet(nn.Module):
    """
    Complete KAC-Net pipeline orchestrator.
    
    Integrates all 8 modules in correct data flow sequence.
    Handles training loop, loss aggregation, and evaluation.
    
    Purpose:
        End-to-end spatial multi-omics integration via cascaded modules.
        Learns optimal latent representation (Z_Fused) that:
        - Preserves spatial relationships
        - Aligns multi-modal information
        - Enables biological domain discovery
    
    Args:
        config (dict): Configuration dictionary with hyperparameters
        device (str): Device placement ('cpu' or 'cuda')
    
    Attributes:
        module1-8: Individual module instances
        optimizer: Adam optimizer for all parameters
        scheduler: Learning rate scheduler
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """Initialize KAC-Net with all 8 modules."""
        super(KACNet, self).__init__()
        
        self.config = config
        self.device = device
        
        # ============ PHASE 1: Feature Extraction ============
        self.module1 = PreprocessingModule()
        self.module2 = EncodingModule(
            input_dim=config.get('rna_dim', 18085),
            output_dim=config.get('encoding_dim', 512),
            num_layers=config.get('encoding_layers', 2)
        )
        
        # ============ PHASE 2: Structural Mapping ============
        self.module3 = GraphConstructionModule(
            n_neighbors_spatial=config.get('k_spatial', 6),
            similarity_metric=config.get('similarity_metric', 'cosine')
        )
        self.module4 = SpatialEncodingModule(
            input_rna_dim=config.get('encoding_dim', 512),
            input_adt_dim=config.get('adt_dim', 31),
            hidden_dim=config.get('gat_hidden', 256),
            output_dim=config.get('latent_dim', 64),
            n_heads=config.get('n_attention_heads', 4)
        )
        
        # ============ PHASE 3: Integration & Optimization ============
        self.module5 = ContrastiveAlignmentModule(
            embedding_dim=config.get('latent_dim', 64),
            temperature=config.get('contrastive_temp', 0.07)
        )
        self.module6 = DualAttentionFusionModule(
            latent_dim=config.get('latent_dim', 64),
            output_dim=config.get('fusion_output_dim', 64)
        )
        self.module7 = ReconstructionModule(
            latent_dim=config.get('fusion_output_dim', 64),
            rna_dim=config.get('rna_dim', 18085),
            adt_dim=config.get('adt_dim', 31)
        )
        
        # Move to device
        self.to(device)
        
        # Optimizer for all parameters
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_decay_steps', 10),
            gamma=config.get('lr_decay_gamma', 0.5)
        )
        
        # Loss weights
        self.lambda_contrastive = config.get('lambda_contrastive', 0.5)
        self.lambda_reconstruction = config.get('lambda_reconstruction', 1.0)
        self.lambda_spatial = config.get('lambda_spatial', 0.3)
        
        # Training history
        self.training_history = {
            'epoch': [],
            'loss_total': [],
            'loss_contrastive': [],
            'loss_reconstruction': [],
            'loss_spatial': [],
            'learning_rate': []
        }
    
    def forward(self, rna, adt, spatial_coords, adj_s, adj_f):
        """
        Complete forward pass through all 8 modules.
        
        Args:
            rna (torch.Tensor): Shape (N, 18085) - RNA expression
            adt (torch.Tensor): Shape (N, 31) - Protein expression
            spatial_coords (torch.Tensor): Shape (N, 2) - Spatial coordinates
            adj_s (torch.sparse_coo_tensor): Spatial adjacency matrix
            adj_f (torch.sparse_coo_tensor): Feature adjacency matrix
        
        Returns:
            outputs (dict): Dictionary containing:
                - z_fused: Final latent embeddings (N, 64)
                - rna_recon: Reconstructed RNA
                - adt_recon: Reconstructed ADT
                - embeddings_dict: All intermediate embeddings
                - losses_dict: All computed losses
        """
        
        # ========== MODULE 1: Preprocessing ==========
        # Normalize both modalities
        x_rna_norm, x_adt_norm = self.module1(rna, adt)
        
        # ========== MODULE 2: Knowledge-Enriched Encoding ==========
        # spaLLM transformer for gene recovery
        h_rna = self.module2(x_rna_norm)  # (N, 512)
        
        # ========== MODULE 3: Multi-Graph Construction ==========
        # Build spatial and feature graphs
        a_s_processed, a_f_processed = self.module3(
            spatial_coords,
            h_rna,
            x_adt_norm
        )
        
        # ========== MODULE 4: Local Spatial Encoding ==========
        # Residual GATv2 on dual graphs
        z_rna, z_adt = self.module4(
            h_rna,
            x_adt_norm,
            a_s_processed,
            a_f_processed
        )
        
        # ========== MODULE 5: Cross-Modal Contrastive Alignment ==========
        # Align embeddings in shared latent space
        z_rna_aligned, z_adt_aligned, loss_contrastive = self.module5(
            z_rna,
            z_adt
        )
        
        # ========== MODULE 6: Adaptive Dual-Attention Fusion ==========
        # Hierarchical gating and fusion
        z_fused = self.module6(
            z_rna_aligned,
            z_adt_aligned,
            a_s_processed,
            a_f_processed
        )
        
        # ========== MODULE 7: Reconstruction & Regularization ==========
        # Decode and compute losses
        rna_recon, adt_recon, loss_recon, loss_spatial = self.module7(
            z_fused,
            x_rna_norm,
            x_adt_norm,
            a_s_processed
        )
        
        # ========== Aggregate Losses ==========
        loss_total = (
            self.lambda_contrastive * loss_contrastive +
            self.lambda_reconstruction * loss_recon +
            self.lambda_spatial * loss_spatial
        )
        
        # Compile outputs
        outputs = {
            'z_fused': z_fused,
            'rna_recon': rna_recon,
            'adt_recon': adt_recon,
            'embeddings': {
                'h_rna': h_rna,
                'z_rna': z_rna,
                'z_adt': z_adt,
                'z_rna_aligned': z_rna_aligned,
                'z_adt_aligned': z_adt_aligned,
                'z_fused': z_fused
            },
            'losses': {
                'loss_total': loss_total,
                'loss_contrastive': loss_contrastive,
                'loss_reconstruction': loss_recon,
                'loss_spatial': loss_spatial
            }
        }
        
        return outputs
    
    def train_epoch(self, dataloader, epoch: int):
        """
        Single training epoch through all data.
        
        Args:
            dataloader: PyTorch DataLoader
            epoch (int): Current epoch number
        
        Returns:
            metrics (dict): Epoch-level metrics
        """
        self.train()
        
        total_loss = 0.0
        total_loss_contrastive = 0.0
        total_loss_reconstruction = 0.0
        total_loss_spatial = 0.0
        n_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            rna = batch['rna'].to(self.device)
            adt = batch['adt'].to(self.device)
            spatial_coords = batch['spatial_coords'].to(self.device)
            adj_s = batch['adj_s'].to(self.device)
            adj_f = batch['adj_f'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.forward(rna, adt, spatial_coords, adj_s, adj_f)
            
            # Backward pass
            loss = outputs['losses']['loss_total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_loss_contrastive += outputs['losses']['loss_contrastive'].item()
            total_loss_reconstruction += outputs['losses']['loss_reconstruction'].item()
            total_loss_spatial += outputs['losses']['loss_spatial'].item()
            n_batches += 1
        
        # Compute averages
        avg_loss = total_loss / n_batches
        avg_loss_cont = total_loss_contrastive / n_batches
        avg_loss_recon = total_loss_reconstruction / n_batches
        avg_loss_spat = total_loss_spatial / n_batches
        
        # Update history
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['epoch'].append(epoch)
        self.training_history['loss_total'].append(avg_loss)
        self.training_history['loss_contrastive'].append(avg_loss_cont)
        self.training_history['loss_reconstruction'].append(avg_loss_recon)
        self.training_history['loss_spatial'].append(avg_loss_spat)
        self.training_history['learning_rate'].append(current_lr)
        
        return {
            'loss_total': avg_loss,
            'loss_contrastive': avg_loss_cont,
            'loss_reconstruction': avg_loss_recon,
            'loss_spatial': avg_loss_spat,
            'learning_rate': current_lr
        }
    
    @torch.no_grad()
    def get_embeddings(self, dataloader) -> np.ndarray:
        """
        Extract Z_Fused embeddings for all data.
        
        Args:
            dataloader: PyTorch DataLoader
        
        Returns:
            z_fused_all (np.ndarray): Shape (N, 64) - All embeddings
        """
        self.eval()
        
        embeddings_list = []
        
        for batch in dataloader:
            rna = batch['rna'].to(self.device)
            adt = batch['adt'].to(self.device)
            spatial_coords = batch['spatial_coords'].to(self.device)
            adj_s = batch['adj_s'].to(self.device)
            adj_f = batch['adj_f'].to(self.device)
            
            outputs = self.forward(rna, adt, spatial_coords, adj_s, adj_f)
            z_fused = outputs['z_fused'].detach().cpu().numpy()
            embeddings_list.append(z_fused)
        
        z_fused_all = np.vstack(embeddings_list)
        return z_fused_all
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training loss curves.
        
        Args:
            save_path (str, optional): Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = self.training_history['epoch']
        
        # Total loss
        axes[0, 0].plot(epochs, self.training_history['loss_total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(alpha=0.3)
        
        # Contrastive loss
        axes[0, 1].plot(epochs, self.training_history['loss_contrastive'], 'g-', linewidth=2)
        axes[0, 1].set_title('Contrastive Loss (Module 5)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(alpha=0.3)
        
        # Reconstruction loss
        axes[1, 0].plot(epochs, self.training_history['loss_reconstruction'], 'r-', linewidth=2)
        axes[1, 0].set_title('Reconstruction Loss (Module 7)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(alpha=0.3)
        
        # Spatial regularization loss
        axes[1, 1].plot(epochs, self.training_history['loss_spatial'], 'orange', linewidth=2)
        axes[1, 1].set_title('Spatial Regularization Loss (Module 7)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history saved to {save_path}")
        
        plt.show()
    
    def save_checkpoint(self, save_path: str):
        """
        Save model checkpoint with all states.
        
        Args:
            save_path (str): Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, save_path: str):
        """
        Load model checkpoint.
        
        Args:
            save_path (str): Path to checkpoint
        """
        checkpoint = torch.load(save_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"✅ Checkpoint loaded from {save_path}")


def create_kac_net(config: Dict, device: str = 'cpu') -> KACNet:
    """
    Factory function to create KAC-Net instance.
    
    Args:
        config (dict): Configuration dictionary
        device (str): Device placement
    
    Returns:
        model (KACNet): Initialized KAC-Net model
    """
    model = KACNet(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"KAC-Net Model Initialized")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    return model


def pipeline_summary():
    """Print complete pipeline overview."""
    summary = """
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                          KAC-NET PIPELINE SUMMARY                              ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    
    INPUT: 3,484 SPOTS × (18,085 RNA GENES + 31 ADT PROTEINS + 2 COORDINATES)
    │
    ├─ [MODULE 1] PREPROCESSING
    │   Function: Normalize RNA (CLR) and ADT (library scaling, log1p)
    │   Output: X̃_RNA (3,484×18,085), X̃_ADT (3,484×31)
    │
    ├─ [MODULE 2] KNOWLEDGE-ENRICHED ENCODING
    │   Function: spaLLM transformer for gene recovery
    │   Output: H_RNA (3,484×512)
    │
    ├─ [MODULE 3] MULTI-GRAPH CONSTRUCTION
    │   Function: Build k-NN spatial graph (k=6) + feature similarity graph
    │   Output: A_s, A_f (3,484×3,484 sparse adjacency matrices)
    │
    ├─ [MODULE 4] LOCAL SPATIAL ENCODING
    │   Function: Residual GATv2 on dual graphs
    │   Output: Z_RNA, Z_ADT (3,484×64)
    │
    ├─ [MODULE 5] CROSS-MODAL CONTRASTIVE ALIGNMENT
    │   Function: InfoNCE loss to align RNA and protein embeddings
    │   Output: Aligned Z_RNA, Z_ADT + L_contrastive
    │
    ├─ [MODULE 6] ADAPTIVE DUAL-ATTENTION FUSION
    │   Function: Hierarchical gating and fusion
    │   Output: Z_Fused (3,484×64)
    │
    ├─ [MODULE 7] RECONSTRUCTION & REGULARIZATION
    │   Function: Decode Z_Fused + compute losses
    │   Output: X̂_RNA, X̂_ADT + L_total
    │
    ├─ [TRAINING LOOP]
    │   Epochs: 50
    │   Optimizer: Adam (LR=1e-3)
    │   Losses: L_contrastive (0.5) + L_reconstruction (1.0) + L_spatial (0.3)
    │
    ├─ [MODULE 8] SPATIAL DOMAIN IDENTIFICATION
    │   Function: Leiden clustering with resolution sweep (0.2-2.0)
    │   Output: Domain labels (3,484,), UMAP coords, ARI score
    │
    └─ OUTPUT: 7 LYMPH NODE DOMAINS + VISUALIZATION + VALIDATION METRICS
    
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║ EXPECTED PERFORMANCE: ARI > 0.68 on 7-domain lymph node dataset               ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """
    print(summary)


if __name__ == '__main__':
    # Print pipeline overview
    pipeline_summary()

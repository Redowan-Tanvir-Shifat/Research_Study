"""
KAC-Net v2 — trainer.py
Trainer class managing the training pipeline:
  - Preprocessing (Module 1)
  - Knowledge extraction (Module 2)
  - Graph construction (Module 3)
  - Training loop optimization & early stopping
"""

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .preprocess import normalize_rna, normalize_adt, get_mock_embedding, fix_seed
from .graph_builder import build_all_graphs
from .model import KACNet
from .losses import compute_total_loss


class Train_KACNet:
    """
    KAC-Net End-to-End training and inference orchestrator.
    """
    def __init__(
        self,
        adata_rna,
        adata_adt,
        scgpt_model_dir=None,
        latent_dim=128,
        proj_dim=64,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=300,
        patience=20,
        weights=[1.0, 5.0, 1.0, 1.0],  # [w_recon_rna, w_recon_adt, w_align, w_laplacian]
        temperature=0.07,
        device=None,
        random_seed=42
    ):
        """
        Parameters
        ----------
        adata_rna        : AnnData  –  raw RNA AnnData object
        adata_adt        : AnnData  –  raw ADT AnnData object
        scgpt_model_dir  : str      –  path to scGPT checkpoint. If None, PCA fallback is used.
        latent_dim       : int      –  dimension of fused embedding space (default 128)
        proj_dim         : int      –  dimension of contrastive projection head (default 64)
        lr               : float    –  learning rate (default 1e-3)
        weight_decay     : float    –  optimizer weight decay
        epochs           : int      –  maximum epochs
        patience         : int      –  early stopping patience
        weights          : list     –  loss blending weights
        temperature      : float    –  InfoNCE temperature
        device           : torch.device or str  –  computing device
        random_seed      : int      –  seed for reproducibility
        """
        self.random_seed = random_seed
        fix_seed(self.random_seed)
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[KAC-Net] Using device: {self.device}")
        
        self.adata_rna = adata_rna.copy()
        self.adata_adt = adata_adt.copy()
        
        # 1. Normalization (Module 1)
        normalize_rna(self.adata_rna, inplace=True)
        normalize_adt(self.adata_adt, inplace=True)
        
        # 2. Knowledge Enrichment (Module 2)
        if scgpt_model_dir is not None:
            # Try importing scGPT
            from .preprocess import get_scgpt_embedding
            try:
                self.H_RNA = get_scgpt_embedding(self.adata_rna, model_dir=scgpt_model_dir, device=str(self.device))
            except Exception as e:
                print(f"[KAC-Net] Failed to load scGPT: {e}. Falling back to PCA.")
                self.H_RNA = get_mock_embedding(self.adata_rna)
        else:
            self.H_RNA = get_mock_embedding(self.adata_rna)
            
        # 3. Graph Construction (Module 3)
        self.graphs = build_all_graphs(self.adata_rna, self.adata_adt, self.H_RNA)
        
        # Extract features
        self.rna_features = torch.FloatTensor(self.adata_rna.X.toarray() if hasattr(self.adata_rna.X, "toarray") else self.adata_rna.X).to(self.device)
        self.adt_features = torch.FloatTensor(self.adata_adt.X.toarray() if hasattr(self.adata_adt.X, "toarray") else self.adata_adt.X).to(self.device)
        
        # Input dimensions
        self.dim_rna = self.rna_features.shape[1]
        self.dim_adt = self.adt_features.shape[1]
        
        # Move graphs to device
        self.adj_spatial = self.graphs['adj_spatial'].to(self.device)
        self.adj_feature = self.graphs['adj_feature'].to(self.device)
        self.adj_spatial_tensor = self.graphs['adj_spatial_tensor'].to(self.device)
        
        # Model config
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.weights = weights
        self.temperature = temperature
        
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def train(self):
        """
        Run the training loop with early stopping.
        """
        model = KACNet(
            dim_rna=self.dim_rna,
            dim_adt=self.dim_adt,
            latent_dim=self.latent_dim,
            proj_dim=self.proj_dim
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        best_model_state = None
        patience_counter = 0
        
        print(f"[KAC-Net] Starting training loop for {self.epochs} epochs...")
        
        for epoch in tqdm(range(1, self.epochs + 1)):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(self.rna_features, self.adt_features, self.adj_spatial, self.adj_feature)
            
            loss_dict = compute_total_loss(
                outputs,
                rna_target=self.rna_features,
                adt_target=self.adt_features,
                adj_spatial_tensor=self.adj_spatial_tensor,
                w_recon_rna=self.weights[0],
                w_recon_adt=self.weights[1],
                w_align=self.weights[2],
                w_laplacian=self.weights[3],
                temperature=self.temperature
            )
            
            loss = loss_dict['loss']
            self.loss_history.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            # Early stopping check
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_epoch = epoch
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"[KAC-Net] Early stopping triggered at epoch {epoch}. Best epoch: {self.best_epoch} (Loss: {self.best_loss:.5f})")
                break
                
        # Load best weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Inference mode
        model.eval()
        with torch.no_grad():
            final_outputs = model(self.rna_features, self.adt_features, self.adj_spatial, self.adj_feature)
            
        # Move final latent embeddings and attention weights to CPU numpy arrays
        results = {
            'emb_latent_rna': final_outputs['z_rna'].cpu().numpy(),
            'emb_latent_adt': final_outputs['z_adt'].cpu().numpy(),
            'h_fused': final_outputs['h_fused'].cpu().numpy(),
            'att_rna_weights': final_outputs['att_rna_weights'].cpu().numpy(),
            'att_adt_weights': final_outputs['att_adt_weights'].cpu().numpy(),
            'gate_weights': final_outputs['gate_weights'].cpu().numpy(),
        }
        
        print("[KAC-Net] Optimization completed successfully.")
        return results

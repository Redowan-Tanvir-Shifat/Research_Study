"""
KAC-Net Training Orchestrator

Unified training engine for the complete KAC-Net pipeline. Handles epoch loops,
loss computation, backpropagation, validation, checkpointing, and learning rate
scheduling. This module abstracts away training complexity from the model definition,
enabling clean code separation and easy experimentation with different training strategies.

Key Features:
    - Multi-loss training (contrastive + reconstruction + spatial)
    - Learning rate scheduling (cosine annealing, step decay)
    - Gradient clipping for stability
    - Validation and early stopping
    - Checkpoint management (save/load)
    - Detailed training history logging
    - GPU/CPU support with mixed precision (optional)

Usage:
    from trainer import KACNetTrainer
    from config import get_config
    
    config = get_config('lymph_node')
    trainer = KACNetTrainer(model, config, device='cuda')
    history = trainer.train(train_loader, val_loader, epochs=50)
    trainer.save_checkpoint('best_model.pt')
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KACNetTrainer:
    """
    Unified trainer for KAC-Net model with complete training pipeline.
    
    Manages:
    - Forward/backward passes with multi-loss optimization
    - Learning rate scheduling and gradient management
    - Validation and early stopping
    - Checkpoint persistence
    - Training history and visualization
    
    Attributes:
        model: KACNet model instance
        config: Configuration dictionary with training parameters
        device: torch.device (cuda or cpu)
        optimizer: Adam optimizer
        scheduler: Learning rate scheduler
        best_val_loss: Best validation loss seen so far
        patience_counter: Early stopping counter
        training_history: Dictionary tracking losses over epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = 'cuda',
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize KACNet trainer.
        
        Args:
            model: KACNet model instance (from kac_net_main.py)
            config: Configuration dictionary (from config.py) containing:
                - 'device': Device type
                - 'learning_rate': Initial learning rate (default: 1e-3)
                - 'weight_decay': L2 regularization (default: 1e-5)
                - 'scheduler_type': 'cosine' or 'step' (default: 'cosine')
                - 'max_epochs': Maximum training epochs (default: 50)
                - 'early_stopping_patience': Patience for ES (default: 10)
                - 'gradient_clip': Max gradient norm (default: 1.0)
                - 'losses': Dict with lambda_contrastive, lambda_reconstruction, lambda_spatial
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints (from config['data']['checkpoint_dir'])
        
        Example:
            >>> from config import get_config
            >>> from kac_net_main import create_kac_net
            >>> config = get_config('lymph_node')
            >>> model = create_kac_net(config, 'cuda')
            >>> trainer = KACNetTrainer(model, config, device='cuda')
        """
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir or config['data']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer configuration
        lr = config['training'].get('learning_rate', 1e-3)
        weight_decay = config['training'].get('weight_decay', 1e-5)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler_type = config['training'].get('scheduler_type', 'cosine')
        max_epochs = config['training'].get('max_epochs', 50)
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=max(1, max_epochs // 5),
                gamma=0.5
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Loss weights
        self.lambda_contrastive = config['losses'].get('lambda_contrastive', 0.5)
        self.lambda_reconstruction = config['losses'].get('lambda_reconstruction', 1.0)
        self.lambda_spatial = config['losses'].get('lambda_spatial', 0.3)
        
        # Training history
        self.training_history = defaultdict(list)
        self.current_epoch = 0
        
        logger.info(f"✓ Trainer initialized on {self.device}")
        logger.info(f"  LR: {lr}, Weight decay: {weight_decay}, Scheduler: {scheduler_type}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Execute single training epoch.
        
        Performs forward pass through all modules, computes multi-component loss,
        backpropagates, and updates model parameters.
        
        Args:
            train_loader: PyTorch DataLoader with batches of (X_RNA, X_ADT, coords)
        
        Returns:
            losses (dict): Dictionary with keys:
                - 'total': Total loss (L_total)
                - 'contrastive': Contrastive alignment loss (L_cl)
                - 'reconstruction': Reconstruction loss (L_recon)
                - 'spatial': Spatial regularization loss (L_spatial)
        
        Example:
            >>> epoch_losses = trainer.train_epoch(train_loader)
            >>> print(f"Epoch loss: {epoch_losses['total']:.4f}")
        """
        self.model.train()
        losses = {'total': 0.0, 'contrastive': 0.0, 'reconstruction': 0.0, 'spatial': 0.0}
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            X_RNA, X_ADT, coords = batch
            X_RNA = X_RNA.to(self.device)
            X_ADT = X_ADT.to(self.device)
            coords = coords.to(self.device)
            
            # Forward pass: returns (X_RNA_recon, X_ADT_recon, L_cl, L_spatial)
            try:
                X_RNA_recon, X_ADT_recon, L_cl, L_spatial = self.model(
                    X_RNA, X_ADT, coords
                )
            except Exception as e:
                logger.error(f"Forward pass failed at batch {batch_idx}: {e}")
                raise
            
            # Reconstruction loss
            L_recon = nn.functional.mse_loss(X_RNA_recon, X_RNA) + \
                      nn.functional.mse_loss(X_ADT_recon, X_ADT)
            
            # Total loss with weighted combination
            L_total = (
                self.lambda_contrastive * L_cl +
                self.lambda_reconstruction * L_recon +
                self.lambda_spatial * L_spatial
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            L_total.backward()
            
            # Gradient clipping for stability
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate losses
            batch_size = X_RNA.shape[0]
            losses['total'] += L_total.item() * batch_size
            losses['contrastive'] += L_cl.item() * batch_size
            losses['reconstruction'] += L_recon.item() * batch_size
            losses['spatial'] += L_spatial.item() * batch_size
        
        # Average over all batches
        n_samples = len(train_loader.dataset)
        for key in losses:
            losses[key] /= n_samples
        
        return losses
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Perform validation on validation set.
        
        Computes losses without gradient tracking or model updates.
        
        Args:
            val_loader: PyTorch DataLoader with validation batches
        
        Returns:
            losses (dict): Dictionary with same structure as train_epoch()
        
        Example:
            >>> val_losses = trainer.validate(val_loader)
            >>> print(f"Validation loss: {val_losses['total']:.4f}")
        """
        self.model.eval()
        losses = {'total': 0.0, 'contrastive': 0.0, 'reconstruction': 0.0, 'spatial': 0.0}
        
        for batch in val_loader:
            X_RNA, X_ADT, coords = batch
            X_RNA = X_RNA.to(self.device)
            X_ADT = X_ADT.to(self.device)
            coords = coords.to(self.device)
            
            # Forward pass
            X_RNA_recon, X_ADT_recon, L_cl, L_spatial = self.model(
                X_RNA, X_ADT, coords
            )
            
            # Reconstruction loss
            L_recon = nn.functional.mse_loss(X_RNA_recon, X_RNA) + \
                      nn.functional.mse_loss(X_ADT_recon, X_ADT)
            
            # Total loss
            L_total = (
                self.lambda_contrastive * L_cl +
                self.lambda_reconstruction * L_recon +
                self.lambda_spatial * L_spatial
            )
            
            # Accumulate
            batch_size = X_RNA.shape[0]
            losses['total'] += L_total.item() * batch_size
            losses['contrastive'] += L_cl.item() * batch_size
            losses['reconstruction'] += L_recon.item() * batch_size
            losses['spatial'] += L_spatial.item() * batch_size
        
        # Average
        n_samples = len(val_loader.dataset)
        for key in losses:
            losses[key] /= n_samples
        
        return losses
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Complete training loop with validation and early stopping.
        
        Trains model for specified epochs, validates at each epoch, saves best model,
        and implements early stopping. Returns complete training history.
        
        Args:
            train_loader: PyTorch DataLoader for training set
            val_loader: PyTorch DataLoader for validation set
            epochs: Number of epochs (uses config['training']['max_epochs'] if None)
        
        Returns:
            history (dict): Training history with keys:
                - 'train_total': List of total losses per epoch
                - 'train_contrastive': List of contrastive losses per epoch
                - 'train_reconstruction': List of reconstruction losses per epoch
                - 'train_spatial': List of spatial losses per epoch
                - 'val_total': List of validation losses per epoch
                - 'val_contrastive': List of validation contrastive losses per epoch
                - 'val_reconstruction': List of validation reconstruction losses per epoch
                - 'val_spatial': List of validation spatial losses per epoch
                - 'best_epoch': Epoch with lowest validation loss
                - 'best_val_loss': Best validation loss achieved
        
        Example:
            >>> history = trainer.train(train_loader, val_loader, epochs=50)
            >>> print(f"Best epoch: {history['best_epoch']}, Loss: {history['best_val_loss']:.4f}")
            >>> trainer.plot_training_history()
        """
        if epochs is None:
            epochs = self.config['training'].get('max_epochs', 50)
        
        best_epoch = 0
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_losses = self.train_epoch(train_loader)
            
            # Validation phase
            val_losses = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record history
            for key, value in train_losses.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_losses.items():
                self.training_history[f'val_{key}'].append(value)
            
            # Log progress (every 5 epochs or last epoch)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_losses['total']:.4f} | "
                    f"Val Loss: {val_losses['total']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Early stopping check
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                best_epoch = epoch
                
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(best loss: {self.best_val_loss:.4f})"
                    )
                    break
        
        # Record best results
        self.training_history['best_epoch'] = best_epoch
        self.training_history['best_val_loss'] = self.best_val_loss
        
        logger.info(f"✓ Training complete! Best loss: {self.best_val_loss:.4f} at epoch {best_epoch+1}")
        return dict(self.training_history)
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """
        Save model checkpoint with training state.
        
        Saves model weights, optimizer state, scheduler state, and training history.
        Allows resuming training from exact point.
        
        Args:
            filename: Checkpoint filename (saved in config['checkpoint_dir'])
        
        Example:
            >>> trainer.save_checkpoint('epoch_50_checkpoint.pt')
            >>> # Later: trainer.load_checkpoint('epoch_50_checkpoint.pt')
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': dict(self.training_history),
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✓ Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pt') -> int:
        """
        Load model checkpoint and restore training state.
        
        Restores model, optimizer, scheduler, and training history. Returns epoch
        to resume training from.
        
        Args:
            filename: Checkpoint filename to load
        
        Returns:
            epoch: Epoch number to resume from
        
        Example:
            >>> start_epoch = trainer.load_checkpoint('epoch_50_checkpoint.pt')
            >>> history = trainer.train(train_loader, val_loader, epochs=100)
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        epoch = checkpoint['epoch']
        logger.info(f"✓ Checkpoint loaded from {checkpoint_path} (epoch {epoch+1})")
        
        return epoch
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot 4-panel training history visualization.
        
        Shows total loss, contrastive loss, reconstruction loss, and spatial loss
        over epochs (train vs validation).
        
        Args:
            save_path: Path to save figure (optional)
        
        Example:
            >>> trainer.plot_training_history('training_history.png')
            >>> plt.show()
        """
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('KAC-Net Training History', fontsize=16, fontweight='bold')
        
        loss_types = ['total', 'contrastive', 'reconstruction', 'spatial']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for loss_type, (row, col) in zip(loss_types, positions):
            ax = axes[row, col]
            
            train_key = f'train_{loss_type}'
            val_key = f'val_{loss_type}'
            
            if train_key in self.training_history and val_key in self.training_history:
                epochs = range(1, len(self.training_history[train_key]) + 1)
                ax.plot(epochs, self.training_history[train_key], 'o-', label='Train', linewidth=2)
                ax.plot(epochs, self.training_history[val_key], 's-', label='Validation', linewidth=2)
                
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Loss', fontsize=11)
                ax.set_title(f'{loss_type.capitalize()} Loss', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.checkpoint_dir / save_path if not str(save_path).startswith('/') else save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Training history plot saved to {save_path}")
        
        return fig
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_training_summary(self) -> Dict:
        """
        Get summary statistics of training.
        
        Returns:
            summary (dict): Dictionary with:
                - 'total_epochs': Total epochs trained
                - 'best_epoch': Epoch with best validation loss
                - 'best_val_loss': Best validation loss
                - 'final_train_loss': Final training loss
                - 'final_val_loss': Final validation loss
        """
        n_epochs = len(self.training_history.get('train_total', []))
        
        return {
            'total_epochs': n_epochs,
            'best_epoch': self.training_history.get('best_epoch', 0),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.training_history['train_total'][-1] if n_epochs > 0 else None,
            'final_val_loss': self.training_history['val_total'][-1] if n_epochs > 0 else None,
        }


if __name__ == '__main__':
    """
    Example usage:
    
    from config import get_config
    from kac_net_main import create_kac_net
    from data_loader import create_data_loaders
    
    # Setup
    config = get_config('lymph_node')
    model = create_kac_net(config, 'cuda')
    train_loader, val_loader = create_data_loaders(config)
    
    # Train
    trainer = KACNetTrainer(model, config, device='cuda')
    history = trainer.train(train_loader, val_loader, epochs=50)
    
    # Save
    trainer.save_checkpoint('final_model.pt')
    trainer.plot_training_history('training_history.png')
    
    # Summary
    print(trainer.get_training_summary())
    """
    print("KACNetTrainer module ready. See docstrings for usage.")

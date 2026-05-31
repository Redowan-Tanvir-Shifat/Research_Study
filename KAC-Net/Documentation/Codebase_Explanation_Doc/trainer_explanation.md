# trainer.py Explanation

**File:** `trainer.py`  
**Lines:** 550  
**Purpose:** Complete training orchestrator for KAC-Net model  
**Status:** ✅ Production-ready

---

## 📌 Why Do You Need trainer.py?

### **The Problem (Without trainer.py)**

If training logic lived inside `kac_net_main.py`:
- Model definition (forward pass) + training code = **messy mixing of concerns**
- Epoch loops, optimizers, schedulers = **duplicated logic** if you want multiple training strategies
- Hard to debug - unclear what's model vs what's training logistics
- Can't reuse training code with different models
- Testing model ≠ testing training = **no separation**

### **The Solution (trainer.py)**

**Separation of Concerns:**
```
kac_net_main.py:  ← WHAT to compute (model architecture)
trainer.py:       ← HOW to optimize it (training strategy)
config.py:        ← WHEN/WHERE/WITH_WHAT parameters to train
```

**Benefits:**
✅ **Clean Architecture:** Model code stays pure, training code is separate  
✅ **Reusability:** Use same trainer with different models or datasets  
✅ **Flexibility:** Swap training strategies without touching model  
✅ **Debugging:** Easy to trace training issues independently  
✅ **Testing:** Unit test model and trainer separately  
✅ **Experimentation:** Try different optimizers, schedulers, strategies quickly  

---

## 🏗️ KACNetTrainer Class Architecture

### **Class Overview**

```python
class KACNetTrainer:
    """Complete training pipeline for KAC-Net"""
    
    def __init__(model, config, device, checkpoint_dir):
        # Initialize model, optimizer, scheduler, early stopping
        
    def train_epoch(train_loader):
        # Execute single training epoch → dict with 4 loss components
        
    def validate(val_loader):
        # Compute validation losses without gradient updates
        
    def train(train_loader, val_loader, epochs):
        # Main loop: train_epoch + validate + early_stopping + checkpointing
        
    def save_checkpoint(filename):
        # Save model + optimizer + scheduler + history (resume-ready)
        
    def load_checkpoint(filename):
        # Restore entire training state from checkpoint
        
    def plot_training_history(save_path):
        # 4-panel visualization of all loss components
        
    def get_training_summary():
        # Summary statistics (best epoch, losses, etc.)
```

---

## 🔧 Core Methods Explained

### **1. `__init__()` - Initialization**

**What it does:**
- Moves model to device
- Creates Adam optimizer with config parameters
- Sets up learning rate scheduler (cosine annealing or step decay)
- Initializes early stopping and checkpoint directories
- Records loss weights (lambda values)

**Configuration Parameters Used:**
```python
config['training']['learning_rate']           # Default: 1e-3
config['training']['weight_decay']            # Default: 1e-5 (L2 regularization)
config['training']['scheduler_type']          # 'cosine' or 'step'
config['training']['max_epochs']              # Default: 50
config['training']['early_stopping_patience'] # Default: 10 epochs
config['training']['gradient_clip']           # Default: 1.0 (max gradient norm)
config['losses']['lambda_contrastive']        # Default: 0.5
config['losses']['lambda_reconstruction']     # Default: 1.0
config['losses']['lambda_spatial']            # Default: 0.3
```

**Example:**
```python
from config import get_config
from kac_net_main import create_kac_net
from trainer import KACNetTrainer

config = get_config('lymph_node')
model = create_kac_net(config, 'cuda')

# Initialize trainer
trainer = KACNetTrainer(
    model=model,
    config=config,
    device='cuda',
    checkpoint_dir=config['data']['checkpoint_dir']
)
```

---

### **2. `train_epoch()` - Single Epoch Training**

**What it does:**
1. Set model to training mode (`model.train()`)
2. Loop through all batches in training set
3. For each batch:
   - Load batch to device
   - **Forward pass:** Pass through all 8 modules → get reconstructions + losses
   - **Compute total loss:** $L_{total} = 0.5 \cdot L_{cl} + 1.0 \cdot L_{recon} + 0.3 \cdot L_{spatial}$
   - **Backward pass:** Compute gradients
   - **Gradient clipping:** Prevent gradient explosion
   - **Update weights:** Optimizer step
4. Average losses over entire dataset
5. Return loss dictionary

**Loss Formula:**
```python
L_contrastive = output from Module 5
L_reconstruction = MSE(X_RNA_recon, X_RNA) + MSE(X_ADT_recon, X_ADT)
L_spatial = output from Module 7

L_total = 0.5 * L_contrastive + 1.0 * L_reconstruction + 0.3 * L_spatial
```

**Returned Dictionary:**
```python
{
    'total': 0.3451,           # Weighted sum of all losses
    'contrastive': 0.1234,     # Cross-modal alignment loss
    'reconstruction': 0.1876,  # Autoencoder reconstruction loss
    'spatial': 0.0341,         # Spatial smoothness regularization
}
```

**Example:**
```python
epoch_losses = trainer.train_epoch(train_loader)
print(f"Epoch Loss (total): {epoch_losses['total']:.4f}")
print(f"Contrastive: {epoch_losses['contrastive']:.4f}")
```

---

### **3. `validate()` - Validation Phase**

**What it does:**
- Same as `train_epoch()` **BUT:**
  - No gradient tracking (`@torch.no_grad()`)
  - No backward passes or weight updates
  - No optimizer steps
  - Used to assess generalization to unseen data

**Why separate validation?**
- Prevents overfitting detection
- Faster (no backprop)
- Independent metric from training
- Enables early stopping

**Example:**
```python
val_losses = trainer.validate(val_loader)
print(f"Validation Loss: {val_losses['total']:.4f}")
```

---

### **4. `train()` - Main Training Loop**

**What it does:**
This is the **complete training orchestrator**:

```
FOR each epoch (0 to max_epochs):
    1. train_epoch()           → train losses
    2. validate()              → validation losses
    3. Scheduler step()        → update learning rate
    4. Record history
    5. Check early stopping:
        IF val_loss < best:
            Save checkpoint
            Reset patience
        ELSE:
            patience++
            IF patience > patience_limit:
                BREAK (early stop)
    6. Log progress (every 5 epochs)
```

**Features:**
- ✅ **Multi-loss training** (contrastive + reconstruction + spatial)
- ✅ **Learning rate scheduling** (decreases over time)
- ✅ **Early stopping** (stops if no improvement)
- ✅ **Checkpointing** (saves best model automatically)
- ✅ **Progress logging** (prints every 5 epochs)

**Returns Training History:**
```python
{
    'train_total': [0.35, 0.32, 0.30, ...],          # 50 epochs
    'train_contrastive': [0.12, 0.11, 0.10, ...],
    'train_reconstruction': [0.19, 0.18, 0.17, ...],
    'train_spatial': [0.04, 0.03, 0.03, ...],
    'val_total': [0.34, 0.31, 0.30, ...],
    'val_contrastive': [...],
    'val_reconstruction': [...],
    'val_spatial': [...],
    'best_epoch': 25,                                # Epoch with lowest val loss
    'best_val_loss': 0.2987,                         # Lowest validation loss
}
```

**Example:**
```python
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50
)

print(f"Best epoch: {history['best_epoch']}")
print(f"Best validation loss: {history['best_val_loss']:.4f}")
```

**Output:**
```
Starting training for 50 epochs...
Epoch 5/50 | Train Loss: 0.3245 | Val Loss: 0.3178 | LR: 9.51e-04
Epoch 10/50 | Train Loss: 0.2891 | Val Loss: 0.2834 | LR: 9.06e-04
Epoch 15/50 | Train Loss: 0.2567 | Val Loss: 0.2512 | LR: 8.67e-04
...
Epoch 45/50 | Train Loss: 0.1654 | Val Loss: 0.1698 | LR: 2.14e-04
Epoch 50/50 | Train Loss: 0.1632 | Val Loss: 0.1715 | LR: 1.87e-04
✓ Training complete! Best loss: 0.1598 at epoch 28
```

---

### **5. `save_checkpoint()` - Save Training State**

**What it saves:**
```python
{
    'epoch': 27,                           # Epoch number
    'model_state_dict': {...},             # Model weights
    'optimizer_state_dict': {...},         # Optimizer state (momentum, etc.)
    'scheduler_state_dict': {...},         # Scheduler state (LR schedule)
    'best_val_loss': 0.1598,              # Best loss so far
    'training_history': {...},             # All losses from all epochs
    'config': {...},                       # Config used for training
}
```

**Why checkpoints matter:**
- **Resume training** from exact point (no recomputation)
- **Keep best model** (in case training gets worse later)
- **Reproducibility** (save config that was used)
- **Experiment tracking** (save at different checkpoints)

**Example:**
```python
# After each epoch or manually
trainer.save_checkpoint('epoch_25_best.pt')
trainer.save_checkpoint('final_model.pt')
```

---

### **6. `load_checkpoint()` - Resume Training**

**What it does:**
- Loads model weights
- Restores optimizer state (momentum, Adam second moments)
- Restores scheduler state (LR schedule position)
- Restores training history
- Returns epoch to resume from

**Resume Training Example:**
```python
# Create new trainer
trainer = KACNetTrainer(model, config, device='cuda')

# Load previous checkpoint
start_epoch = trainer.load_checkpoint('epoch_25_best.pt')
# → Returns: 25

# Continue training from epoch 26
history = trainer.train(train_loader, val_loader, epochs=100)
# Starts at epoch 26, continues until epoch 100
```

---

### **7. `plot_training_history()` - Visualization**

**What it does:**
Creates **4-panel figure** showing:
- **Panel 1:** Total loss (train vs validation)
- **Panel 2:** Contrastive loss component
- **Panel 3:** Reconstruction loss component
- **Panel 4:** Spatial regularization loss

**Useful for:**
- Detecting overfitting (diverging train/val curves)
- Choosing best epoch
- Visualizing early stopping point
- Monitoring learning rate decay effects

**Example:**
```python
trainer.plot_training_history('training_history.png')
plt.show()
```

**Output:** 4 subplots showing loss curves with train (blue) vs validation (orange)

---

### **8. `get_training_summary()` - Summary Statistics**

**Returns:**
```python
{
    'total_epochs': 45,                    # How many epochs ran
    'best_epoch': 27,                      # Epoch with best val loss
    'best_val_loss': 0.1598,              # Best validation loss
    'final_train_loss': 0.1632,           # Training loss at last epoch
    'final_val_loss': 0.1715,             # Validation loss at last epoch
}
```

**Example:**
```python
summary = trainer.get_training_summary()
print(f"✓ Training Summary:")
print(f"  Best epoch: {summary['best_epoch']}")
print(f"  Best val loss: {summary['best_val_loss']:.4f}")
print(f"  Total epochs: {summary['total_epochs']}")
```

---

## 📊 Complete Training Workflow

### **Full Example - From Start to Finish**

```python
import torch
from config import get_config
from kac_net_main import create_kac_net
from trainer import KACNetTrainer
from data_loader import create_data_loaders

# ========== SETUP ==========
# 1. Configuration
config = get_config('lymph_node')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Model
model = create_kac_net(config, device)
print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# 3. Data
train_loader, val_loader = create_data_loaders(config, batch_size=256)
print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

# ========== TRAINING ==========
# 4. Initialize trainer
trainer = KACNetTrainer(
    model=model,
    config=config,
    device=device,
    checkpoint_dir=config['data']['checkpoint_dir']
)

# 5. Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config['training']['max_epochs']
)

# ========== RESULTS ==========
# 6. Save results
trainer.save_checkpoint('final_model.pt')
trainer.plot_training_history('training_history.png')

# 7. Print summary
summary = trainer.get_training_summary()
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Best Epoch: {summary['best_epoch'] + 1}")
print(f"Best Loss: {summary['best_val_loss']:.4f}")
print(f"Total Epochs: {summary['total_epochs']}")
print(f"Final Train Loss: {summary['final_train_loss']:.4f}")
print(f"Final Val Loss: {summary['final_val_loss']:.4f}")
print("="*50)
```

---

## ⚙️ Configuration Parameters Explained

### **Training Settings** (from config.py)

```python
config['training'] = {
    'learning_rate': 1e-3,              # Initial LR. Typical: 1e-3 to 1e-4
    'weight_decay': 1e-5,               # L2 regularization. Prevents overfitting
    'batch_size': 256,                  # Samples per batch. Larger = more stable but slower
    'max_epochs': 50,                   # Maximum training epochs
    'early_stopping_patience': 10,      # Stop if val loss doesn't improve for 10 epochs
    'gradient_clip': 1.0,               # Clip gradients to prevent explosion. 0 = disabled
    'scheduler_type': 'cosine',         # 'cosine' (smooth decay) or 'step' (periodic)
}

config['losses'] = {
    'lambda_contrastive': 0.5,          # Weight for cross-modal alignment
    'lambda_reconstruction': 1.0,       # Weight for reconstruction loss
    'lambda_spatial': 0.3,              # Weight for spatial smoothness
}
```

### **Hyperparameter Tuning Guide**

| Parameter | Effect | Recommendation |
|-----------|--------|-----------------|
| `learning_rate` | Higher = faster convergence but unstable; Lower = slow but stable | Start with 1e-3, reduce if diverges |
| `weight_decay` | Higher = stronger regularization, prevents overfitting | 1e-5 to 1e-4 usually good |
| `batch_size` | Larger = faster, more stable; Smaller = noisier, better generalization | 128-512 typical for omics |
| `early_stopping_patience` | How many epochs to wait for improvement | 10-20 epochs |
| `gradient_clip` | Prevents gradient explosion | 1.0 is standard, 0.5 if unstable |
| `lambda_contrastive` | Balance between alignement and other losses | 0.3-0.7 typical |
| `lambda_reconstruction` | How much to emphasize reconstruction | Usually 1.0 (baseline) |
| `lambda_spatial` | Strength of spatial smoothness | 0.1-0.5 typical |

---

## 🐛 Troubleshooting

### **Problem 1: Training Loss Explodes (NaN/Inf)**

**Causes:**
- Learning rate too high
- Gradient clipping disabled
- Numerical instability in forward pass

**Solutions:**
```python
# Reduce learning rate
config['training']['learning_rate'] = 5e-4

# Enable gradient clipping
config['training']['gradient_clip'] = 0.5

# Use smaller batch size
config['training']['batch_size'] = 128
```

---

### **Problem 2: Validation Loss Never Improves (Plateaus)**

**Causes:**
- Learning rate too low
- Model not complex enough
- Data normalization issues

**Solutions:**
```python
# Increase learning rate
config['training']['learning_rate'] = 2e-3

# Increase patience
config['training']['early_stopping_patience'] = 20

# Increase model capacity
config['spatial_encoding']['gat_hidden'] = 128
config['dual_attention_fusion']['fusion_hidden'] = 256
```

---

### **Problem 3: Out of Memory (OOM)**

**Causes:**
- Batch size too large
- Model weights too large

**Solutions:**
```python
# Reduce batch size
config['training']['batch_size'] = 64

# Reduce model dimensions
config['spatial_encoding']['gat_hidden'] = 32
config['encoding']['encoding_dim'] = 256
```

---

### **Problem 4: Training is Slow**

**Causes:**
- Batch size too small
- Gradient computation expensive
- Data loading bottleneck

**Solutions:**
```python
# Increase batch size
config['training']['batch_size'] = 512

# Use more workers for data loading
config['num_workers'] = 8

# Use faster GPU (check torch.cuda.get_device_name())
device = 'cuda:0'  # Use specific GPU if multiple
```

---

## 🎯 Best Practices

### **1. Always Use Validation Set**
```python
# Good ✓
history = trainer.train(train_loader, val_loader, epochs=50)

# Bad ✗
history = trainer.train(train_loader, train_loader, epochs=50)  # No generalization check!
```

### **2. Save Checkpoints Frequently**
```python
# After each training run
trainer.save_checkpoint('final_model.pt')

# At specific checkpoints
if (epoch + 1) % 10 == 0:
    trainer.save_checkpoint(f'epoch_{epoch+1}.pt')
```

### **3. Monitor Learning Rate**
```python
lr = trainer.get_learning_rate()
print(f"Current LR: {lr:.2e}")  # Should decrease over time with scheduler
```

### **4. Visualize Training**
```python
# Always plot results
trainer.plot_training_history('results/training_history.png')

# Compare different runs
fig1 = trainer1.plot_training_history()
fig2 = trainer2.plot_training_history()
```

### **5. Keep Logs**
```python
# Access full history for analysis
history = trainer.training_history
best_train = min(history['train_total'])
best_val = min(history['val_total'])
print(f"Best losses - Train: {best_train:.4f}, Val: {best_val:.4f}")
```

---

## 📈 Understanding Training Curves

### **Ideal Training Curve**
```
Loss ↑
  |     ╱╲╱╲ (train - noisy)
  |    ╱  ╲╱  ╲
  |   ╱        ╲ (validation - smooth, starts diverging at overfitting point)
  |  ╱          ╲___
  | ╱_______________
  └─────────────────→ Epoch
```

**Characteristics:**
- Both losses decrease initially ✓
- Validation drops slower than training (expected) ✓
- Early stopping triggers when val plateaus ✓
- No NaN/Inf values ✓

### **Warning Signs**

| Pattern | Problem | Solution |
|---------|---------|----------|
| Training loss increases | Learning rate too high | Reduce LR |
| Validation diverges from training | Overfitting | Add regularization, increase patience |
| Both plateau immediately | Model not learning | Check data, increase LR, increase model size |
| Spiky/noisy losses | Batch size too small | Increase batch size |

---

## 🔗 Integration with Other Modules

### **Data Flow: trainer.py Context**

```
config.py
    ↓
trainer.py ← Reads all hyperparameters
    ↓
kac_net_main.py (KACNet model)
    ↓
train_epoch() → Forward pass through all 8 modules
    ↓
[Module 1-8] → Loss computation
    ↓
Backpropagation + Weight update
    ↓
validate() → Evaluation on validation set
    ↓
Early stopping + Checkpointing
    ↓
clustering.py ← Use trained Z_Fused embeddings
```

---

## ✅ Summary

**trainer.py provides:**
- ✅ Complete training orchestrator (no boilerplate code needed)
- ✅ Multi-loss optimization (contrastive + reconstruction + spatial)
- ✅ Automatic checkpointing and early stopping
- ✅ Learning rate scheduling
- ✅ Gradient clipping for stability
- ✅ Training history tracking and visualization
- ✅ Resume-from-checkpoint capability
- ✅ Clean separation from model definition (kac_net_main.py)

**Why it matters:**
- Enables reproducible, professional training workflows
- Separates concerns (model ≠ training strategy)
- Makes hyperparameter tuning systematic
- Provides debugging-friendly architecture
- Reusable for different models/datasets

---

## 📚 Related Files

- **config.py** - Provides all hyperparameters
- **kac_net_main.py** - Model being trained
- **data_loader.py** - Creates train/val loaders (needs to be implemented)
- **clustering.py** - Uses trained embeddings from trainer


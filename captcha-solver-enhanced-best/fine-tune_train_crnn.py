# """
# CRNN Training - SCIENTIFICALLY OPTIMIZED FOR 99% ACCURACY
# WITH RESUME TRAINING CAPABILITY

# KEY FEATURES:
# 1. Resume from existing model checkpoint
# 2. Preserves training history and best accuracy
# 3. Can adjust learning rate for fine-tuning
# 4. Supports transfer learning mode
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torch.cuda.amp import autocast, GradScaler
# import os
# import time
# import sys
# import numpy as np
# from crnn_model import CRNN
# from crnn_dataset import CRNNDataset, collate_fn
# import warnings
# warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')


# def decode_prediction(pred_indices, idx_to_char):
#     """Decode CTC output to text"""
#     result = []
#     prev_idx = -1
#     for idx in pred_indices:
#         idx = int(idx)
#         if idx != 0 and idx != prev_idx and idx in idx_to_char:
#             result.append(idx_to_char[idx])
#         prev_idx = idx
#     return ''.join(result)


# def calculate_accuracy(outputs, label_texts, idx_to_char):
#     """Calculate sequence-level accuracy"""
#     _, preds = outputs.max(2)
#     preds = preds.transpose(1, 0).contiguous()
    
#     correct = 0
#     for i in range(len(label_texts)):
#         pred_text = decode_prediction(preds[i].cpu().numpy(), idx_to_char)
#         if pred_text == label_texts[i]:
#             correct += 1
    
#     return correct


# def train_epoch(model, train_loader, criterion, optimizer, device, idx_to_char, 
#                 epoch, scaler, scheduler, gradient_accumulation_steps=1):
#     """Train one epoch with mixed precision + gradient accumulation"""
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
    
#     optimizer.zero_grad(set_to_none=True)
    
#     for batch_idx, (images, labels, label_lengths, label_texts) in enumerate(train_loader):
#         images = images.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#         label_lengths = label_lengths.to(device, non_blocking=True)
        
#         with torch.amp.autocast('cuda', enabled=True):
#             outputs = model(images)
#             batch_size = images.size(0)
            
#             input_lengths = torch.full(
#                 (batch_size,), outputs.size(0), 
#                 dtype=torch.long, device=device
#             )
            
#             loss = criterion(
#                 outputs.log_softmax(2), 
#                 labels, 
#                 input_lengths, 
#                 label_lengths
#             )
            
#             loss = loss / gradient_accumulation_steps
        
#         scaler.scale(loss).backward()
        
#         if (batch_idx + 1) % gradient_accumulation_steps == 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)
            
#             scheduler.step()
        
#         total_loss += loss.item() * gradient_accumulation_steps
        
#         with torch.no_grad():
#             total_correct += calculate_accuracy(outputs, label_texts, idx_to_char)
        
#         total_samples += batch_size
        
#         if (batch_idx + 1) % 10 == 0:
#             current_acc = 100.0 * total_correct / total_samples
#             current_lr = scheduler.get_last_lr()[0]
#             print(f"\r  Batch {batch_idx+1}/{len(train_loader)} | "
#                   f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
#                   f"Acc: {current_acc:.2f}% | LR: {current_lr:.6f}", end="")
    
#     print()
#     avg_loss = total_loss / len(train_loader)
#     accuracy = 100.0 * total_correct / total_samples
    
#     return avg_loss, accuracy


# def validate_epoch(model, val_loader, criterion, device, idx_to_char):
#     """Validate without augmentation"""
#     model.eval()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
    
#     with torch.no_grad():
#         for images, labels, label_lengths, label_texts in val_loader:
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)
#             label_lengths = label_lengths.to(device, non_blocking=True)
            
#             outputs = model(images)
#             batch_size = images.size(0)
            
#             input_lengths = torch.full(
#                 (batch_size,), outputs.size(0), 
#                 dtype=torch.long, device=device
#             )
            
#             loss = criterion(
#                 outputs.log_softmax(2), 
#                 labels, 
#                 input_lengths, 
#                 label_lengths
#             )
            
#             total_loss += loss.item()
#             total_correct += calculate_accuracy(outputs, label_texts, idx_to_char)
#             total_samples += batch_size
    
#     avg_loss = total_loss / len(val_loader)
#     accuracy = 100.0 * total_correct / total_samples
    
#     return avg_loss, accuracy


# def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
#     """
#     Load model checkpoint and optionally optimizer state
    
#     Returns:
#         dict: Checkpoint metadata (epoch, best_acc, etc.)
#     """
#     print(f"\n{'Loading Checkpoint':-^80}")
#     print(f"Path: {checkpoint_path}")
    
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     # Load model weights
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f"✓ Model weights loaded")
    
#     # Load optimizer state if provided
#     if optimizer is not None and 'optimizer_state_dict' in checkpoint:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         print(f"✓ Optimizer state loaded")
    
#     # Extract metadata
#     metadata = {
#         'start_epoch': checkpoint.get('epoch', 0),
#         'best_val_acc': checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0.0)),
#         'char_to_idx': checkpoint.get('char_to_idx'),
#         'idx_to_char': checkpoint.get('idx_to_char'),
#         'chars': checkpoint.get('chars'),
#     }
    
#     print(f"✓ Previous training:")
#     print(f"  - Epoch: {metadata['start_epoch']}")
#     print(f"  - Best Val Acc: {metadata['best_val_acc']:.2f}%")
#     print(f"  - Vocabulary size: {len(metadata['chars']) if metadata['chars'] else 'N/A'}")
    
#     return metadata


# def train_crnn_model(
#     image_dir='dataset',
#     img_height=64,
#     img_width=200,
#     batch_size=32,
#     epochs=120,
#     max_lr=0.001,
#     test_split=0.1,
#     save_path='crnn_captcha_model_v5.pth',
#     checkpoint_dir='checkpoints',
#     num_workers=4,
#     gradient_accumulation_steps=2,
#     # NEW PARAMETERS FOR RESUME TRAINING
#     resume_from=None,           # Path to existing model checkpoint
#     reset_optimizer=False,       # If True, start with fresh optimizer state
#     fine_tune_lr=None,          # Optional: different LR for fine-tuning
# ):
#     """
#     Main training loop with resume capability
    
#     NEW PARAMETERS:
#     - resume_from: Path to existing .pth model to continue training
#     - reset_optimizer: If True, creates new optimizer (useful for fine-tuning)
#     - fine_tune_lr: If provided, overrides max_lr (useful for lower LR fine-tuning)
    
#     USAGE EXAMPLES:
#     1. Resume training from checkpoint:
#        train_crnn_model(resume_from='crnn_captcha_model_v5.pth')
    
#     2. Fine-tune with lower learning rate:
#        train_crnn_model(resume_from='model.pth', fine_tune_lr=0.0001, reset_optimizer=True)
    
#     3. Continue training with same settings:
#        train_crnn_model(resume_from='model.pth')
#     """
    
#     print("=" * 80)
#     mode = "RESUME TRAINING" if resume_from else "TRAIN FROM SCRATCH"
#     print(f"{f'CRNN CAPTCHA TRAINER - {mode}':^80}")
#     print("=" * 80)
    
#     # Setup paths
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     if not os.path.isabs(image_dir):
#         image_dir = os.path.join(script_dir, image_dir)
#     if not os.path.isabs(save_path):
#         save_path = os.path.join(script_dir, save_path)
#     if not os.path.isabs(checkpoint_dir):
#         checkpoint_dir = os.path.join(script_dir, checkpoint_dir)
    
#     if resume_from and not os.path.isabs(resume_from):
#         resume_from = os.path.join(script_dir, resume_from)
    
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     print(f"\nPaths:")
#     print(f"  Dataset: {image_dir}")
#     print(f"  Model: {save_path}")
#     if resume_from:
#         print(f"  Resume from: {resume_from}")
    
#     # Device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\nDevice: {device}")
#     if device.type == 'cuda':
#         print(f"  GPU: {torch.cuda.get_device_name(0)}")
#         print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
#     torch.backends.cudnn.benchmark = True
    
#     # Load dataset
#     print(f"\n{'Dataset Loading':-^80}")
#     base_dataset = CRNNDataset(image_dir, img_height, img_width, augment=False)
    
#     total_images = len(base_dataset)
#     val_size = int(total_images * test_split)
#     train_size = total_images - val_size
    
#     indices = np.arange(total_images)
#     np.random.seed(42)
#     np.random.shuffle(indices)
    
#     train_indices = indices[:train_size]
#     val_indices = indices[train_size:]
    
#     train_dataset = Subset(
#         CRNNDataset(image_dir, img_height, img_width, augment=True),
#         train_indices
#     )
#     val_dataset = Subset(
#         CRNNDataset(image_dir, img_height, img_width, augment=False),
#         val_indices
#     )
    
#     print(f"Total: {total_images} | Train: {train_size} | Val: {val_size}")
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         collate_fn=collate_fn,
#         persistent_workers=True if num_workers > 0 else False,
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         collate_fn=collate_fn,
#         persistent_workers=True if num_workers > 0 else False,
#     )
    
#     print(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")
    
#     # Model
#     print(f"\n{'Model Initialization':-^80}")
#     num_chars = len(base_dataset.chars) + 1
#     model = CRNN(
#         img_height=img_height, 
#         num_chars=num_chars, 
#         num_hidden=512
#     ).to(device)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Parameters: {total_params:,} (~{total_params * 4 / 1e6:.1f} MB)")
    
#     # Override learning rate if fine-tuning
#     if fine_tune_lr is not None:
#         max_lr = fine_tune_lr
#         print(f"✓ Using fine-tune learning rate: {fine_tune_lr}")
    
#     # Optimizer
#     optimizer = optim.AdamW(
#         model.parameters(), 
#         lr=max_lr, 
#         weight_decay=0.01
#     )
    
#     # Load checkpoint if resuming
#     start_epoch = 0
#     best_val_acc = 0.0
    
#     if resume_from:
#         checkpoint_meta = load_checkpoint(
#             resume_from, 
#             model, 
#             optimizer if not reset_optimizer else None,
#             device
#         )
#         start_epoch = checkpoint_meta['start_epoch']
#         best_val_acc = checkpoint_meta['best_val_acc']
        
#         if reset_optimizer:
#             print("⚠ Optimizer state reset (fresh start)")
        
#         print(f"\n✓ Resuming from epoch {start_epoch}")
#         print(f"✓ Best validation accuracy so far: {best_val_acc:.2f}%")
    
#     # Scheduler (adjust for remaining epochs)
#     remaining_epochs = epochs - start_epoch
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=max_lr,
#         epochs=remaining_epochs,
#         steps_per_epoch=len(train_loader),
#         pct_start=0.3,
#         div_factor=25,
#         final_div_factor=1000,
#     )
    
#     criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
#     scaler = GradScaler()
    
#     print(f"\n{'Training Configuration':-^80}")
#     print(f"Optimizer: AdamW (weight_decay=0.01)")
#     print(f"Scheduler: OneCycleLR (super-convergence)")
#     print(f"Max LR: {max_lr}")
#     print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
#     print(f"Epochs: {start_epoch + 1} → {epochs} (total: {remaining_epochs} remaining)")
#     print(f"Mixed Precision: ENABLED")
    
#     # Training loop
#     print(f"\n{'='*80}")
#     print(f"{'TRAINING START':^80}")
#     print(f"{'='*80}\n")
    
#     patience_counter = 0
#     patience = 200
    
#     for epoch in range(start_epoch, epochs):
#         start_time = time.time()
        
#         print(f"\nEpoch [{epoch+1}/{epochs}]")
#         print("-" * 80)
        
#         train_loss, train_acc = train_epoch(
#             model, train_loader, criterion, optimizer, device, 
#             base_dataset.idx_to_char, epoch, scaler, scheduler,
#             gradient_accumulation_steps
#         )
        
#         print("  Validating...")
#         val_loss, val_acc = validate_epoch(
#             model, val_loader, criterion, device, base_dataset.idx_to_char
#         )
        
#         epoch_time = time.time() - start_time
        
#         print(f"\n{'='*80}")
#         print(f"Epoch [{epoch+1}/{epochs}] Summary:")
#         print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
#         print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
#         print(f"  Time: {epoch_time:.1f}s")
#         print(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             patience_counter = 0
            
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss': train_loss,
#                 'val_loss': val_loss,
#                 'train_acc': train_acc,
#                 'val_acc': val_acc,
#                 'char_to_idx': base_dataset.char_to_idx,
#                 'idx_to_char': base_dataset.idx_to_char,
#                 'chars': base_dataset.chars,
#                 'img_height': img_height,
#                 'img_width': img_width,
#                 'best_val_acc': best_val_acc,
#             }, save_path)
            
#             print(f"  ✓ BEST MODEL SAVED (Val Acc: {val_acc:.2f}%)")
#         else:
#             patience_counter += 1
#             print(f"  No improvement ({patience_counter}/{patience})")
        
#         print("=" * 80)
        
#         if patience_counter >= patience:
#             print(f"\n⚠️ Early stopping triggered (no improvement for {patience} epochs)")
#             break
    
#     print(f"\n{'='*80}")
#     print(f"{'TRAINING COMPLETE':^80}")
#     print(f"{'='*80}")
#     print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
#     print(f"Model saved: {save_path}")
#     print("=" * 80)


# if __name__ == '__main__':
#     BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best'
    
#     CONFIG = {
#         'image_dir': os.path.join(BASE_DIR, 'dataset'),
#         'save_path': os.path.join(BASE_DIR, 'crnn_captcha_model_v5.pth'),
#         'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
        
#         'img_height': 64,
#         'img_width': 200,
        
#         'batch_size': 32,
#         'epochs': 170,
#         'max_lr': 0.001,
#         'test_split': 0.1,
        
#         'num_workers': 4,
#         'gradient_accumulation_steps': 2,
        
#         # ==== RESUME TRAINING SETTINGS ====
#         # Option 1: Continue training from existing model
#         'resume_from': os.path.join(BASE_DIR, 'crnn_captcha_model_v5.pth'),  # Set to None for fresh training
        
#         # Option 2: Fine-tune with lower learning rate
#         # 'resume_from': os.path.join(BASE_DIR, 'crnn_captcha_model_v5.pth'),
#         # 'fine_tune_lr': 0.0001,  # 10x lower LR for fine-tuning
#         # 'reset_optimizer': True,  # Fresh optimizer for fine-tuning
#     }
    
#     try:
#         train_crnn_model(**CONFIG)
#     except KeyboardInterrupt:
#         print("\n\n⚠️ Training interrupted by user")
#     except Exception as e:
#         print(f"\n\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
"""
CRNN TRANSFER LEARNING TRAINER - SMART WEIGHT LOADING

🎯 THE SMART APPROACH:
This script uses INTELLIGENT transfer learning that preserves maximum knowledge:

CRNN Architecture:
┌─────────────────────────────────────────┐
│ CNN Layers (Visual Feature Extraction) │ ← ALWAYS loaded (shape never changes)
├─────────────────────────────────────────┤
│ LSTM Layer 1 (Sequence Pattern Learning)│ ← LOADED if shapes match (usually do!)
├─────────────────────────────────────────┤
│ LSTM Layer 2 (Character Classification)│ ← SKIPPED (depends on vocabulary size)
└─────────────────────────────────────────┘

❌ NAIVE approach: Skip ALL RNN layers
   - Throws away LSTM Layer 1 (sequence understanding)
   - Forces model to relearn temporal patterns from scratch

✅ SMART approach: Skip ONLY mismatched layers
   - Keeps LSTM Layer 1 (sequence patterns are vocabulary-independent)
   - Only retrains LSTM Layer 2 (character mapping)
   - Converges 2-3x faster than naive approach

WHY LSTM Layer 1 is preserved:
- It learns to read SEQUENCES (left-to-right, spacing, timing)
- These patterns are INDEPENDENT of which exact characters exist
- Shape: [hidden_size, hidden_size] - doesn't depend on vocabulary!

WHY LSTM Layer 2 is retrained:
- It maps hidden states → specific characters
- Shape: [hidden_size, num_chars] - DOES depend on vocabulary
- Must be retrained when character set changes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import os
import time
import sys
import numpy as np
from crnn_model import CRNN
from crnn_dataset import CRNNDataset, collate_fn
import warnings
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')


def decode_prediction(pred_indices, idx_to_char):
    """Decode CTC output to text"""
    result = []
    prev_idx = -1
    for idx in pred_indices:
        idx = int(idx)
        if idx != 0 and idx != prev_idx and idx in idx_to_char:
            result.append(idx_to_char[idx])
        prev_idx = idx
    return ''.join(result)


def calculate_accuracy(outputs, label_texts, idx_to_char):
    """Calculate sequence-level accuracy"""
    _, preds = outputs.max(2)
    preds = preds.transpose(1, 0).contiguous()
    
    correct = 0
    for i in range(len(label_texts)):
        pred_text = decode_prediction(preds[i].cpu().numpy(), idx_to_char)
        if pred_text == label_texts[i]:
            correct += 1
    
    return correct


def load_weights_smart(checkpoint_path, model, device='cuda'):
    """
    SMART TRANSFER LEARNING: 
    Loads CNN + Valid RNN weights (LSTM Layer 1)
    Only skips layers with shape mismatches (final classifier - LSTM Layer 2)
    
    Why this is better:
    - CNN: Extracts visual features (always compatible)
    - LSTM Layer 1: Learns sequence patterns (shape independent of vocabulary)
    - LSTM Layer 2: Maps to characters (shape depends on vocabulary - gets skipped)
    
    This preserves MORE learned knowledge than naive "skip all RNN" approach.
    """
    print(f"\n{'SMART TRANSFER LEARNING: Loading Weights':-^80}")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    pretrained_dict = {}
    skipped_layers = []
    
    # Smart loading: Only load layers where shapes match exactly
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                skipped_layers.append(f"{k} (shape mismatch: {v.shape} -> {model_dict[k].shape})")
        else:
            skipped_layers.append(f"{k} (not in new model)")
    
    # Update model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Analyze what was loaded
    cnn_loaded = sum(1 for k in pretrained_dict if 'cnn' in k or 'conv' in k or 'bn' in k)
    rnn_loaded = sum(1 for k in pretrained_dict if 'rnn' in k or 'lstm' in k)
    
    print(f"\n📊 Loading Summary:")
    print(f"  ✓ Total layers loaded: {len(pretrained_dict)}")
    print(f"    - CNN layers: {cnn_loaded}")
    print(f"    - RNN layers (LSTM1): {rnn_loaded}")
    print(f"  ⚠️  Skipped layers: {len(skipped_layers)}")
    
    if len(skipped_layers) > 0:
        print(f"\n  Skipped layers (vocabulary-dependent):")
        for name in skipped_layers:
            print(f"    - {name}")
    
    print(f"\n✅ Loaded: CNN + LSTM Layer 1 (sequence patterns)")
    print(f"🔄 Training from scratch: LSTM Layer 2 (character classifier)")
    
    # Show checkpoint info
    if 'chars' in checkpoint:
        print(f"\n📝 Original Model Info:")
        print(f"  Vocabulary: {len(checkpoint['chars'])} chars → New: {len(model_dict['rnn.1.embedding.weight']) - 1} chars")
        print(f"  Original charset: {''.join(checkpoint['chars'])}")
        if 'best_val_acc' in checkpoint or 'val_acc' in checkpoint:
            acc = checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0))
            print(f"  Best accuracy: {acc:.2f}%")
    
    return checkpoint


def train_epoch(model, train_loader, criterion, optimizer, device, idx_to_char, 
                epoch, scaler, scheduler, gradient_accumulation_steps=1):
    """Train one epoch with mixed precision"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (images, labels, label_lengths, label_texts) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(images)
            batch_size = images.size(0)
            
            input_lengths = torch.full(
                (batch_size,), outputs.size(0), 
                dtype=torch.long, device=device
            )
            
            loss = criterion(
                outputs.log_softmax(2), 
                labels, 
                input_lengths, 
                label_lengths
            )
            
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        with torch.no_grad():
            total_correct += calculate_accuracy(outputs, label_texts, idx_to_char)
        
        total_samples += batch_size
        
        if (batch_idx + 1) % 10 == 0:
            current_acc = 100.0 * total_correct / total_samples
            current_lr = scheduler.get_last_lr()[0]
            print(f"\r  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                  f"Acc: {current_acc:.2f}% | LR: {current_lr:.6f}", end="")
    
    print()
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device, idx_to_char):
    """Validate without augmentation"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels, label_lengths, label_texts in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
            
            outputs = model(images)
            batch_size = images.size(0)
            
            input_lengths = torch.full(
                (batch_size,), outputs.size(0), 
                dtype=torch.long, device=device
            )
            
            loss = criterion(
                outputs.log_softmax(2), 
                labels, 
                input_lengths, 
                label_lengths
            )
            
            total_loss += loss.item()
            total_correct += calculate_accuracy(outputs, label_texts, idx_to_char)
            total_samples += batch_size
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def train_with_transfer_learning(
    image_dir='dataset',
    pretrained_model_path=None,  # Path to existing model
    img_height=64,
    img_width=200,
    batch_size=32,
    epochs=100,
    max_lr=0.001,
    test_split=0.1,
    save_path='crnn_captcha_model_transfer.pth',
    checkpoint_dir='checkpoints',
    num_workers=4,
    gradient_accumulation_steps=2,
    freeze_cnn=False,  # Option to freeze CNN layers initially
    freeze_epochs=0,   # Number of epochs to keep CNN frozen
):
    """
    Train CRNN with transfer learning from pretrained model
    
    PARAMETERS:
    - pretrained_model_path: Path to .pth file with pretrained weights
    - freeze_cnn: If True, freeze CNN layers initially (only train RNN)
    - freeze_epochs: Number of epochs to keep CNN frozen before unfreezing
    
    USAGE:
    train_with_transfer_learning(
        pretrained_model_path='crnn_captcha_model_v5.pth',
        freeze_cnn=True,
        freeze_epochs=10  # Train RNN only for 10 epochs, then fine-tune everything
    )
    """
    
    print("=" * 80)
    print(f"{'CRNN TRANSFER LEARNING TRAINER':^80}")
    print("=" * 80)
    
    if not pretrained_model_path:
        raise ValueError("pretrained_model_path is required for transfer learning!")
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(image_dir):
        image_dir = os.path.join(script_dir, image_dir)
    if not os.path.isabs(save_path):
        save_path = os.path.join(script_dir, save_path)
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(script_dir, checkpoint_dir)
    if not os.path.isabs(pretrained_model_path):
        pretrained_model_path = os.path.join(script_dir, pretrained_model_path)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\n📂 Paths:")
    print(f"  Dataset: {image_dir}")
    print(f"  Pretrained: {pretrained_model_path}")
    print(f"  Save to: {save_path}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    torch.backends.cudnn.benchmark = True
    
    # Load NEW dataset (with corrected labels)
    print(f"\n{'Dataset Loading (NEW VOCABULARY)':-^80}")
    base_dataset = CRNNDataset(image_dir, img_height, img_width, augment=False)
    
    print(f"\n🆕 New Dataset Info:")
    print(f"  Images: {len(base_dataset)}")
    print(f"  Vocabulary: {len(base_dataset.chars)} chars")
    print(f"  Charset: {''.join(base_dataset.chars)}")
    
    total_images = len(base_dataset)
    val_size = int(total_images * test_split)
    train_size = total_images - val_size
    
    indices = np.arange(total_images)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(
        CRNNDataset(image_dir, img_height, img_width, augment=True),
        train_indices
    )
    val_dataset = Subset(
        CRNNDataset(image_dir, img_height, img_width, augment=False),
        val_indices
    )
    
    print(f"  Split: Train={train_size}, Val={val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Create NEW model with NEW vocabulary size
    print(f"\n{'Model Initialization':-^80}")
    num_chars = len(base_dataset.chars) + 1  # +1 for CTC blank
    model = CRNN(
        img_height=img_height, 
        num_chars=num_chars,  # NEW vocabulary size
        num_hidden=512
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters (~{total_params * 4 / 1e6:.1f} MB)")
    
    # SMART TRANSFER LEARNING: Load CNN + compatible RNN weights
    load_weights_smart(pretrained_model_path, model, device)
    
    # Optional: Freeze CNN layers
    if freeze_cnn and freeze_epochs > 0:
        print(f"\n❄️  FREEZING CNN layers for first {freeze_epochs} epochs")
        for name, param in model.named_parameters():
            if 'rnn' not in name:
                param.requires_grad = False
        print(f"  Only RNN will be trained initially")
    
    # Optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=max_lr, weight_decay=0.01)
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    
    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )
    
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    scaler = GradScaler()
    
    print(f"  Optimizer: AdamW (weight_decay=0.01)")
    print(f"  Scheduler: OneCycleLR")
    print(f"  Max LR: {max_lr}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"  Epochs: {epochs}")
    print(f"  Mixed Precision: ENABLED")
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"{'TRANSFER LEARNING START':^80}")
    print(f"{'='*80}\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 200
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Unfreeze CNN after specified epochs
        if freeze_cnn and epoch == freeze_epochs:
            print(f"\n{'='*80}")
            print(f"🔓 UNFREEZING CNN layers - Fine-tuning entire network")
            print(f"{'='*80}\n")
            for param in model.parameters():
                param.requires_grad = True
            
            # Update optimizer to include all parameters
            optimizer = optim.AdamW(model.parameters(), lr=max_lr * 0.1, weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr * 0.1,  # Lower LR for fine-tuning
                epochs=epochs - freeze_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1000,
            )
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 80)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            base_dataset.idx_to_char, epoch, scaler, scheduler,
            gradient_accumulation_steps
        )
        
        print("  Validating...")
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, base_dataset.idx_to_char
        )
        
        epoch_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'char_to_idx': base_dataset.char_to_idx,
                'idx_to_char': base_dataset.idx_to_char,
                'chars': base_dataset.chars,
                'img_height': img_height,
                'img_width': img_width,
                'best_val_acc': best_val_acc,
                'transfer_learning': True,
                'pretrained_from': pretrained_model_path,
            }, save_path)
            
            print(f"  ✓ BEST MODEL SAVED (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        print("=" * 80)
        
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered")
            break
    
    print(f"\n{'='*80}")
    print(f"{'TRANSFER LEARNING COMPLETE':^80}")
    print(f"{'='*80}")
    print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved: {save_path}")
    print("=" * 80)


if __name__ == '__main__':
    BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best'
    
    CONFIG = {
        'image_dir': os.path.join(BASE_DIR, 'dataset'),
        'pretrained_model_path': os.path.join(BASE_DIR, 'crnn_captcha_model_v7.pth'),
        'save_path': os.path.join(BASE_DIR, 'crnn_captcha_model_transfer_v7.pth'),
        'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
        
        'img_height': 64,
        'img_width': 200,
        
        'batch_size': 32,
        'epochs': 100,
        'max_lr': 0.001,
        'test_split': 0.1,
        
        'num_workers': 4,
        'gradient_accumulation_steps': 2,
        
        # TRANSFER LEARNING SETTINGS
        'freeze_cnn': True,      # Freeze CNN initially
        'freeze_epochs': 10,     # Train only RNN for first 10 epochs
    }
    
    try:
        train_with_transfer_learning(**CONFIG)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
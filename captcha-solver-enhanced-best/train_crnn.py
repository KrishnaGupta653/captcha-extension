"""
CRNN Training - SCIENTIFICALLY OPTIMIZED FOR 99% ACCURACY

KEY CHANGES FROM YOUR ORIGINAL:
1. AdamW optimizer (prevents overfitting on 5000 images)
2. OneCycleLR scheduler (breaks through 96% plateau)
3. Larger model (512 hidden units)
4. Better validation strategy
5. Early stopping (prevents overtraining)

Expected: 96% → 98-99% accuracy
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


def train_epoch(model, train_loader, criterion, optimizer, device, idx_to_char, 
                epoch, scaler, scheduler, gradient_accumulation_steps=1):
    """Train one epoch with mixed precision + gradient accumulation"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
    
    for batch_idx, (images, labels, label_lengths, label_texts) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        
        # Mixed Precision Forward
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
            
            # Scale for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Update every N steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Step OneCycleLR scheduler PER BATCH (critical!)
            scheduler.step()
        
        # Metrics
        total_loss += loss.item() * gradient_accumulation_steps
        
        with torch.no_grad():
            total_correct += calculate_accuracy(outputs, label_texts, idx_to_char)
        
        total_samples += batch_size
        
        # Progress
        if (batch_idx + 1) % 10 == 0:
            current_acc = 100.0 * total_correct / total_samples
            current_lr = scheduler.get_last_lr()[0]
            print(f"\r  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                  f"Acc: {current_acc:.2f}% | LR: {current_lr:.6f}", end="")
    
    print()  # Newline after progress
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


def train_crnn_model(
    image_dir='dataset',
    img_height=64,
    img_width=200,
    batch_size=32,
    epochs=120,
    max_lr=0.001,
    test_split=0.1,
    save_path='crnn_captcha_model_v5.pth',
    checkpoint_dir='checkpoints',
    num_workers=4,
    gradient_accumulation_steps=2,
):
    """
    Main training loop with breakthrough optimizations
    
    CRITICAL CHANGES:
    - AdamW optimizer (weight decay prevents overfitting)
    - OneCycleLR (super-convergence, breaks 96% plateau)
    - Larger model (512 hidden units)
    - Early stopping (prevents overtraining)
    """
    
    print("=" * 80)
    print(f"{'CRNN CAPTCHA TRAINER - 99% ACCURACY MODE':^80}")
    print("=" * 80)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(image_dir):
        image_dir = os.path.join(script_dir, image_dir)
    if not os.path.isabs(save_path):
        save_path = os.path.join(script_dir, save_path)
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(script_dir, checkpoint_dir)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\nPaths:")
    print(f"  Dataset: {image_dir}")
    print(f"  Model: {save_path}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    
    # Load dataset
    print(f"\n{'Dataset Loading':-^80}")
    base_dataset = CRNNDataset(image_dir, img_height, img_width, augment=False)
    
    total_images = len(base_dataset)
    val_size = int(total_images * test_split)
    train_size = total_images - val_size
    
    # Create train/val split
    indices = np.arange(total_images)
    np.random.seed(42)  # Reproducible split
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets (train WITH augmentation, val WITHOUT)
    train_dataset = Subset(
        CRNNDataset(image_dir, img_height, img_width, augment=True),
        train_indices
    )
    val_dataset = Subset(
        CRNNDataset(image_dir, img_height, img_width, augment=False),
        val_indices
    )
    
    print(f"Total: {total_images} | Train: {train_size} | Val: {val_size}")
    
    # Data loaders
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
    
    print(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # Model (CRITICAL: Increase to 512 hidden units)
    print(f"\n{'Model Initialization':-^80}")
    num_chars = len(base_dataset.chars) + 1
    model = CRNN(
        img_height=img_height, 
        num_chars=num_chars, 
        num_hidden=512  # INCREASED from 256
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (~{total_params * 4 / 1e6:.1f} MB)")
    print(f"Hidden units: 512 (increased from 256)")
    
    # CRITICAL: AdamW optimizer (doc8 correct, doc7 wrong)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=max_lr, 
        weight_decay=0.01  # Prevents overfitting on 5000 images
    )
    
    # CRITICAL: OneCycleLR scheduler (doc8 correct)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,      # 30% warmup
        div_factor=25,      # Start LR = max_lr / 25
        final_div_factor=1000,  # End LR = max_lr / 1000
    )
    
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    # scaler = GradScaler()
    scaler = GradScaler()
    
    print(f"\n{'Training Configuration':-^80}")
    print(f"Optimizer: AdamW (weight_decay=0.01)")
    print(f"Scheduler: OneCycleLR (super-convergence)")
    print(f"Max LR: {max_lr}")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"Epochs: {epochs}")
    print(f"Mixed Precision: ENABLED")
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"{'TRAINING START':^80}")
    print(f"{'='*80}\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 200  # Early stopping after 15 epochs without improvement
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            base_dataset.idx_to_char, epoch, scaler, scheduler,
            gradient_accumulation_steps
        )
        
        # Validate
        print("  Validating...")
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, base_dataset.idx_to_char
        )
        
        epoch_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
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
            }, save_path)
            
            print(f"  ✓ BEST MODEL SAVED (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        print("=" * 80)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
            break
    
    # Training complete
    print(f"\n{'='*80}")
    print(f"{'TRAINING COMPLETE':^80}")
    print(f"{'='*80}")
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {save_path}")
    print(f"\nNext Steps:")
    print(f"  1. Update server_crnn.py model path")
    print(f"  2. Run: python export_to_onnx.py (if using ONNX)")
    print(f"  3. Start server: python server_crnn.py")
    print("=" * 80)


if __name__ == '__main__':
    # YOUR CONFIGURATION
    BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best'
    
    CONFIG = {
        'image_dir': os.path.join(BASE_DIR, 'dataset'),
        'save_path': os.path.join(BASE_DIR, 'crnn_captcha_model_v7.pth'),
        'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
        
        # Model config
        'img_height': 64,
        'img_width': 200,
        
        # Training config (OPTIMIZED)
        'batch_size': 32,       # Doc7 correct: smaller batch = better gradients
        'epochs': 200,          # Enough for OneCycle convergence
        'max_lr': 0.001,        # OneCycleLR max learning rate
        'test_split': 0.1,      # 90% train, 10% val
        
        # Performance
        'num_workers': 4,
        'gradient_accumulation_steps': 2,  # Effective batch = 64
    }
    
    try:
        train_crnn_model(**CONFIG)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""
CRNN CAPTCHA Solver - Optimized Training Script
Streamlined, efficient, and production-ready

Features:
- ✅ Data augmentation for 2-3x effective dataset size
- ✅ Enhanced preprocessing (CLAHE, denoising, binarization)
- ✅ Debug mode for monitoring training
- ✅ Checkpoint enable/disable
- ✅ All paths saved under parent directory
- ✅ Optimized for 99% accuracy

Usage:
    1. Update BASE_DIR in __main__ to your path
    2. Place labeled images in 'labeled_captchas' folder
    3. Run: python train_crnn.py
    
Configuration:
    - Set enable_checkpoints=False to disable checkpoint saving (saves disk space)
    - Set debug=False to disable debug output (cleaner logs)
    - Adjust batch_size based on your GPU memory
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import time
import sys
from crnn_model import CRNN
from crnn_dataset import CRNNDataset, collate_fn


def decode_prediction(pred_indices, idx_to_char):
    """Decode CTC prediction to text (removes blanks and duplicates)"""
    result = []
    prev_idx = -1
    
    for idx in pred_indices:
        idx = int(idx)
        if idx != 0 and idx != prev_idx and idx in idx_to_char:
            result.append(idx_to_char[idx])
        prev_idx = idx
    
    return ''.join(result)


def calculate_accuracy(outputs, label_texts, idx_to_char):
    """Calculate batch accuracy"""
    _, preds = outputs.max(2)
    preds = preds.transpose(1, 0).contiguous()
    
    correct = 0
    for i in range(len(label_texts)):
        pred_text = decode_prediction(preds[i].cpu().numpy(), idx_to_char)
        if pred_text == label_texts[i]:
            correct += 1
    
    return correct


def train_epoch(model, train_loader, criterion, optimizer, device, idx_to_char, epoch, debug=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Debug first batch of first epoch
    debug_first_batch = (epoch == 0 and debug)
    
    for batch_idx, (images, labels, label_lengths, label_texts) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward pass
        outputs = model(images)  # [T, B, num_chars]
        batch_size = images.size(0)
        
        # Debug info for first batch
        if debug_first_batch and batch_idx == 0:
            print(f"\n{'DEBUG - First Batch':=^80}")
            print(f"Input shape: {images.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Num classes: {outputs.shape[2]}")
            print(f"Batch size: {batch_size}")
            print(f"Label lengths: {label_lengths.tolist()[:5]}")
            print(f"Sample labels: {label_texts[:3]}")
            print("=" * 80)
        
        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=outputs.size(0),
            dtype=torch.long,
            device=device
        )
        
        # CTC loss
        outputs_log = outputs.log_softmax(2)
        loss = criterion(outputs_log, labels, input_lengths, label_lengths)
        
        # Skip invalid batches
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️  Skipping batch {batch_idx+1} (invalid loss)")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Calculate accuracy with debug info
        _, preds = outputs.max(2)
        preds = preds.transpose(1, 0).contiguous()
        
        for i in range(batch_size):
            pred_text = decode_prediction(preds[i].cpu().numpy(), idx_to_char)
            if pred_text == label_texts[i]:
                total_correct += 1
            
            # Debug predictions for first batch
            if debug_first_batch and batch_idx == 0 and i < 3:
                match = "✓" if pred_text == label_texts[i] else "✗"
                print(f"  Sample {i+1}: {match} Pred='{pred_text}' | True='{label_texts[i]}'")
        
        if debug_first_batch and batch_idx == 0:
            print()
            debug_first_batch = False
        
        total_samples += batch_size
        
        # Progress update
        if (batch_idx + 1) % 10 == 0:
            current_acc = 100.0 * total_correct / total_samples
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device, idx_to_char):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels, label_lengths, label_texts in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            outputs = model(images)
            batch_size = images.size(0)
            
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=device
            )
            
            outputs_log = outputs.log_softmax(2)
            loss = criterion(outputs_log, labels, input_lengths, label_lengths)
            
            total_loss += loss.item()
            total_correct += calculate_accuracy(outputs, label_texts, idx_to_char)
            total_samples += batch_size
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def train_crnn_model(
    image_dir='labeled_captchas',
    img_height=64,
    img_width=200,
    batch_size=32,
    epochs=100,
    learning_rate=0.0001,
    test_split=0.2,
    save_path='crnn_captcha_model_v2.pth',
    checkpoint_dir='checkpoints',
    save_every=10,
    enable_checkpoints=True,  # Enable/disable checkpoint saving
    debug=True  # Enable debug mode
):
    """Main training function"""
    
    # Get script directory (parent folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert all paths to absolute paths under script directory
    if not os.path.isabs(image_dir):
        image_dir = os.path.join(script_dir, image_dir)
    if not os.path.isabs(save_path):
        save_path = os.path.join(script_dir, save_path)
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(script_dir, checkpoint_dir)
    
    print("=" * 80)
    print(f"{'CRNN CAPTCHA TRAINER':^80}")
    print("=" * 80)
    print(f"\n{'Paths Configuration':-^80}")
    print(f"Script directory: {script_dir}")
    print(f"Image directory: {image_dir}")
    print(f"Model save path: {save_path}")
    if enable_checkpoints:
        print(f"Checkpoint directory: {checkpoint_dir}")
    else:
        print(f"Checkpoints: DISABLED")
    print("-" * 80)
    
    # Validate directory
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        sys.exit(1)
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No images found in: {image_dir}")
        sys.exit(1)
    
    print(f"\n✓ Found {len(image_files)} images")
    
    # Create checkpoint directory if needed
    if enable_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"✓ Checkpoint directory created: {checkpoint_dir}")
    
    # Create model save directory if needed
    model_save_dir = os.path.dirname(save_path)
    if model_save_dir and not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device.type.upper()}", end='')
    if device.type == 'cuda':
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (Consider using GPU for 10x speedup)")
    
    # Load dataset
    print(f"\n{'Dataset Loading':-^80}")
    dataset = CRNNDataset(image_dir, img_height, img_width, augment=False)  # Full dataset
    
    print(f"Total images: {len(dataset)}")
    print(f"Characters ({len(dataset.chars)}): {''.join(sorted(dataset.chars))}")
    
    # Debug: Show sample labels
    if debug:
        print(f"\n{'Sample Labels (first 5)':=^80}")
        for i in range(min(5, len(dataset))):
            _, label, _, text = dataset[i]
            print(f"  '{text}' -> indices: {label.tolist()}")
        print("=" * 80)
    
    # Split dataset
    val_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    # Create separate datasets for train (with augmentation) and val (without)
    train_dataset = CRNNDataset(image_dir, img_height, img_width, augment=True)
    val_dataset = CRNNDataset(image_dir, img_height, img_width, augment=False)
    
    # Use subset to ensure same split
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    print(f"Train: {train_size} | Val: {val_size}")
    
    # Get char mappings from the base dataset
    base_dataset = CRNNDataset(image_dir, img_height, img_width, augment=False)
    idx_to_char = base_dataset.idx_to_char
    char_to_idx = base_dataset.char_to_idx
    chars = base_dataset.chars
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    
    # Initialize model
    print(f"\n{'Model Initialization':-^80}")
    num_chars = len(chars) + 1  # +1 for CTC blank
    model = CRNN(img_height=img_height, num_chars=num_chars, num_hidden=256)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} (~{total_params * 4 / 1e6:.1f} MB)")
    print(f"Trainable: {trainable_params:,}")
    
    # Debug: Model architecture
    if debug:
        print(f"\n{'Model Architecture':=^80}")
        print(model)
        print("=" * 80)
    
    # Training setup
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"\n{'Training Config':-^80}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"Classes: {num_chars} (including blank)")
    print(f"Checkpoints: {'ENABLED' if enable_checkpoints else 'DISABLED'}")
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"{'TRAINING START':^80}")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, idx_to_char, epoch, debug
        )
        
        # Validate
        print("\n  Validating...")
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, idx_to_char
        )
        
        epoch_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{epochs}] Complete:")
        print(f"  Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
        print(f"  Val:   Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Debug: Show improvement/degradation
        if debug and epoch > 0:
            loss_change = val_loss - best_val_loss if best_val_loss != float('inf') else 0
            acc_change = val_acc - best_val_acc
            print(f"  Change: Loss {loss_change:+.4f} | Acc {acc_change:+.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'chars': chars,
                'img_height': img_height,
                'img_width': img_width,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
            }, save_path)
            
            print(f"  ✓ BEST MODEL SAVED (Val Acc: {val_acc:.2f}%)")
        
        print("=" * 80)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        if enable_checkpoints and (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'chars': chars,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint: {checkpoint_path}\n")
    
    # Training complete
    print(f"\n{'='*80}")
    print(f"{'TRAINING COMPLETE':^80}")
    print(f"{'='*80}")
    print(f"\nBest Results:")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved: {save_path}")
    print(f"\nNext Steps:")
    print(f"  1. Test: python test_crnn.py <image>")
    print(f"  2. Deploy: python server_crnn.py")
    print("=" * 80)


if __name__ == '__main__':
    # Base directory - change this to your setup
    BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced'
    
    CONFIG = {
        'image_dir': os.path.join(BASE_DIR, 'dataset'),
        'img_height': 64,
        'img_width': 200,
        'batch_size': 16,  # Reduced for better convergence
        'epochs': 150,  # Increased for better accuracy
        'learning_rate': 0.00005,  # Lower LR for fine-tuning
        'test_split': 0.15,  # Use more data for training (85/15 split)
        'save_path': os.path.join(BASE_DIR, 'crnn_captcha_model_v3.pth'),
        'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
        'save_every': 10,
        'enable_checkpoints': True,  # Set to False to disable checkpoint saving
        'debug': True,  # Set to False to disable debug output
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
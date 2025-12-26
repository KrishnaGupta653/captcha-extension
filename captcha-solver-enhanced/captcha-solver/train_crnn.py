# """
# CRNN CAPTCHA Solver - Training Script
# ========================================
# Trains a CRNN (CNN + RNN + CTC) model for CAPTCHA recognition

# Features:
# - Handles variable-length CAPTCHAs
# - Automatic character set detection
# - Works with ANY characters (@, =, letters, numbers, symbols)
# - GPU accelerated training
# - Auto-saves best model and checkpoints
# - Real-time accuracy tracking

# Usage:
#     python train_crnn.py

# Author: CAPTCHA Solver
# Date: 2024
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# import os
# import time
# import sys
# from crnn_model import CRNN
# from crnn_dataset import CRNNDataset, collate_fn


# def train_crnn_model(
#     image_dir='C:\\Users\\kg060\\Desktop\\tatkal\\captcha_solver\\labeled_captchas',
#     img_height=64,
#     img_width=200,
#     batch_size=32,
#     epochs=100,
#     learning_rate=0.0001,
#     test_split=0.2,
#     save_path='crnn_captcha_model.pth',
#     checkpoint_dir='checkpoints',
#     save_every=10
# ):
#     """
#     Train CRNN model for CAPTCHA recognition
    
#     Args:
#         image_dir (str): Directory containing labeled CAPTCHA images
#         img_height (int): Height to resize images to
#         img_width (int): Width to resize images to
#         batch_size (int): Batch size for training
#         epochs (int): Number of training epochs
#         learning_rate (float): Learning rate for optimizer
#         test_split (float): Fraction of data to use for validation (0.0-1.0)
#         save_path (str): Path to save the final model
#         checkpoint_dir (str): Directory to save checkpoints
#         save_every (int): Save checkpoint every N epochs
#     """
    
#     print("=" * 80)
#     print(" " * 20 + "CRNN CAPTCHA SOLVER - TRAINING")
#     print("=" * 80)
#     print()
    
#     # ========================================================================
#     # STEP 1: CHECK DATA DIRECTORY
#     # ========================================================================
#     if not os.path.exists(image_dir):
#         print(f"❌ ERROR: Directory '{image_dir}' not found!")
#         print()
#         print("Please create the directory and add your CAPTCHA images:")
#         print(f"  1. mkdir {image_dir}")
#         print(f"  2. Copy your CAPTCHA images to {image_dir}/")
#         sys.exit(1)
    
#     # Check if directory is empty
#     image_files = [f for f in os.listdir(image_dir) 
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     if len(image_files) == 0:
#         print(f"❌ ERROR: No images found in '{image_dir}'!")
#         print()
#         print("Please add CAPTCHA images to the directory.")
#         print("Filename should match the CAPTCHA text:")
#         print("  Example: @=@9T9.png (for CAPTCHA showing '@=@9T9')")
#         print()
#         sys.exit(1)
    
#     print(f"✓ Found {len(image_files)} images in '{image_dir}'")
#     print()
    
#     # Create checkpoint directory
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     # ========================================================================
#     # STEP 2: SETUP DEVICE (GPU/CPU)
#     # ========================================================================
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("-" * 80)
#     print("Device Configuration:")
#     print("-" * 80)
#     print(f"Device: {device}")
#     if device.type == 'cuda':
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA Version: {torch.version.cuda}")
#         print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#     else:
#         print("⚠️  Using CPU - Training will be slower")
#         print("   Consider using a GPU for 10x speed improvement")
#     print("-" * 80)
#     print()
    
#     # ========================================================================
#     # STEP 3: LOAD DATASET
#     # ========================================================================
#     print("Loading dataset...")
#     try:
#         dataset = CRNNDataset(image_dir, img_height, img_width)
#     except Exception as e:
#         print(f"❌ ERROR loading dataset: {e}")
#         sys.exit(1)
    
#     print()
#     print("-" * 80)
#     print("Dataset Information:")
#     print("-" * 80)
#     print(f"Total images: {len(dataset)}")
#     print(f"Image size: {img_height}x{img_width}")
#     print(f"Character set size: {len(dataset.chars)}")
#     print(f"Characters: {''.join(sorted(dataset.chars))}")
#     print("-" * 80)
#     print()
    
#     # Split into train and validation
#     val_size = int(len(dataset) * test_split)
#     train_size = len(dataset) - val_size
    
#     if train_size < 100:
#         print(f"⚠️  WARNING: Only {train_size} training samples!")
#         print("   Recommend at least 500 images for good accuracy")
#         print()
    
#     train_dataset, val_dataset = random_split(
#         dataset, 
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )
    
#     print(f"Training samples: {train_size}")
#     print(f"Validation samples: {val_size}")
#     print()
    
#     # ========================================================================
#     # STEP 4: CREATE DATA LOADERS
#     # ========================================================================
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,  # Set to 0 for Windows compatibility
#         collate_fn=collate_fn,
#         pin_memory=True if device.type == 'cuda' else False
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn,
#         pin_memory=True if device.type == 'cuda' else False
#     )
    
#     # ========================================================================
#     # STEP 5: INITIALIZE MODEL
#     # ========================================================================
#     print("Initializing CRNN model...")
#     num_chars = len(dataset.chars) + 1  # +1 for CTC blank character
#     model = CRNN(img_height=img_height, num_chars=num_chars, num_hidden=256)
#     model = model.to(device)
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print()
#     print("-" * 80)
#     print("Model Architecture:")
#     print("-" * 80)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#     print(f"Model size: ~{total_params * 4 / 1e6:.2f} MB")
#     print("-" * 80)
#     print()
    
#     # ========================================================================
#     # STEP 6: SETUP TRAINING
#     # ========================================================================
#     criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, verbose=True
#     )
    
#     print("-" * 80)
#     print("Training Configuration:")
#     print("-" * 80)
#     print(f"Epochs: {epochs}")
#     print(f"Batch size: {batch_size}")
#     print(f"Learning rate: {learning_rate}")
#     print(f"Optimizer: Adam")
#     print(f"Loss function: CTC Loss")
#     print(f"LR Scheduler: ReduceLROnPlateau")
#     print("-" * 80)
#     print()
    
#     # ========================================================================
#     # STEP 7: TRAINING LOOP
#     # ========================================================================
#     print("=" * 80)
#     print(" " * 28 + "STARTING TRAINING")
#     print("=" * 80)
#     print()
    
#     best_val_loss = float('inf')
#     best_val_acc = 0.0
#     training_history = []
    
#     for epoch in range(epochs):
#         epoch_start_time = time.time()
        
#         # ====================================================================
#         # TRAINING PHASE
#         # ====================================================================
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
        
#         print(f"Epoch [{epoch+1}/{epochs}]")
#         print("-" * 80)
        
#         for batch_idx, (images, labels, label_lengths, label_texts) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device)
#             label_lengths = label_lengths.to(device)
            
#             # Forward pass
#             outputs = model(images)  # [T, B, num_chars]
            
#             # Calculate input lengths
#             batch_size = images.size(0)
#             input_lengths = torch.full(
#                 size=(batch_size,),
#                 fill_value=outputs.size(0),
#                 dtype=torch.long
#             )
            
#             # CTC loss
#             outputs_log = outputs.log_softmax(2)
#             loss = criterion(outputs_log, labels, input_lengths, label_lengths)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             optimizer.step()
            
#             train_loss += loss.item()
            
#             # Calculate accuracy
#             _, preds = outputs.max(2)
#             preds = preds.transpose(1, 0).contiguous()
            
#             for i in range(batch_size):
#                 pred_text = decode_prediction(
#                     preds[i].cpu().numpy(),
#                     dataset.idx_to_char
#                 )
#                 if pred_text == label_texts[i]:
#                     train_correct += 1
#                 train_total += 1
            
#             # Print progress every 10 batches
#             if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
#                 current_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
#                 print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | "
#                       f"Loss: {loss.item():.4f} | "
#                       f"Acc: {current_acc:.2f}%")
        
#         avg_train_loss = train_loss / len(train_loader)
#         train_acc = 100.0 * train_correct / train_total
        
#         # ====================================================================
#         # VALIDATION PHASE
#         # ====================================================================
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
        
#         print()
#         print("  Validating...")
        
#         with torch.no_grad():
#             for images, labels, label_lengths, label_texts in val_loader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 label_lengths = label_lengths.to(device)
                
#                 outputs = model(images)
#                 batch_size = images.size(0)
#                 input_lengths = torch.full(
#                     size=(batch_size,),
#                     fill_value=outputs.size(0),
#                     dtype=torch.long
#                 )
                
#                 outputs_log = outputs.log_softmax(2)
#                 loss = criterion(outputs_log, labels, input_lengths, label_lengths)
#                 val_loss += loss.item()
                
#                 # Calculate accuracy
#                 _, preds = outputs.max(2)
#                 preds = preds.transpose(1, 0).contiguous()
                
#                 for i in range(batch_size):
#                     pred_text = decode_prediction(
#                         preds[i].cpu().numpy(),
#                         dataset.idx_to_char
#                     )
#                     if pred_text == label_texts[i]:
#                         val_correct += 1
#                     val_total += 1
        
#         avg_val_loss = val_loss / len(val_loader)
#         val_acc = 100.0 * val_correct / val_total
        
#         epoch_time = time.time() - epoch_start_time
        
#         # ====================================================================
#         # EPOCH SUMMARY
#         # ====================================================================
#         print()
#         print("=" * 80)
#         print(f"Epoch [{epoch+1}/{epochs}] Summary:")
#         print("=" * 80)
#         print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
#         print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
#         print(f"  Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # Save to history
#         training_history.append({
#             'epoch': epoch + 1,
#             'train_loss': avg_train_loss,
#             'train_acc': train_acc,
#             'val_loss': avg_val_loss,
#             'val_acc': val_acc,
#             'time': epoch_time
#         })
        
#         # ====================================================================
#         # SAVE BEST MODEL
#         # ====================================================================
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_val_acc = val_acc
            
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss': avg_train_loss,
#                 'val_loss': avg_val_loss,
#                 'train_acc': train_acc,
#                 'val_acc': val_acc,
#                 'char_to_idx': dataset.char_to_idx,
#                 'idx_to_char': dataset.idx_to_char,
#                 'chars': dataset.chars,
#                 'img_height': img_height,
#                 'img_width': img_width,
#                 'best_val_loss': best_val_loss,
#                 'best_val_acc': best_val_acc,
#             }, save_path)
            
#             print(f"  ✓ BEST MODEL SAVED! (Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        
#         print("=" * 80)
#         print()
        
#         # ====================================================================
#         # LEARNING RATE SCHEDULING
#         # ====================================================================
#         scheduler.step(avg_val_loss)
        
#         # ====================================================================
#         # SAVE CHECKPOINT
#         # ====================================================================
#         if (epoch + 1) % save_every == 0:
#             checkpoint_path = os.path.join(
#                 checkpoint_dir,
#                 f'checkpoint_epoch_{epoch+1}.pth'
#             )
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'train_loss': avg_train_loss,
#                 'val_loss': avg_val_loss,
#                 'char_to_idx': dataset.char_to_idx,
#                 'idx_to_char': dataset.idx_to_char,
#                 'chars': dataset.chars,
#             }, checkpoint_path)
#             print(f"  ✓ Checkpoint saved: {checkpoint_path}")
#             print()
    
#     # ========================================================================
#     # TRAINING COMPLETE
#     # ========================================================================
#     print()
#     print("=" * 80)
#     print(" " * 26 + "TRAINING COMPLETE!")
#     print("=" * 80)
#     print()
#     print("Training Summary:")
#     print("-" * 80)
#     print(f"Total epochs: {epochs}")
#     print(f"Best validation loss: {best_val_loss:.4f}")
#     print(f"Best validation accuracy: {best_val_acc:.2f}%")
#     print(f"Model saved to: {save_path}")
#     print(f"Character set: {''.join(sorted(dataset.chars))}")
#     print("-" * 80)
#     print()
#     print("Next Steps:")
#     print("  1. Start the server: python server_crnn.py")
#     print("  2. Test the model: python test_crnn.py <image_path>")
#     print()
#     print("=" * 80)


# def decode_prediction(pred_indices, idx_to_char):
#     """
#     Decode CTC prediction to text
#     Removes blanks and consecutive duplicates
    
#     Args:
#         pred_indices: Array of predicted indices
#         idx_to_char: Dictionary mapping indices to characters
    
#     Returns:
#         Decoded text string
#     """
#     result = []
#     prev_idx = -1
    
#     for idx in pred_indices:
#         # Skip blank (0) and consecutive duplicates
#         if idx != 0 and idx != prev_idx:
#             if idx in idx_to_char:
#                 result.append(idx_to_char[idx])
#         prev_idx = idx
    
#     return ''.join(result)


# if __name__ == '__main__':
#     # ========================================================================
#     # CONFIGURATION
#     # ========================================================================
#     # You can modify these parameters based on your needs
    
#     CONFIG = {
#         'image_dir': 'labeled_captchas',  # Directory with CAPTCHA images
#         'img_height': 64,                  # Image height
#         'img_width': 200,                  # Image width
#         'batch_size': 32,                  # Batch size (reduce if OOM)
#         'epochs': 50,                      # Number of epochs
#         'learning_rate': 0.001,            # Learning rate
#         'test_split': 0.2,                 # Validation split (20%)
#         'save_path': 'crnn_captcha_model.pth',  # Model save path
#         'checkpoint_dir': 'checkpoints',   # Checkpoint directory
#         'save_every': 10,                  # Save checkpoint every N epochs
#     }
    
#     # Start training
#     try:
#         train_crnn_model(**CONFIG)
#     except KeyboardInterrupt:
#         print("\n\n" + "=" * 80)
#         print(" " * 28 + "TRAINING INTERRUPTED")
#         print("=" * 80)
#         print("\nTraining was stopped by user.")
#         print("Partial results may have been saved.")
#         print()
#     except Exception as e:
#         print("\n\n" + "=" * 80)
#         print(" " * 32 + "ERROR")
#         print("=" * 80)
#         print(f"\nAn error occurred during training:")
#         print(f"  {str(e)}")
#         print()
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
"""
CRNN CAPTCHA Solver - Optimized Training Script
Streamlined, efficient, and production-ready
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


def train_epoch(model, train_loader, criterion, optimizer, device, idx_to_char, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (images, labels, label_lengths, label_texts) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward pass
        outputs = model(images)  # [T, B, num_chars]
        batch_size = images.size(0)
        
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
        total_correct += calculate_accuracy(outputs, label_texts, idx_to_char)
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
    script_dir = os.path.dirname(os.path.abspath(__file__)),
    img_height=64,
    img_width=200,
    batch_size=32,
    epochs=100,
    learning_rate=0.0001,
    test_split=0.2,
    save_path='crnn_captcha_model_v2.pth',
    checkpoint_dir='checkpoints',
    save_every=10
):
    """Main training function"""
    
    print("=" * 80)
    print(f"{'CRNN CAPTCHA TRAINER':^80}")
    print("=" * 80)
    
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
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device.type.upper()}", end='')
    if device.type == 'cuda':
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (Consider using GPU for 10x speedup)")
    
    # Load dataset
    print(f"\n{'Dataset Loading':-^80}")
    dataset = CRNNDataset(image_dir, img_height, img_width)
    
    print(f"Total images: {len(dataset)}")
    print(f"Characters ({len(dataset.chars)}): {''.join(sorted(dataset.chars))}")
    
    # Split dataset
    val_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {train_size} | Val: {val_size}")
    
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
    num_chars = len(dataset.chars) + 1  # +1 for CTC blank
    model = CRNN(img_height=img_height, num_chars=num_chars, num_hidden=256)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (~{total_params * 4 / 1e6:.1f} MB)")
    
    # Training setup
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"\n{'Training Config':-^80}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
    print(f"Classes: {num_chars} (including blank)")
    
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
            device, dataset.idx_to_char, epoch
        )
        
        # Validate
        print("\n  Validating...")
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, dataset.idx_to_char
        )
        
        epoch_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{epochs}] Complete:")
        print(f"  Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
        print(f"  Val:   Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
                'char_to_idx': dataset.char_to_idx,
                'idx_to_char': dataset.idx_to_char,
                'chars': dataset.chars,
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
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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
    BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced'
    CONFIG = {
        'image_dir': os.path.join(BASE_DIR, 'dataset'),
        'img_height': 64,
        'img_width': 200,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0001,
        'test_split': 0.2,
        'save_path': os.path.join(BASE_DIR, 'crnn_captcha_model_v3.pth'),
        'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
        'save_every': 10,
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
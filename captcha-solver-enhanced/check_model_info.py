"""
Check what's inside your saved model checkpoint
"""

import torch
import os

def inspect_checkpoint(checkpoint_path):
    """
    Display all information stored in a model checkpoint
    """
    print("=" * 80)
    print(f"{'MODEL CHECKPOINT INSPECTOR':^80}")
    print("=" * 80)
    print(f"\nFile: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ ERROR: File not found!")
        return
    
    # File size
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Size: {file_size_mb:.2f} MB")
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("\n" + "=" * 80)
    print(f"{'CHECKPOINT CONTENTS':^80}")
    print("=" * 80)
    
    # Display all keys
    print(f"\nKeys in checkpoint:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  - {key}: <dict with {len(checkpoint[key])} items>")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  - {key}: <tensor shape {checkpoint[key].shape}>")
        else:
            print(f"  - {key}: {type(checkpoint[key]).__name__}")
    
    print("\n" + "-" * 80)
    print(f"{'TRAINING METADATA':^80}")
    print("-" * 80)
    
    # Key information
    if 'epoch' in checkpoint:
        print(f"\n✓ Last Epoch Saved: {checkpoint['epoch']}")
        print(f"  (This model was saved after completing epoch {checkpoint['epoch']})")
    
    if 'best_val_acc' in checkpoint:
        print(f"\n✓ Best Validation Accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    if 'val_acc' in checkpoint:
        print(f"✓ Validation Accuracy (this epoch): {checkpoint['val_acc']:.2f}%")
    
    if 'train_acc' in checkpoint:
        print(f"✓ Train Accuracy (this epoch): {checkpoint['train_acc']:.2f}%")
    
    if 'train_loss' in checkpoint:
        print(f"✓ Train Loss: {checkpoint['train_loss']:.4f}")
    
    if 'val_loss' in checkpoint:
        print(f"✓ Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Model configuration
    print("\n" + "-" * 80)
    print(f"{'MODEL CONFIGURATION':^80}")
    print("-" * 80)
    
    if 'img_height' in checkpoint:
        print(f"\n✓ Image Height: {checkpoint['img_height']}")
    
    if 'img_width' in checkpoint:
        print(f"✓ Image Width: {checkpoint['img_width']}")
    
    if 'chars' in checkpoint:
        chars = checkpoint['chars']
        print(f"✓ Character Set ({len(chars)} chars): {chars}")
    
    if 'char_to_idx' in checkpoint:
        print(f"✓ Character to Index mapping: {len(checkpoint['char_to_idx'])} entries")
    
    # Model state
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\n✓ Model Parameters: {total_params:,}")
        print(f"✓ Model Layers: {len(state_dict)} layers")
    
    if 'optimizer_state_dict' in checkpoint:
        print(f"✓ Optimizer State: Saved")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"{'SUMMARY':^80}")
    print("=" * 80)
    
    if 'epoch' in checkpoint and 'best_val_acc' in checkpoint:
        print(f"\nThis checkpoint represents:")
        print(f"  📍 Training stopped/saved at: Epoch {checkpoint['epoch']}")
        print(f"  🏆 Best accuracy achieved: {checkpoint['best_val_acc']:.2f}%")
        
        if 'val_acc' in checkpoint:
            if checkpoint['val_acc'] >= checkpoint['best_val_acc']:
                print(f"  ✓ This IS the best model (saved because it improved)")
            else:
                print(f"  ⚠ This is NOT the best model")
                print(f"    Current acc: {checkpoint['val_acc']:.2f}%")
                print(f"    Best acc: {checkpoint['best_val_acc']:.2f}%")
        
        print(f"\n💡 To resume training:")
        print(f"   - Use 'resume_from': Set this path")
        print(f"   - Training will continue from epoch {checkpoint['epoch'] + 1}")
        print(f"   - To train 10 more epochs, set 'epochs': {checkpoint['epoch'] + 10}")
        print(f"   - To train 50 more epochs, set 'epochs': {checkpoint['epoch'] + 50}")
    
    print("=" * 80)


if __name__ == '__main__':
    # YOUR MODEL PATH
    BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best'
    model_path = os.path.join(BASE_DIR, 'crnn_captcha_model_v5.pth')
    
    inspect_checkpoint(model_path)
    
    # If you have multiple checkpoints, check them all:
    print("\n\n" + "=" * 80)
    print(f"{'CHECKING FOR OTHER CHECKPOINTS':^80}")
    print("=" * 80)
    
    checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoint_files:
            print(f"\nFound {len(checkpoint_files)} checkpoint(s) in '{checkpoint_dir}':")
            for f in checkpoint_files:
                print(f"  - {f}")
            print("\n💡 Inspect these files by changing 'model_path' above")
        else:
            print(f"\nNo checkpoint files found in '{checkpoint_dir}'")
    else:
        print(f"\nCheckpoint directory doesn't exist: '{checkpoint_dir}'")
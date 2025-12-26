import torch
import torch.nn as nn
from crnn_model import CRNN
import json
import os

# --- CONFIG ---
# Make sure these match exactly what you used in train_crnn.py
BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best'
MODEL_PATH = os.path.join(BASE_DIR, 'crnn_captcha_model_transfer_v7.pth')
ONNX_PATH = os.path.join(BASE_DIR, 'captcha_solver.onnx')
IMG_HEIGHT = 64
IMG_WIDTH = 200

def export():
    print(f"Loading checkpoint from: {MODEL_PATH}")
    
    # 1. Load Checkpoint
    # weights_only=False is required for this specific checkpoint format
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    idx_to_char = checkpoint['idx_to_char']
    
    # --- THE FIX IS HERE ---
    # The saved idx_to_char ALREADY includes the <BLANK> key (index 0).
    # So we should NOT add +1 again.
    num_chars = len(idx_to_char) 
    
    print(f"Detected {num_chars} classes (including blank).")

    # 2. Initialize Model with correct size
    model = CRNN(IMG_HEIGHT, num_chars, num_hidden=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ Model weights loaded successfully!")

    # 3. Create dummy input
    dummy_input = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH)

    # 4. Export to ONNX
    print(f"Exporting to {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {1: 'batch_size'}
        }
    )

    # 5. Save Vocab (Critical for the extension to know what index 1 means)
    vocab_path = os.path.join(BASE_DIR, 'vocab.json')
    with open(vocab_path, 'w') as f:
        # Convert integer keys to strings for JSON compatibility
        json_safe_vocab = {str(k): v for k, v in idx_to_char.items()}
        json.dump(json_safe_vocab, f)

    print(f"✅ Success! Files created:\n 1. {ONNX_PATH}\n 2. {vocab_path}")

if __name__ == "__main__":
    try:
        export()
    except Exception as e:
        print(f"\n❌ Error: {e}")
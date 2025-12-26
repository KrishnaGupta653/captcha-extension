import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader
from crnn_model import CRNN
from crnn_dataset import CRNNDataset, collate_fn
from collections import Counter
import sys

# --- CONFIG ---
BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best'
MODEL_PATH = os.path.join(BASE_DIR, 'crnn_captcha_model_v5.pth') 
IMAGE_DIR = os.path.join(BASE_DIR, 'dataset')
IMG_HEIGHT = 64
IMG_WIDTH = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode(preds, idx_to_char):
    # preds shape is now [Batch, Time] after the fix in analyze()
    decoded_batch = []
    for sequence in preds:
        text = ""
        prev_idx = -1
        for idx in sequence:
            if idx != 0 and idx != prev_idx:
                text += idx_to_char[idx]
            prev_idx = idx
        decoded_batch.append(text)
    return decoded_batch

def analyze():
    print(f"Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Load Metadata
    idx_to_char = checkpoint['idx_to_char']
    num_chars = len(checkpoint['chars']) + 1
    
    # Load Model
    model = CRNN(IMG_HEIGHT, num_chars, num_hidden=512).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load Dataset (No Augmentation for testing)
    dataset = CRNNDataset(IMAGE_DIR, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, augment=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    print(f"Analyzing {len(dataset)} images...")
    
    errors = []
    char_confusions = Counter()
    
    with torch.no_grad():
        for images, _, _, texts in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            # --- CRITICAL FIX IS HERE ---
            # Output is [Time, Batch, Class]. We need [Batch, Time, Class]
            outputs = outputs.transpose(0, 1) 
            # ----------------------------
            
            # Get argmax to get indices [Batch, Time]
            preds_indices = outputs.argmax(dim=2).cpu().numpy()
            
            # Decode
            preds = decode(preds_indices, idx_to_char)
            
            for i, pred in enumerate(preds):
                # Safety check for index
                if i >= len(texts):
                    break
                    
                true_text = texts[i]
                if pred != true_text:
                    errors.append((true_text, pred))
                    
                    # Analyze specifically WHICH char is wrong
                    # Only compares if lengths match to avoid misalignment noise
                    if len(pred) == len(true_text):
                        for c_true, c_pred in zip(true_text, pred):
                            if c_true != c_pred:
                                char_confusions[f"{c_true} -> {c_pred}"] += 1

    # --- REPORT ---
    print("\n" + "="*50)
    print("🚨 ERROR ANALYSIS REPORT 🚨")
    print("="*50)
    print(f"Total Images: {len(dataset)}")
    print(f"Total Errors: {len(errors)}")
    accuracy = 100 * (len(dataset) - len(errors)) / len(dataset)
    print(f"Accuracy:     {accuracy:.2f}%")
    print("-" * 30)
    
    print("\nTOP 5 MOST COMMON CONFUSIONS:")
    if not char_confusions:
        if len(errors) > 0:
            print("  (Errors found, but they were length mismatches, e.g., Missing characters)")
        else:
            print("  (None! Perfect match)")
    for conf, count in char_confusions.most_common(5):
        print(f"  '{conf}' : {count} times")

    print("\nSAMPLE ERRORS (True vs Predicted):")
    for true_t, pred_t in errors[:20]: # Show top 20 errors
        print(f"  True: {true_t:<10} | Pred: {pred_t}")
        
    print("="*50)
    print("💡 SUGGESTIONS:")
    if len(errors) > 0:
        top_conf = char_confusions.most_common(1)
        if top_conf:
            bad_char = top_conf[0][0].split(' -> ')[0]
            print(f"1. Check your dataset images for the letter '{bad_char}'.")
            print(f"2. Is '{bad_char}' labeled correctly? Sometimes '0' is labeled as 'O'.")
        else:
            print("1. Your model is missing characters entirely.")
            print("2. Check if your images are too blurry or have lines crossing the text.")

if __name__ == '__main__':
    try:
        analyze()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
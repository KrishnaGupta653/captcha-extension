import torch
import os
import shutil
from torch.utils.data import DataLoader
from crnn_model import CRNN
from crnn_dataset import CRNNDataset, collate_fn
import sys

# --- CONFIGURATION ---
BASE_DIR = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best'
MODEL_PATH = os.path.join(BASE_DIR, 'crnn_captcha_model_v7.pth') 
IMAGE_DIR = os.path.join(BASE_DIR, 'dataset')
ERROR_DIR = os.path.join(BASE_DIR, 'dataset_errors') # New home for bad images

IMG_HEIGHT = 64
IMG_WIDTH = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode_batch(preds_indices, idx_to_char):
    decoded_batch = []
    for sequence in preds_indices:
        text = ""
        prev_idx = -1
        for idx in sequence:
            idx = int(idx)
            if idx != 0 and idx != prev_idx:
                text += idx_to_char[idx]
            prev_idx = idx
        decoded_batch.append(text)
    return decoded_batch

def move_errors():
    print(f"{'='*50}")
    print(f" DATASET CLEANER (Move Errors)")
    print(f"{'='*50}")
    
    # 1. Setup Output Directory
    if not os.path.exists(ERROR_DIR):
        os.makedirs(ERROR_DIR)
    print(f"Source: {IMAGE_DIR}")
    print(f"Target: {ERROR_DIR}\n")

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found!")
        return

    print(f"Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    idx_to_char = checkpoint['idx_to_char']
    num_chars = len(checkpoint['chars']) + 1
    
    model = CRNN(IMG_HEIGHT, num_chars, num_hidden=512).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Load Dataset (Sequential, No Shuffle)
    dataset = CRNNDataset(IMAGE_DIR, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, augment=False)
    
    # IMPORTANT: shuffle=False ensures indexes match dataset.image_files
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"Scanning {len(dataset)} images...\n")

    moved_count = 0
    processed = 0

    with torch.no_grad():
        for images, labels, _, true_texts in loader:
            images = images.to(DEVICE)
            
            # Predict
            outputs = model(images)
            outputs = outputs.transpose(0, 1) # [Batch, Time, Class]
            preds_indices = outputs.argmax(dim=2).cpu().numpy()
            pred_texts = decode_batch(preds_indices, idx_to_char)
            
            # Check Batch
            for i in range(len(true_texts)):
                true = true_texts[i]
                pred = pred_texts[i]
                
                # Global index in the dataset list
                global_idx = processed + i
                
                if true != pred:
                    # Get original filename
                    original_filename = dataset.image_files[global_idx]
                    source_path = os.path.join(IMAGE_DIR, original_filename)
                    
                    # Construct new helpful filename
                    # clean_true = "".join(x for x in true if x.isalnum())
                    clean_pred = "".join(x for x in pred if x.isalnum())
                    ext = os.path.splitext(original_filename)[1]
                    
                    # New Name: TRUE_abc_VS_PRED_xyz.png
                    # We keep the original 'true' label in filename so you can restore it if needed
                    new_name = f"TRUE_{true}__VS_PRED_{clean_pred}{ext}"
                    target_path = os.path.join(ERROR_DIR, new_name)
                    
                    # Handle duplicates
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(ERROR_DIR, f"TRUE_{true}__VS_PRED_{clean_pred}_{counter}{ext}")
                        counter += 1
                    
                    # MOVE THE FILE
                    try:
                        shutil.move(source_path, target_path)
                        print(f"Moved: {original_filename} -> {new_name}")
                        moved_count += 1
                    except Exception as e:
                        print(f"❌ Failed to move {original_filename}: {e}")

            processed += len(images)
            print(f"\rProgress: {processed}/{len(dataset)}", end="")

    print(f"\n\n{'='*50}")
    print(f"CLEANUP COMPLETE")
    print(f"Moved {moved_count} error images to: {ERROR_DIR}")
    print(f"Remaining clean images: {len(dataset) - moved_count}")
    print(f"{'='*50}")

if __name__ == '__main__':
    try:
        move_errors()
    except Exception as e:
        print(f"\n❌ Error: {e}")
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import logging
from pathlib import Path

# --- CONFIGURATION ---
IMAGE_FOLDER = r'C:\Users\kg060\Desktop\tatkal\dataset\raw_captcha_images'
MODEL_PATH = r'C:\Users\kg060\Desktop\tatkal\captcha-solver-enhanced-best\crnn_captcha_model_v7.pth'

IMG_HEIGHT = 64
IMG_WIDTH = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug settings
DEBUG_MODE = False  # Set True to see detailed logs
SAVE_PREPROCESSED = True  # Save preprocessed images for inspection
DEBUG_OUTPUT_DIR = 'debug_output'

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

# --- CRNN MODEL ---
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)
        output = self.embedding(t_rec)
        output = output.reshape(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, img_height, num_chars, num_hidden=256):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512 * 3, num_hidden, num_hidden),
            BidirectionalLSTM(num_hidden, num_hidden, num_chars)
        )

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.contiguous()
        conv = conv.view(b, c * h, w)
        conv = conv.permute(2, 0, 1)
        conv = conv.contiguous()
        output = self.rnn(conv)
        return output

# --- HELPER FUNCTIONS ---
def load_model():
    """Load the trained CRNN model"""
    logger.info(f"Loading model: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    chars = checkpoint['chars']
    idx_to_char = checkpoint['idx_to_char']
    
    global IMG_HEIGHT, IMG_WIDTH
    IMG_HEIGHT = checkpoint['img_height']
    IMG_WIDTH = checkpoint['img_width']
    
    num_chars = len(chars) + 1
    model = CRNN(img_height=IMG_HEIGHT, num_chars=num_chars, num_hidden=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(DEVICE)
    model.eval()
    torch.set_grad_enabled(False)
    
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    logger.info("✅ Model loaded")
    logger.info(f"📊 Characters: {''.join(sorted(chars))}")
    logger.info(f"📐 Size: {IMG_WIDTH}x{IMG_HEIGHT}")
    logger.info(f"📈 Val Accuracy: {checkpoint.get('val_acc', 0):.2f}%")
    
    return model, idx_to_char, chars

def decode_prediction(pred_indices, idx_to_char):
    """Decode CTC prediction"""
    result = []
    prev_idx = -1
    
    for idx in pred_indices:
        idx = int(idx)
        if idx != 0 and idx != prev_idx and idx in idx_to_char:
            result.append(idx_to_char[idx])
        prev_idx = idx
    
    return ''.join(result)

def preprocess_image_exact_match(image_path, save_debug=False):
    """
    EXACT COPY of preprocessing from crnn_dataset.py
    This is the critical fix - must match training preprocessing perfectly
    """
    try:
        # Load grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            logger.error(f"Failed to load: {image_path}")
            return None
        
        basename = os.path.basename(image_path)
        
        # Save original for debugging
        if save_debug:
            debug_dir = os.path.join(os.path.dirname(image_path), DEBUG_OUTPUT_DIR)
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"1_original_{basename}"), image)
        
        # ============================================
        # EXACT PREPROCESSING FROM crnn_dataset.py
        # ============================================
        
        # Step 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        if save_debug:
            debug_dir = os.path.join(os.path.dirname(image_path), DEBUG_OUTPUT_DIR)
            cv2.imwrite(os.path.join(debug_dir, f"2_clahe_{basename}"), image)
        
        # Step 2: Denoise
        image = cv2.fastNlMeansDenoising(image, h=10)
        
        if save_debug:
            debug_dir = os.path.join(os.path.dirname(image_path), DEBUG_OUTPUT_DIR)
            cv2.imwrite(os.path.join(debug_dir, f"3_denoised_{basename}"), image)
        
        # Step 3: Binarization with Otsu's method (CRITICAL!)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if save_debug:
            debug_dir = os.path.join(os.path.dirname(image_path), DEBUG_OUTPUT_DIR)
            cv2.imwrite(os.path.join(debug_dir, f"4_binary_{basename}"), image)
        
        # ============================================
        # End of preprocessing - now resize
        # ============================================
        
        # Resize
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
        if save_debug:
            debug_dir = os.path.join(os.path.dirname(image_path), DEBUG_OUTPUT_DIR)
            cv2.imwrite(os.path.join(debug_dir, f"5_resized_{basename}"), image)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add dimensions [1, 1, H, W]
        image = image[np.newaxis, np.newaxis, :, :]
        
        # Convert to tensor
        tensor = torch.from_numpy(image)
        
        if DEVICE == 'cuda':
            tensor = tensor.pin_memory()
        
        tensor = tensor.to(DEVICE, non_blocking=True)
        
        if DEBUG_MODE:
            logger.debug(f"  Tensor: {tensor.shape}, range [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        return tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing {image_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_captcha_label(model, idx_to_char, image_path):
    """Extract CAPTCHA text"""
    try:
        # Preprocess with EXACT training preprocessing
        image_tensor = preprocess_image_exact_match(
            image_path, 
            save_debug=(SAVE_PREPROCESSED and DEBUG_MODE)
        )
        
        if image_tensor is None:
            return None
        
        # Inference
        with torch.inference_mode():
            output = model(image_tensor)
            preds = output.argmax(dim=2).squeeze(1)
            pred_indices = preds.cpu().numpy()
        
        # Debug predictions
        if DEBUG_MODE:
            num_blanks = np.sum(pred_indices == 0)
            blank_pct = 100 * num_blanks / len(pred_indices)
            logger.debug(f"  Predictions for {os.path.basename(image_path)}:")
            logger.debug(f"    Blanks: {num_blanks}/{len(pred_indices)} ({blank_pct:.1f}%)")
            logger.debug(f"    Indices: {pred_indices[:30]}")
        
        # Decode
        text = decode_prediction(pred_indices, idx_to_char)
        text = text.strip()
        
        if DEBUG_MODE and text:
            logger.debug(f"    ✅ Result: '{text}'")
        
        return text if text else None
        
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(image_path)}: {e}")
        return None

def is_already_labeled(filename):
    """Check if already labeled"""
    name_no_ext = Path(filename).stem
    return (4 <= len(name_no_ext) <= 10 
            and not name_no_ext.split('_')[-1].isdigit())

def process_images():
    """Main processing"""
    
    if not os.path.exists(IMAGE_FOLDER):
        logger.error(f"Folder not found: {IMAGE_FOLDER}")
        return
    
    # Load model
    try:
        model, idx_to_char, chars = load_model()
    except Exception:
        return
    
    # Get images
    files = [f for f in os.listdir(IMAGE_FOLDER) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not files:
        logger.warning(f"No images in {IMAGE_FOLDER}")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📂 Found {len(files)} images")
    logger.info(f"{'='*60}\n")
    
    # Stats
    count_renamed = 0
    count_skipped = 0
    count_failed = 0
    failed_files = []
    
    # Process
    for i, filename in enumerate(files, 1):
        original_path = os.path.join(IMAGE_FOLDER, filename)
        
        # Skip labeled
        if is_already_labeled(filename):
            count_skipped += 1
            continue
        
        if DEBUG_MODE:
            logger.debug(f"\n[{i}/{len(files)}] {filename}")
        
        # Get label
        label = get_captcha_label(model, idx_to_char, original_path)
        
        if label:
            extension = Path(filename).suffix
            safe_label = label
            
            # Sanitize
            for char in '<>:"/\\|?*':
                safe_label = safe_label.replace(char, '_')
            
            new_filename = f"{safe_label}{extension}"
            new_path = os.path.join(IMAGE_FOLDER, new_filename)
            
            # Handle duplicates
            counter = 1
            while os.path.exists(new_path):
                if new_path == original_path:
                    break
                new_filename = f"{safe_label}_{counter}{extension}"
                new_path = os.path.join(IMAGE_FOLDER, new_filename)
                counter += 1
            
            # Rename
            if original_path != new_path:
                try:
                    os.rename(original_path, new_path)
                    logger.info(f"[{i}/{len(files)}] ✅ {filename} → {new_filename}")
                    count_renamed += 1
                except OSError as e:
                    logger.error(f"Rename failed {filename}: {e}")
                    count_failed += 1
                    failed_files.append(filename)
            else:
                logger.info(f"[{i}/{len(files)}] ✓ Already named: {filename}")
        else:
            logger.warning(f"[{i}/{len(files)}] ❌ Failed: {filename}")
            count_failed += 1
            failed_files.append(filename)
        
        # Progress
        if i % 50 == 0 or i == len(files):
            progress = (i / len(files)) * 100
            logger.info(f"Progress: {progress:.0f}% | ✅ {count_renamed} | ⏭️ {count_skipped} | ❌ {count_failed}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {len(files)}")
    logger.info(f"✅ Renamed: {count_renamed}")
    logger.info(f"⏭️ Skipped: {count_skipped}")
    logger.info(f"❌ Failed: {count_failed}")
    
    if count_renamed + count_failed > 0:
        success_rate = (count_renamed / (count_renamed + count_failed)) * 100
        logger.info(f"📈 Success: {success_rate:.1f}%")
    
    if failed_files:
        logger.info(f"\n❌ Failed files (first 10):")
        for f in failed_files[:10]:
            logger.info(f"  {f}")
        if len(failed_files) > 10:
            logger.info(f"  ... and {len(failed_files)-10} more")
    
    logger.info(f"{'='*60}")
    
    if SAVE_PREPROCESSED and count_failed > 0:
        debug_dir = os.path.join(IMAGE_FOLDER, DEBUG_OUTPUT_DIR)
        logger.info(f"\n💡 Preprocessed samples in: {debug_dir}")
        logger.info(f"   Check these to see how images are processed")

if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
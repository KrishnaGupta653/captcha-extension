from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import PIL.Image
import os
import logging
from pathlib import Path
import torch

# --- CONFIGURATION ---
# 1. Folder containing your images
IMAGE_FOLDER = r'C:\Users\kg060\Desktop\tatkal\cnn-captcha-solver\captcha_images'

# 2. Model Selection
# Using: "anuashok/ocr-captcha-v3" (98.6% accuracy, very stable for CAPTCHAs)
MODEL_NAME = "anuashok/ocr-captcha-v3"

# Model cache location - same folder as this script
SCRIPT_DIR = Path(__file__).parent
MODEL_CACHE_DIR = SCRIPT_DIR / "trocr_model_cache"

# 3. Device selection (auto-detects GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

def load_model():
    """
    Load TrOCR model and processor components.
    Models are saved in the same folder as this script.
    First run will download the model (~300MB).
    Subsequent runs use cached version.
    """
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Model cache: {MODEL_CACHE_DIR}")
    
    # Create cache directory if it doesn't exist
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    
    try:
        logger.info("Loading image processor...")
        image_processor = ViTImageProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        logger.info("Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        logger.info(f"Moving model to {DEVICE}...")
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"✅ Successfully loaded: {MODEL_NAME}")
        logger.info(f"📁 Model saved in: {MODEL_CACHE_DIR}")
        return image_processor, tokenizer, model, MODEL_NAME
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("💡 Troubleshooting tips:")
        logger.info("   1. Run: pip install --upgrade transformers torch pillow")
        logger.info("   2. Check internet connection for model download")
        logger.info("   3. Make sure you have ~500MB free disk space")
        raise

def get_captcha_label(image_processor, tokenizer, model, image_path):
    """
    Extract CAPTCHA text using TrOCR.
    Returns text with letters, numbers, and special chars like @, =, etc.
    """
    try:
        # Load and convert image to RGB
        image = PIL.Image.open(image_path).convert("RGB")
        
        # Preprocess image
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(DEVICE)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        # Decode prediction
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean text - only remove spaces and newlines, keep special chars
        text = text.strip().replace(" ", "").replace("\n", "")
        
        # Return if we got any text (allows alphanumeric + special chars)
        if text:
            return text
        
        return None
        
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(image_path)}: {e}")
        return None

def is_already_labeled(filename):
    """Check if file appears to be already labeled"""
    name_no_ext = Path(filename).stem
    # If it's 4-8 chars (allowing special chars) without underscore suffix, consider it labeled
    return (4 <= len(name_no_ext) <= 8 
            and not name_no_ext.split('_')[-1].isdigit())  # Skip if ends with _1, _2, etc.

def process_images():
    """Main processing function"""
    
    # Validate folder
    if not os.path.exists(IMAGE_FOLDER):
        logger.error(f"Folder not found: {IMAGE_FOLDER}")
        return
    
    # Load model once
    try:
        image_processor, tokenizer, model, loaded_model_name = load_model()
    except Exception:
        return
    
    # Get image files
    files = [f for f in os.listdir(IMAGE_FOLDER) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not files:
        logger.warning(f"No images found in {IMAGE_FOLDER}")
        return
    
    logger.info(f"📂 Found {len(files)} images. Starting processing...")
    logger.info(f"🤖 Using model: {loaded_model_name}")
    logger.info("="*50)
    
    # Statistics
    count_renamed = 0
    count_skipped = 0
    count_failed = 0
    
    for i, filename in enumerate(files, 1):
        original_path = os.path.join(IMAGE_FOLDER, filename)
        
        # Skip if already labeled
        if is_already_labeled(filename):
            count_skipped += 1
            if count_skipped % 50 == 0:  # Show progress every 50 skips
                print(f"Skipping already labeled files... ({count_skipped} so far)", end='\r')
            continue
        
        # Get label using TrOCR
        label = get_captcha_label(image_processor, tokenizer, model, original_path)
        
        if label:
            # Construct new filename
            extension = Path(filename).suffix
            
            # Sanitize label for filename (replace problematic chars)
            # Windows doesn't allow: < > : " / \ | ? *
            safe_label = label
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
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
            
            # Rename file
            if original_path != new_path:
                try:
                    os.rename(original_path, new_path)
                    logger.info(f"[{i}/{len(files)}] ✅ {filename} -> {new_filename}")
                    count_renamed += 1
                except OSError as e:
                    logger.error(f"Could not rename {filename}: {e}")
                    count_failed += 1
            else:
                logger.info(f"[{i}/{len(files)}] Already correctly named: {filename}")
        else:
            logger.warning(f"[{i}/{len(files)}] ❌ Failed to extract label from {filename}")
            count_failed += 1
    
    # Print summary
    logger.info("="*50)
    logger.info("📊 SUMMARY")
    logger.info("="*50)
    logger.info(f"Total files: {len(files)}")
    logger.info(f"✅ Renamed: {count_renamed}")
    logger.info(f"⏭️  Skipped (already labeled): {count_skipped}")
    logger.info(f"❌ Failed: {count_failed}")
    
    if count_renamed + count_failed > 0:
        success_rate = (count_renamed / (count_renamed + count_failed)) * 100
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
    
    logger.info("="*50)

if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
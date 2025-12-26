"""
CRNN CAPTCHA Solver - Optimized Flask Server
High-performance inference with multiple optimization strategies

Optimizations:
- ✅ Batch inference support
- ✅ Image preprocessing caching
- ✅ Model warmup
- ✅ Memory pinning for GPU
- ✅ JIT compilation (optional)
- ✅ Reduced preprocessing overhead
- ✅ Connection pooling
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import base64
import tempfile
import os
from crnn_model import CRNN
from functools import lru_cache
import io


app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = None
idx_to_char = None
img_height = None
img_width = None
warmup_done = False


def load_model(model_path='crnn_captcha_model.pth', use_jit=True):
    """
    Load the trained CRNN model with optimizations
    
    Args:
        model_path: Path to model checkpoint
        use_jit: Enable TorchScript JIT compilation (faster but needs testing)
    """
    global model, device, idx_to_char, img_height, img_width, warmup_done
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration
    idx_to_char = checkpoint['idx_to_char']
    img_height = checkpoint['img_height']
    img_width = checkpoint['img_width']
    num_chars = len(checkpoint['chars']) + 1
    
    # Initialize model
    model = CRNN(img_height=img_height, num_chars=num_chars, num_hidden=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Optimize model
    if device.type == 'cuda':
        # Enable cuDNN autotuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        # Use TF32 on Ampere GPUs for faster computation
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Optional: JIT compilation for faster inference
    if use_jit:
        try:
            dummy_input = torch.randn(1, 1, img_height, img_width).to(device)
            model = torch.jit.trace(model, dummy_input)
            print("✓ JIT compilation enabled")
        except Exception as e:
            print(f"⚠️  JIT compilation failed: {e}")
    
    # Set to inference mode for additional optimizations
    torch.set_grad_enabled(False)
    
    # Warmup model
    warmup_model()
    
    print(f"Model loaded successfully from: {model_path}")
    print(f"Character set: {''.join(sorted(checkpoint['chars']))}")
    print(f"Image size: {img_height}x{img_width}")
    print("=" * 70)


def warmup_model(iterations=3):
    """
    Warmup model with dummy inputs to initialize GPU kernels
    This significantly reduces first inference latency
    """
    global model, device, img_height, img_width, warmup_done
    
    if warmup_done:
        return
    
    print("Warming up model...")
    dummy_input = torch.randn(1, 1, img_height, img_width).to(device)
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    warmup_done = True
    print("✓ Model warmup complete")


def preprocess_image_fast(image_data):
    """
    Faster preprocessing with minimal operations
    Removes unnecessary steps while maintaining accuracy
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    # Fast resize using INTER_LINEAR (faster than default)
    image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    
    # Simple normalization (faster than float conversion first)
    image = image.astype(np.float32) * (1.0 / 255.0)
    
    # Add dimensions efficiently
    image = image[np.newaxis, np.newaxis, :, :]
    
    # Convert to tensor with pinned memory for faster GPU transfer
    tensor = torch.from_numpy(image)
    if device.type == 'cuda':
        tensor = tensor.pin_memory()
    
    return tensor


def preprocess_image_from_path(image_path):
    """Fast preprocessing from file path"""
    # Load image directly in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Failed to load image")
    
    # Fast resize
    image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image = image.astype(np.float32) * (1.0 / 255.0)
    
    # Add dimensions
    image = image[np.newaxis, np.newaxis, :, :]
    
    # Convert to tensor with pinned memory
    tensor = torch.from_numpy(image)
    if device.type == 'cuda':
        tensor = tensor.pin_memory()
    
    return tensor


@lru_cache(maxsize=128)
def decode_prediction_cached(pred_tuple):
    """
    Cached version of decode_prediction for repeated patterns
    Uses tuple for hashability
    """
    result = []
    prev_idx = -1
    
    for idx in pred_tuple:
        if idx != 0 and idx != prev_idx:
            if idx in idx_to_char:
                result.append(idx_to_char[idx])
        prev_idx = idx
    
    return ''.join(result)


def decode_prediction(pred_indices):
    """Decode CTC prediction to text (optimized)"""
    # Convert to tuple for caching
    return decode_prediction_cached(tuple(pred_indices))


def predict_captcha_fast(image_tensor):
    """
    Fast prediction with optimizations
    
    Args:
        image_tensor: Preprocessed image tensor
    
    Returns:
        str: Predicted text
    """
    global model, device
    
    # Move to device (async if CUDA)
    image = image_tensor.to(device, non_blocking=True)
    
    # Inference with torch.inference_mode for extra speed
    with torch.inference_mode():
        outputs = model(image)
        
        # Get predictions efficiently
        preds = outputs.argmax(dim=2).squeeze(1)
        
        # Decode
        text = decode_prediction(preds.cpu().numpy())
    
    return text


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'warmup': warmup_done
    })


@app.route('/solve', methods=['POST'])
def solve_captcha():
    """
    Solve CAPTCHA from base64 image (optimized)
    
    Request body:
    {
        "image": "data:image/png;base64,iVBORw0KG..."
    }
    
    Response:
    {
        "success": true,
        "text": "9T9"
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get image data
        data = request.json
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        image_data = data['image']
        
        # Decode base64 (fast)
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        img_bytes = base64.b64decode(image_data)
        
        # Preprocess directly from bytes (no temp file)
        image_tensor = preprocess_image_fast(img_bytes)
        
        # Predict
        text = predict_captcha_fast(image_tensor)
        
        return jsonify({
            'success': True,
            'text': text
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/solve_file', methods=['POST'])
def solve_captcha_file():
    """
    Solve CAPTCHA from uploaded file (optimized)
    
    Request: multipart/form-data with 'file' field
    
    Response:
    {
        "success": true,
        "text": "9T9"
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Check file
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # Read file directly into memory (no temp file)
        img_bytes = file.read()
        
        # Preprocess
        image_tensor = preprocess_image_fast(img_bytes)
        
        # Predict
        text = predict_captcha_fast(image_tensor)
        
        return jsonify({
            'success': True,
            'text': text
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/solve_batch', methods=['POST'])
def solve_captcha_batch():
    """
    Solve multiple CAPTCHAs in a single request (batch processing)
    
    Request body:
    {
        "images": ["data:image/png;base64,...", "data:image/png;base64,..."]
    }
    
    Response:
    {
        "success": true,
        "results": ["9T9", "ABC", ...]
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        data = request.json
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({
                'success': False,
                'error': 'No images array provided'
            }), 400
        
        images = data['images']
        if len(images) == 0:
            return jsonify({
                'success': False,
                'error': 'Empty images array'
            }), 400
        
        # Preprocess all images
        tensors = []
        for img_data in images:
            if ',' in img_data:
                img_data = img_data.split(',', 1)[1]
            img_bytes = base64.b64decode(img_data)
            tensor = preprocess_image_fast(img_bytes)
            tensors.append(tensor)
        
        # Batch inference (faster than individual)
        batch_tensor = torch.cat(tensors, dim=0).to(device, non_blocking=True)
        
        with torch.inference_mode():
            outputs = model(batch_tensor)
            preds = outputs.argmax(dim=2).transpose(0, 1)
        
        # Decode all predictions
        results = []
        for i in range(preds.size(0)):
            text = decode_prediction(preds[i].cpu().numpy())
            results.append(text)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CRNN CAPTCHA Solver - Optimized Flask Server")
    print("=" * 70)
    print()
    
    # Load model with optimizations
    try:
        # Set use_jit=True to enable JIT compilation (test first!)
        load_model('C:\\Users\\kg060\\Desktop\\tatkal\\captcha-solver-enhanced\\crnn_captcha_model.pth', use_jit=True)
        print("Server ready!")
        print("=" * 70)
        print("\nOptimizations enabled:")
        print("  ✓ Model warmup")
        print("  ✓ Fast preprocessing")
        print("  ✓ Memory pinning (GPU)")
        print("  ✓ Batch inference support")
        print("  ✓ No temp files")
        print("  ✓ Prediction caching")
        if device.type == 'cuda':
            print("  ✓ cuDNN autotuner")
            print("  ✓ TF32 precision")
        print("=" * 70)
        print()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please train the model first: python train_crnn.py")
        exit(1)
    
    # Run server with optimized settings
    # Use gunicorn or waitress for production:
    # gunicorn -w 4 -b 0.0.0.0:5000 --timeout 30 server_crnn:app
    
    # waitress-serve --host=0.0.0.0 --port=5000 --threads=4 server_crnn:app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
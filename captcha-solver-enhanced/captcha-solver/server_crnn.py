#!/usr/bin/env python3
"""
CRNN CAPTCHA Solver - Flask Server
Serves the trained CRNN model via REST API
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


app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = None
idx_to_char = None
img_height = None
img_width = None


def load_model(model_path='crnn_captcha_model.pth'):
    """Load the trained CRNN model"""
    global model, device, idx_to_char, img_height, img_width
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    idx_to_char = checkpoint['idx_to_char']
    img_height = checkpoint['img_height']
    img_width = checkpoint['img_width']
    num_chars = len(checkpoint['chars']) + 1
    
    # Initialize and load model
    model = CRNN(img_height=img_height, num_chars=num_chars, num_hidden=256)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from: {model_path}")
    print(f"Character set: {''.join(sorted(checkpoint['chars']))}")
    print(f"Image size: {img_height}x{img_width}")
    print("=" * 70)


def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Failed to load image")
    
    # Resize to model input size
    image = cv2.resize(image, (img_width, img_height))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions [1, 1, H, W]
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    
    # Convert to tensor
    image = torch.FloatTensor(image)
    
    return image


def decode_prediction(pred_indices):
    """Decode CTC prediction to text"""
    result = []
    prev_idx = -1
    
    for idx in pred_indices:
        if idx != 0 and idx != prev_idx:  # Skip blank and duplicates
            if idx in idx_to_char:
                result.append(idx_to_char[idx])
        prev_idx = idx
    
    return ''.join(result)


def predict_captcha(image_path):
    """Predict CAPTCHA text from image"""
    global model, device
    
    # Preprocess image
    image = preprocess_image(image_path).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image)  # [T, 1, num_chars]
        
        # Get predictions
        _, preds = outputs.max(2)
        preds = preds.squeeze(1)  # Remove batch dimension
        
        # Decode
        text = decode_prediction(preds.cpu().numpy())
    
    return text


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/solve', methods=['POST'])
def solve_captcha():
    """
    Solve CAPTCHA from base64 image
    
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
        
        # Decode base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(img_bytes)
            tmp_path = tmp_file.name
        
        # Predict
        text = predict_captcha(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return jsonify({
            'success': True,
            'text': text
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/solve_file', methods=['POST'])
def solve_captcha_file():
    """
    Solve CAPTCHA from uploaded file
    
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
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        # Predict
        text = predict_captcha(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return jsonify({
            'success': True,
            'text': text
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CRNN CAPTCHA Solver - Flask Server")
    print("=" * 70)
    print()
    
    # Load model
    try:
        load_model('crnn_captcha_model.pth')
        print("Server ready!")
        print("=" * 70)
        print()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please train the model first: python train_crnn.py")
        exit(1)
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)
#!/bin/bash
# setup.sh - Complete extension setup

echo "======================================"
echo "CAPTCHA Auto-Solver Extension Setup"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "manifest.json" ]; then
    echo "❌ Error: Please run this script from the extension directory"
    exit 1
fi

echo "✓ Found extension files"
echo ""

# Check for required files
echo "Checking required files..."

if [ -f "vocab.json" ]; then
    echo "✓ vocab.json found"
else
    echo "❌ vocab.json missing - copy from your training folder"
fi

if [ -f "captcha_solver.onnx" ]; then
    echo "✓ captcha_solver.onnx found"
else
    echo "❌ captcha_solver.onnx missing - copy from your training folder"
fi

if [ -f "onnxruntime-web.min.js" ]; then
    echo "✓ onnxruntime-web.min.js found"
else
    echo "❌ onnxruntime-web.min.js missing"
    echo ""
    echo "To download, run ONE of these commands:"
    echo ""
    echo "Method 1 - Using curl:"
    echo "  curl -L -o onnxruntime-web.min.js https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js"
    echo ""
    echo "Method 2 - Using wget:"
    echo "  wget -O onnxruntime-web.min.js https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js"
    echo ""
    echo "Method 3 - Using npm:"
    echo "  npm install onnxruntime-web"
    echo "  cp node_modules/onnxruntime-web/dist/ort.min.js onnxruntime-web.min.js"
    echo ""
fi

echo ""
echo "======================================"
echo "Installation Instructions:"
echo "======================================"
echo ""
echo "1. Complete missing files (see above)"
echo ""
echo "2. Open Chrome browser"
echo ""
echo "3. Go to: chrome://extensions/"
echo ""
echo "4. Enable 'Developer mode' (top right toggle)"
echo ""
echo "5. Click 'Load unpacked'"
echo ""
echo "6. Select this folder: $(pwd)"
echo ""
echo "7. The extension should appear in your extensions list"
echo ""
echo "======================================"
echo "Usage:"
echo "======================================"
echo ""
echo "Automatic:"
echo "  - Extension detects CAPTCHAs automatically"
echo "  - Solves and fills them when found"
echo ""
echo "Manual:"
echo "  - Click extension icon in Chrome toolbar"
echo "  - Click 'Solve CAPTCHA Now' button"
echo ""
echo "======================================"

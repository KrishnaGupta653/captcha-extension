import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class CRNNDataset(Dataset):
    """
    Enhanced Dataset for CRNN training with data augmentation
    Handles variable-length CAPTCHAs with any characters
    """
    
    def __init__(self, image_dir, img_height=64, img_width=200, augment=False):
        """
        Args:
            image_dir: Directory containing CAPTCHA images (can be relative or absolute)
            img_height: Target height for images
            img_width: Target width for images
            augment: Whether to apply data augmentation (use True for training)
        """
        # Convert to absolute path if relative
        self.image_dir = os.path.abspath(image_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        
        # Get all image files
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Build character set from all filenames
        self.chars = self._build_charset()
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        # Add blank character at index 0 for CTC
        self.char_to_idx['<BLANK>'] = 0
        self.idx_to_char[0] = '<BLANK>'
        
        # Data augmentation pipeline (only applied during training)
        if self.augment:
            self.transform = A.Compose([
                # Random brightness and contrast
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                # Random gamma correction
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                # Gaussian noise
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                # Gaussian blur
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                # Motion blur (simulates camera shake)
                A.MotionBlur(blur_limit=3, p=0.2),
                # JPEG compression artifacts
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
                # Random rotation (slight)
                A.Rotate(limit=3, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.3),
                # Perspective transform (slight)
                A.Perspective(scale=(0.02, 0.05), p=0.2),
            ])
        else:
            self.transform = None
        
        print(f"Dataset initialized with {len(self.image_files)} images")
        print(f"Character set ({len(self.chars)} chars): {''.join(sorted(self.chars))}")
        print(f"Augmentation: {'ENABLED' if augment else 'DISABLED'}")
    
    def _build_charset(self):
        """Build character set from all filenames"""
        chars = set()
        for filename in self.image_files:
            label = os.path.splitext(filename)[0]
            label = label.strip('_')
            
            for char in label:
                chars.add(char)
        
        return sorted(list(chars))
    
    def _extract_label(self, filename):
        """Extract label from filename"""
        label = os.path.splitext(filename)[0]
        label = label.strip('_')
        return label
    
    def _preprocess_image(self, image):
        """Enhanced preprocessing with adaptive methods"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Denoise
        image = cv2.fastNlMeansDenoising(image, h=10)
        
        # Binarization with Otsu's method
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return image
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        label_text = self._extract_label(img_name)
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Enhanced preprocessing
        image = self._preprocess_image(image)
        
        # Resize image
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Apply augmentation (only during training)
        if self.augment and self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)
        
        # Convert to tensor
        image = torch.FloatTensor(image)
        
        # Encode label as indices
        label = [self.char_to_idx[char] for char in label_text]
        label_length = len(label)
        
        return image, torch.LongTensor(label), label_length, label_text


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    """
    images, labels, label_lengths, label_texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Concatenate labels
    labels = torch.cat(labels, 0)
    
    # Label lengths
    label_lengths = torch.LongTensor(label_lengths)
    
    return images, labels, label_lengths, label_texts
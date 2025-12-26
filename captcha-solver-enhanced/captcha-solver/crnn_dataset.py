import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CRNNDataset(Dataset):
    """
    Dataset for CRNN training
    Handles variable-length CAPTCHAs with any characters
    """
    
    def __init__(self, image_dir, img_height=64, img_width=200, augment=False):
        """
        Args:
            image_dir: Directory containing CAPTCHA images
            img_height: Target height for images
            img_width: Target width for images
        """
        self.image_dir = os.path.abspath(image_dir)
        self.img_height = img_height
        self.img_width = img_width
        
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
        
        print(f"Dataset initialized with {len(self.image_files)} images")
        print(f"Character set ({len(self.chars)} chars): {''.join(sorted(self.chars))}")
    
    def _build_charset(self):
        """Build character set from all filenames"""
        chars = set()
        for filename in self.image_files:
            # Extract label from filename (remove extension and underscores)
            label = os.path.splitext(filename)[0]
            label = label.strip('_')  # Remove leading/trailing underscores
            
            # Add all characters
            for char in label:
                chars.add(char)
        
        return sorted(list(chars))
    
    def _extract_label(self, filename):
        """Extract label from filename"""
        # Remove extension
        label = os.path.splitext(filename)[0]
        # Remove leading/trailing underscores
        label = label.strip('_')
        return label
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image filename
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Extract label from filename
        label_text = self._extract_label(img_name)
        
        # Load and preprocess image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Resize image
        image = cv2.resize(image, (self.img_width, self.img_height))
        # if self.training:  # Only during training
        # # Random brightness
        #     if np.random.rand() < 0.5:
        #         brightness = np.random.uniform(0.8, 1.2)
        #         image = image * brightness
        #         image = np.clip(image, 0, 255)
            
        #     # Random noise
        #     if np.random.rand() < 0.3:
        #         noise = np.random.normal(0, 5, image.shape)
        #         image = image + noise
        #         image = np.clip(image, 0, 255)
            
        #     # Random blur
        #     if np.random.rand() < 0.2:
        #         image = cv2.GaussianBlur(image, (3, 3), 0)
        
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
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class CRNNDataset(Dataset):
    """
    OPTIMIZED Dataset for 96% → 99% Accuracy
    - FIXED: Matches server preprocessing exactly ([0,1] normalization)
    - KEPT: CAPTCHA-specific augmentations (ElasticTransform)
    - KEPT: Balanced intensity
    """
    
    def __init__(self, image_dir, img_height=64, img_width=200, augment=False):
        self.image_dir = os.path.abspath(image_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        self.chars = self._build_charset()
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.char_to_idx['<BLANK>'] = 0
        self.idx_to_char[0] = '<BLANK>'
        
        # CAPTCHA-Specific Augmentation (KEPT - This is the 99% secret sauce)
        if self.augment:
            self.transform = A.Compose([
                # 1. ELASTIC TRANSFORM - Critical for CAPTCHAs
                A.OneOf([
                    A.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        p=1.0
                    ),
                    A.GridDistortion(
                        num_steps=5, 
                        distort_limit=0.2,
                        p=1.0
                    ),
                    # Optical distortion mimics lens curvature
                    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=1.0),
                ], p=0.7),
                
                # 2. Morphological (font weight variations)
                A.OneOf([
                    A.Morphological(scale=(2, 3), operation='erosion', p=1.0),
                    A.Morphological(scale=(2, 3), operation='dilation', p=1.0),
                ], p=0.25),
                
                # # 3. Light Noise
                # A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),

                # 3. NOISE (Robustness)
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),
                    A.ISONoise(
                        color_shift=(0.01, 0.05), 
                        intensity=(0.1, 0.5), 
                        p=1.0
                    ),
                ], p=0.2),
                
                # 4. Minimal Rotation
                A.Rotate(
                    limit=5,
                    border_mode=cv2.BORDER_CONSTANT, 
                    # value=255, 
                    p=0.4
                ),
                # 5. BRIGHTNESS/CONTRAST - Critical for robustness
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.3),
                # 6. BLUR - Focus variations (15% probability)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.15),
            ])
        else:
            self.transform = None
        
        print(f"Dataset: {len(self.image_files)} images")
        print(f"Charset ({len(self.chars)}): {''.join(sorted(self.chars))}")
        print(f"Augmentation: {'ENABLED (Optimized)' if augment else 'DISABLED'}")
    
    def _build_charset(self):
        chars = set()
        for filename in self.image_files:
            label = os.path.splitext(filename)[0].strip('_')
            for char in label:
                chars.add(char)
        return sorted(list(chars))
    
    def _extract_label(self, filename):
        return os.path.splitext(filename)[0].strip('_')
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_text = self._extract_label(img_name)
        
        # 1. Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        # 2. CRITICAL FIX: Resize FIRST (matches server_crnn.py behavior)
        # Server does: cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # 3. Apply augmentation AFTER resize (geometric transforms stay consistent)
        # if self.augment and self.transform:
        #     augmented = self.transform(image=image)
        #     image = augmented['image']
        # if self.augment and self.transform:
        #     try:
        #         augmented = self.transform(image=image)
        #         image = augmented['image']
        #     except Exception as e:
        #         print(f"⚠️  Augmentation failed on {img_name}: {e}")
        #         pass  # Fallback if augmentation fails
        if self.augment:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            try:
                augmented = self.transform(image=image_rgb)
                image_rgb = augmented['image']
            except Exception as e:
                print(f"⚠️  Augmentation failed on {img_name}: {e}")
            
            # Convert back to grayscale
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # 4. CRITICAL FIX: Normalize to [0, 1] (matches server_crnn.py)
        # Server uses: image.astype(np.float32) * (1.0 / 255.0)
        # My original code used [-1, 1] which would BREAK inference
        image = image.astype(np.float32) / 255.0
        
        # 5. Add channel dimension
        image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)
        
        # 6. Encode label
        label = [self.char_to_idx[char] for char in label_text]
        
        return image, torch.LongTensor(label), len(label), label_text


def collate_fn(batch):
    """Collate variable-length sequences"""
    images, labels, label_lengths, label_texts = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels, 0)
    label_lengths = torch.LongTensor(label_lengths)
    return images, labels, label_lengths, label_texts
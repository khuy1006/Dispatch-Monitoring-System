"""
Data augmentation utilities for image classification.
"""

import cv2
import random
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageFilter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataAugmenter:
    """Handles data augmentation for training images."""
    
    def __init__(self, train_dir: str, classes: List[str]):
        """
        Initialize DataAugmenter.
        
        Args:
            train_dir: Path to training directory
            classes: List of class names
        """
        self.train_dir = Path(train_dir)
        self.classes = classes
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Augmentation methods mapping
        self.augmentation_methods = {
            'flip_horizontal': self._flip_horizontal,
            'flip_vertical': self._flip_vertical,
            'rotate_90_cw': self._rotate_90_cw,
            'rotate_90_ccw': self._rotate_90_ccw,
            'rotate_random': self._rotate_random,
            'blur': self._blur,
            'noise': self._add_noise
        }
    
    def augment_dataset(self, augmentation_types: List[str]) -> bool:
        """
        Apply augmentation to training dataset.
        
        Args:
            augmentation_types: List of augmentation types to apply
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting data augmentation...")
        
        # Validate augmentation types
        invalid_types = [t for t in augmentation_types if t not in self.augmentation_methods]
        if invalid_types:
            logger.error(f"Invalid augmentation types: {invalid_types}")
            return False
        
        try:
            for class_name in self.classes:
                if not self._augment_class(class_name, augmentation_types):
                    return False
                    
            logger.info("✅ Data augmentation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during data augmentation: {e}")
            return False
    
    def _augment_class(self, class_name: str, augmentation_types: List[str]) -> bool:
        """Augment images for a single class."""
        class_dir = self.train_dir / class_name
        
        if not class_dir.exists():
            logger.warning(f"Class directory '{class_name}' not found")
            return True
        
        # Get original images
        original_images = [
            f for f in class_dir.glob("*") 
            if f.suffix.lower() in self.image_extensions
        ]
        
        if not original_images:
            logger.warning(f"No images found in class '{class_name}'")
            return True
        
        logger.info(f"Augmenting class: {class_name}")
        num_original = len(original_images)
        
        # Apply augmentations with progress bar
        for img_path in tqdm(original_images, desc=f"Processing {class_name}"):
            for aug_type in augmentation_types:
                if not self._apply_augmentation(img_path, aug_type):
                    logger.warning(f"Failed to apply {aug_type} to {img_path}")
        
        final_count = len(list(class_dir.glob("*")))
        logger.info(f"Class '{class_name}': {num_original} original -> {final_count} total images")
        
        return True
    
    def _apply_augmentation(self, image_path: Path, aug_type: str) -> bool:
        """Apply a specific augmentation to an image."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return False
            
            # Apply augmentation
            augmented, suffix = self.augmentation_methods[aug_type](image)
            
            if augmented is not None:
                # Save augmented image
                base_name = image_path.stem
                output_path = image_path.parent / f"{base_name}{suffix}{image_path.suffix}"
                return cv2.imwrite(str(output_path), augmented)
                
            return False
            
        except Exception as e:
            logger.error(f"Error applying {aug_type} to {image_path}: {e}")
            return False
    
    # Augmentation methods
    def _flip_horizontal(self, image: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Apply horizontal flip."""
        return cv2.flip(image, 1), "_flip_h"
    
    def _flip_vertical(self, image: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Apply vertical flip."""
        return cv2.flip(image, 0), "_flip_v"
    
    def _rotate_90_cw(self, image: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Rotate 90 degrees clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "_rot_90cw"
    
    def _rotate_90_ccw(self, image: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Rotate 90 degrees counter-clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "_rot_90ccw"
    
    def _rotate_random(self, image: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Apply random rotation (-15 to +15 degrees)."""
        angle = random.uniform(-15, 15)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return rotated, f"_rot_{angle:.1f}"
    
    def _blur(self, image: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Apply Gaussian blur."""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            blur_radius = random.uniform(0.1, 0.6)
            blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            result = cv2.cvtColor(np.array(blurred), cv2.COLOR_RGB2BGR)
            return result, f"_blur_{blur_radius:.1f}"
        except Exception as e:
            logger.warning(f"Blur failed: {e}")
            return None, ""
    
    def _add_noise(self, image: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Add random noise to image."""
        noise_ratio = random.uniform(0.0001, 0.001)
        noisy = image.copy()
        num_pixels = int(image.shape[0] * image.shape[1] * noise_ratio)
        
        for _ in range(num_pixels):
            y = random.randint(0, image.shape[0] - 1)
            x = random.randint(0, image.shape[1] - 1)
            noisy[y, x] = [random.randint(0, 255) for _ in range(3)]
            
        return noisy, f"_noise_{noise_ratio:.4f}" 
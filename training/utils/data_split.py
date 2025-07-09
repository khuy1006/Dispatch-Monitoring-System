"""
Data splitting utilities for classification datasets.
"""

import shutil
import random
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DataSplitter:
    """Handles dataset splitting into train/val/test sets."""
    
    def __init__(self, source_dir: str, dest_dir: str, classes: List[str]):
        """
        Initialize DataSplitter.
        
        Args:
            source_dir: Path to source directory containing class folders
            dest_dir: Path to destination directory for split data
            classes: List of class names
        """
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.classes = classes
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def split_dataset(self, train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._validate_ratios(train_ratio, val_ratio, test_ratio):
            return False
            
        logger.info("Starting dataset splitting...")
        
        try:
            self._create_directory_structure()
            
            for class_name in self.classes:
                if not self._split_class(class_name, train_ratio, val_ratio, test_ratio):
                    return False
                    
            logger.info("✅ Dataset splitting completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during dataset splitting: {e}")
            return False
    
    def _validate_ratios(self, train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
        """Validate that ratios sum to 1.0"""
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            logger.error(f"Ratios must sum to 1.0, got {total}")
            return False
        return True
    
    def _create_directory_structure(self) -> None:
        """Create directory structure for split data."""
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:
                (self.dest_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    def _split_class(self, class_name: str, train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
        """Split files for a single class."""
        class_path = self.source_dir / class_name
        
        if not class_path.exists():
            logger.warning(f"Class directory '{class_name}' not found in source")
            return True  # Continue with other classes
            
        # Get all image files
        image_files = [
            f for f in class_path.glob("*") 
            if f.suffix.lower() in self.image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No image files found in class '{class_name}'")
            return True
            
        random.shuffle(image_files)
        
        # Calculate split indices
        total_files = len(image_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # Split files
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end], 
            'test': image_files[val_end:]
        }
        
        # Copy files to destination
        for split_name, files in splits.items():
            dest_path = self.dest_dir / split_name / class_name
            for file in files:
                try:
                    shutil.copy2(file, dest_path / file.name)
                except Exception as e:
                    logger.error(f"Failed to copy {file}: {e}")
                    return False
        
        logger.info(f"Class '{class_name}': {len(splits['train'])} train, "
                   f"{len(splits['val'])} val, {len(splits['test'])} test")
        
        return True 
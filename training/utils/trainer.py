"""
YOLO model training utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Handles YOLO model training for classification tasks."""
    
    def __init__(self, model_name: str = "yolov8n-cls.pt"):
        """
        Initialize YOLOTrainer.
        
        Args:
            model_name: Name of the YOLO model to use
        """
        self.model_name = model_name
        self.model = None
        
    def train(self, 
              data_path: Union[str, Path],
              epochs: int = 100,
              patience: int = 5,
              device: Union[int, str] = 0,
              batch_size: int = 64,
              **kwargs) -> Optional[object]:
        """
        Train YOLO classification model.
        
        Args:
            data_path: Path to dataset directory
            epochs: Number of training epochs
            patience: Early stopping patience
            device: Device to use (0 for GPU, 'cpu' for CPU)
            batch_size: Training batch size
            **kwargs: Additional training arguments
            
        Returns:
            Training results object or None if failed
        """
        logger.info("Starting YOLO model training...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dataset: {data_path}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Device: {device}")
        
        try:
            # Initialize model
            self.model = YOLO(self.model_name)
            
            # Start training
            results = self.model.train(
                data=str(data_path),
                epochs=epochs,
                patience=patience,
                device=device,
                batch=batch_size,
                **kwargs
            )
            
            logger.info("✅ Model training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during model training: {e}")
            return None
    
    def validate(self, data_path: Union[str, Path]) -> Optional[object]:
        """
        Validate trained model.
        
        Args:
            data_path: Path to validation dataset
            
        Returns:
            Validation results or None if failed
        """
        if self.model is None:
            logger.error("No model available for validation. Train first.")
            return None
            
        try:
            logger.info("Starting model validation...")
            results = self.model.val(data=str(data_path))
            logger.info("✅ Model validation completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during validation: {e}")
            return None
    
    def save_model(self, save_path: Union[str, Path]) -> bool:
        """
        Save trained model.
        
        Args:
            save_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save. Train first.")
            return False
            
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save(str(save_path))
            logger.info(f"✅ Model saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """
        Load pre-trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = YOLO(str(model_path))
            logger.info(f"✅ Model loaded from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
            
        try:
            info = {
                "model_name": self.model_name,
                "task": getattr(self.model, 'task', 'Unknown'),
                "device": getattr(self.model.model, 'device', 'Unknown') if hasattr(self.model, 'model') else 'Unknown',
            }
            return info
            
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {"status": "Error getting model info"} 
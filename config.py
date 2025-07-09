"""
Configuration Management for Dispatch Monitoring System
"""
import os
from pathlib import Path

class Config:
    """Configuration class for the monitoring system"""
    
    def __init__(self):
        # Model paths - check local models first
        self.DETECTION_MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', self._find_detection_model())
        self.CLASSIFICATION_MODEL_PATH = os.getenv('CLASSIFICATION_MODEL_PATH', self._find_classification_model())
        
        # Video capture settings
        self.FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '1280'))
        self.FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', '720'))
        self.FPS = int(os.getenv('FPS', '30'))
        
        # Detection settings
        self.CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        self.CLASSIFICATION_THRESHOLD = float(os.getenv('CLASSIFICATION_THRESHOLD', '0.3'))
        
        # Database settings
        self.DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/monitoring.db')
        
        # Model improvement settings
        self.MIN_FEEDBACK_FOR_RETRAINING = int(os.getenv('MIN_FEEDBACK_FOR_RETRAINING', '100'))
        self.MODEL_BACKUP_ENABLED = os.getenv('MODEL_BACKUP_ENABLED', 'true').lower() == 'true'
        
        # Paths
        self.DATA_DIR = Path('data')
        self.MODELS_DIR = Path('models')
        self.FEEDBACK_DIR = Path('data/feedback')
        
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.FEEDBACK_DIR.mkdir(exist_ok=True)
        
        # Web interface settings
        self.WEB_HOST = os.getenv('WEB_HOST', '0.0.0.0')
        self.WEB_PORT = int(os.getenv('WEB_PORT', '5000'))
        
        # Logging settings
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', 'data/monitoring.log')
        
    def _find_detection_model(self):
        """Find detection model in models directory"""
        models_dir = Path('models')
        
        # Priority list for detection models
        detection_models = [
            'detection.pt'
        ]
        
        for model_name in detection_models:
            model_path = models_dir / model_name
            if model_path.exists():
                print(f"Found detection model: {model_path}")
                return str(model_path)
        
        # Fallback to default YOLOv8n (will auto-download)
        return 'yolo11n.pt'
    
    def _find_classification_model(self):
        """Find classification model in models directory"""
        models_dir = Path('models')
        
        # Priority list for classification models
        classification_models = [
            'classification.pt'
        ]
        
        for model_name in classification_models:
            model_path = models_dir / model_name
            if model_path.exists():
                print(f"Found classification model: {model_path}")
                return str(model_path)
        
        # Fallback to default YOLOv8n-cls (will auto-download)
        return 'yolo11n-cls.pt'
        
    def __repr__(self):
        return f"Config(detection_model={self.DETECTION_MODEL_PATH}, classification_model={self.CLASSIFICATION_MODEL_PATH})"

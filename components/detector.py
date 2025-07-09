"""
Detection Component - Handles YOLO object detection
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    """
    YOLO Object Detection module
    """
    
    def __init__(self, model_path):
        """
        Initialize YOLO detection model
        
        Args:
            model_path (str): Path to detection model
        """
        print("Loading YOLO detection model...")
        
        # Fix for PyTorch 2.7+ weights_only=True issue
        try:
            # Allow ultralytics models to load with weights_only=False
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load with safe globals, trying alternative method: {e}")
            # Fallback: temporarily set weights_only behavior
            old_load = torch.load
            torch.load = lambda *args, **kwargs: old_load(*args, **kwargs, weights_only=False)
            try:
                self.model = YOLO(model_path)
            finally:
                torch.load = old_load
        
        self.names = self.model.names
        print(f"Detection model loaded: {Path(model_path).name}")
        print(f"Detection classes: {list(self.names.values())}")
    
    def detect(self, image, conf_threshold=0.5, track=False):
        """
        Perform object detection on image
        
        Args:
            image: Input image
            conf_threshold (float): Confidence threshold
            track (bool): Enable tracking
            
        Returns:
            Detection results
        """
        if track:
            results = self.model.track(
                image,
                conf=conf_threshold,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml"
            )
        else:
            results = self.model(image, conf=conf_threshold, verbose=False)
        
        return results
    
    def get_detections(self, results):
        """
        Extract detection information from results
        
        Args:
            results: YOLO detection results
            
        Returns:
            tuple: (boxes, confidences, classes, track_ids)
        """
        if not results or len(results) == 0:
            return [], [], [], []
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return [], [], [], []
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Get track IDs if available
        track_ids = None
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        return boxes, confs, classes, track_ids

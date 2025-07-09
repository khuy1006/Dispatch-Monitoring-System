"""
Classification Component - Handles YOLO object classification
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


class YOLOClassifier:
    """
    YOLO Object Classification module
    """
    
    def __init__(self, model_path):
        """
        Initialize YOLO classification model
        
        Args:
            model_path (str): Path to classification model
        """
        print("Loading YOLO classification model...")
        
        # Fix for PyTorch 2.7+ weights_only=True issue
        try:
            # Allow ultralytics models to load with weights_only=False
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.ClassificationModel'])
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
        print(f"Classification model loaded: {Path(model_path).name}")
        print(f"Classification classes: {list(self.names.values())}")
    
    def classify_object(self, image_crop):
        """
        Classify detected object using classification model
        
        Args:
            image_crop: Cropped image of detected object
            
        Returns:
            tuple: (classification_result, confidence)
        """
        try:
            if image_crop.size == 0:
                return "empty_crop", 0.0
                
            results = self.model(image_crop, verbose=False)
            
            if len(results) > 0 and results[0].probs is not None:
                probs = results[0].probs
                top_class_idx = probs.top1
                confidence = probs.top1conf.item()
                class_name = results[0].names[top_class_idx]
                
                return class_name, confidence
            else:
                return "unknown", 0.0
                
        except Exception as e:
            print(f"Classification error: {e}")
            return "error", 0.0
    
    def classify_batch(self, image_crops):
        """
        Classify multiple objects at once
        
        Args:
            image_crops: List of cropped images
            
        Returns:
            list: List of (classification_result, confidence) tuples
        """
        results = []
        for crop in image_crops:
            cls_result, cls_conf = self.classify_object(crop)
            results.append((cls_result, cls_conf))
        return results

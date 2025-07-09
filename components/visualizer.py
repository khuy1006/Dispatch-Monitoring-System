"""
Visualization Component - Handles drawing and display functions
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


class Visualizer:
    """
    Visualization and display module
    """
    
    def __init__(self):
        """
        Initialize visualizer with default colors
        """
        # Colors for different classes
        self.colors = {
            0: (0, 255, 0),     # Green
            1: (255, 0, 0),     # Blue
            2: (0, 0, 255),     # Red
            'unknown': (128, 128, 128)  # Gray
        }
        
        # Track history for trajectories
        self.track_history = defaultdict(list)
    
    def draw_detection(self, frame, box, conf, cls_idx, cls_names, 
                      cls_result=None, cls_conf=None, track_id=None):
        """
        Draw detection box and labels on frame
        
        Args:
            frame: Input frame
            box: Bounding box [x1, y1, x2, y2]
            conf: Detection confidence
            cls_idx: Class index
            cls_names: Dictionary of class names
            cls_result: Classification result (optional)
            cls_conf: Classification confidence (optional)
            track_id: Track ID (optional)
            
        Returns:
            Modified frame
        """
        try:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Get class name safely
            if cls_idx in cls_names:
                detected_class = cls_names[cls_idx]
            else:
                detected_class = f"class_{cls_idx}"
            
            # Get color with enhanced visibility
            color = self.colors.get(cls_idx, self.colors['unknown'])
            # Make colors brighter for better visibility
            color = tuple(min(255, c + 50) for c in color)
            
            # Draw bounding box with thicker lines
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare comprehensive label
            label = f"{detected_class} {conf:.2f}"
            
            if track_id is not None:
                label = f"ID:{track_id} | " + label
            
            if cls_result and cls_conf and cls_conf > 0.3:
                label += f" | {cls_result} {cls_conf:.2f}"
            
            # Draw label background with padding
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label above the box
            label_y = y1 - 10
            if label_y - label_height < 0:  # If too close to top, put it below
                label_y = y2 + label_height + 10
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                         (x1, label_y - label_height - 5), 
                         (x1 + label_width + 10, label_y + 5), 
                         color, -1)
            
            # Draw label text with white color for contrast
            cv2.putText(frame, label, (x1 + 5, label_y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing detection: {e}")
            return frame
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_trajectory(self, frame, track_id, center, cls_idx):
        """
        Draw tracking trajectory
        
        Args:
            frame: Input frame
            track_id: Track ID
            center: Center point (x, y)
            cls_idx: Class index for color
            
        Returns:
            Modified frame
        """
        # Add center to track history
        self.track_history[track_id].append(center)
        
        # Keep only recent history
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id].pop(0)
        
        # Draw trajectory
        if len(self.track_history[track_id]) > 1:
            points = np.array(self.track_history[track_id], dtype=np.int32)
            color = self.colors.get(cls_idx, self.colors['unknown'])
            cv2.polylines(frame, [points], False, color, 2)
        
        return frame

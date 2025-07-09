"""
Dispatch Monitoring System - Main Application
Commercial Kitchen Monitoring with Real-time Detection, Tracking and Classification
"""
import cv2
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from threading import Thread
import queue
import logging

from components.detector import YOLODetector
from components.classifier import YOLOClassifier
from components.visualizer import Visualizer
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/monitoring.log') if Path('data').exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
config = Config()

class DispatchMonitoringSystem:
    """
    Main monitoring system class
    """
    
    def __init__(self):
        """Initialize the monitoring system"""
        self.detector = None
        self.classifier = None
        self.visualizer = Visualizer()
        
        # Video capture
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        self.is_running = False
        self.video_source = None  # Store current video source
        self.is_video_file = False  # Track if source is file or camera
        self.video_paused = False  # Video playback control
        self.video_position = 0  # Current frame position
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'active_tracks': 0,
            'unique_track_ids': set(),
            'detected_classes': {},
            'classification_states': {},
            'classifications_made': 0,
            'feedback_received': 0,
            'uptime_start': datetime.now()
        }
        
        # Detection Log - Track individual objects by ID
        self.detection_log = {}  # {track_id: {detect: class, state: classification, timestamp: time, crop: image}}
        self.log_max_entries = 50  # Keep last 50 detections
        
        # Feedback and training data paths
        self.feedback_data_path = Path('data_wrong')
        self.setup_feedback_folders()
        
        # Auto retrain settings
        self.last_retrain_time = datetime.now()
        self.retrain_interval_hours = 24  # Retrain every 24 hours
        self.min_feedback_samples = 20  # Minimum samples before retraining
        
        # Initialize database
        self.init_database()
        
        # Load models
        self.load_models()
    
    def setup_feedback_folders(self):
        """Setup feedback data folders for training"""
        self.feedback_data_path.mkdir(exist_ok=True)
        
        # Create classification folders based on model classes
        if self.classifier and hasattr(self.classifier, 'names'):
            for class_name in self.classifier.names.values():
                class_folder = self.feedback_data_path / class_name
                class_folder.mkdir(exist_ok=True)
                logger.info(f"Created feedback folder: {class_folder}")
        else:
            # Default folders if model not loaded yet
            for class_name in ['empty', 'kakigori', 'not_empty']:
                class_folder = self.feedback_data_path / class_name
                class_folder.mkdir(exist_ok=True)
                logger.info(f"Created default feedback folder: {class_folder}")

    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create feedback table for training data persistence
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                track_id TEXT,
                original_class TEXT,
                correct_class TEXT,
                confidence REAL,
                image_path TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def load_models(self):
        """Load detection and classification models"""
        try:
            # Load detection model - prioritize local models
            logger.info(f"Looking for detection model: {config.DETECTION_MODEL_PATH}")
            
            try:
                self.detector = YOLODetector(config.DETECTION_MODEL_PATH)
                logger.info(f"Detection model loaded: {config.DETECTION_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load detection model {config.DETECTION_MODEL_PATH}: {e}")
                logger.info("Trying fallback YOLOv8n...")
                self.detector = YOLODetector('yolov8n.pt')  # Will auto-download
                logger.info("Fallback detection model loaded: yolov8n.pt")
            
            # Load classification model - prioritize local models  
            logger.info(f"Looking for classification model: {config.CLASSIFICATION_MODEL_PATH}")
            
            try:
                self.classifier = YOLOClassifier(config.CLASSIFICATION_MODEL_PATH)
                logger.info(f"Classification model loaded: {config.CLASSIFICATION_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load classification model {config.CLASSIFICATION_MODEL_PATH}: {e}")
                logger.info("Trying fallback YOLOv8n-cls...")
                self.classifier = YOLOClassifier('yolov8n-cls.pt')  # Will auto-download
                logger.info("Fallback classification model loaded: yolov8n-cls.pt")
                
            # Log model information
            if self.detector:
                logger.info(f"Detection classes: {list(self.detector.names.values())}")
            if self.classifier:
                logger.info(f"Classification classes: {list(self.classifier.names.values())}")
                
        except Exception as e:
            logger.error(f"Critical error loading models: {e}")
            logger.warning("Running without AI models - only video streaming will work")
    
    def start_video_capture(self, source=0):
        """Start video capture from camera or video file"""
        try:
            # Stop any existing capture
            if self.cap:
                self.stop()
                time.sleep(1)
            
            # Handle different source types
            if isinstance(source, str):
                # Check if it's a file path
                if source.isdigit():
                    source = int(source)  # Convert string number to int for camera
                    self.is_video_file = False
                elif not source.startswith(('http://', 'https://', 'rtsp://')):
                    # It's a file path, check if it exists
                    if not Path(source).exists():
                        logger.error(f"Video file not found: {source}")
                        return False
                    logger.info(f"Loading video file: {source}")
                    self.is_video_file = True
                else:
                    # Stream URL
                    self.is_video_file = False
            else:
                # Camera index
                self.is_video_file = False
                
            self.video_source = source
            
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                return False
            
            # For video files, get the actual properties
            if isinstance(source, str) and not source.isdigit() and not source.startswith(('http://', 'https://', 'rtsp://')):
                # Video file - get original properties
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                logger.info(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
            else:
                # Camera - set properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
            
            self.is_running = True
            
            # Start capture thread
            capture_thread = Thread(target=self._capture_frames, daemon=True)
            capture_thread.start()
            
            # Start processing thread
            process_thread = Thread(target=self._process_frames, daemon=True)
            process_thread.start()
            
            logger.info(f"Video capture started from source: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video capture: {e}")
            return False
    
    def _capture_frames(self):
        """Capture frames from video source"""
        frame_delay = 0.033  # Default delay for 30 FPS
        
        # Adjust delay based on video FPS if available
        if self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_delay = 1.0 / fps
        
        while self.is_running and self.cap:
            # Check if video is paused
            if self.video_paused:
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    # For video files, stop when video ends
                    logger.info("Video file ended, stopping playback")
                    self.is_running = False
                    break
                else:
                    # For camera/stream, try to reconnect
                    logger.warning("Failed to read frame from camera/stream")
                    time.sleep(1)
                    continue
            
            # Update video position for files
            if self.is_video_file:
                self.video_position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Add frame to queue if not full
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())  # Make a copy to avoid memory issues
            
            # Control playback speed for video files
            if self.is_video_file:
                time.sleep(frame_delay)
            else:
                time.sleep(max(0.033, frame_delay))  # Limit camera FPS
    
    def _process_frames(self):
        """Process frames for detection and tracking"""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    processed_frame = self.process_frame(frame)
                    self.latest_frame = processed_frame
                else:
                    time.sleep(0.01)  # Small delay if no frames
                    
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
    
    def process_frame(self, frame):
        """Process a single frame"""
        if frame is None:
            return frame
            
        # Create a copy to avoid modifying original
        processed_frame = frame.copy()
        
        # Add status indicator even if no models
        height, width = processed_frame.shape[:2]
        
        if self.detector is None:
            # No detector loaded - show warning
            cv2.putText(processed_frame, "No Detection Model Loaded", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(processed_frame, "Loading models... Please wait", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return processed_frame
        
        try:
            # Detect objects
            results = self.detector.detect(processed_frame, 
                                         conf_threshold=config.CONFIDENCE_THRESHOLD,
                                         track=True)
            
            # Get detection data
            boxes, confs, classes, track_ids = self.detector.get_detections(results)
            
            # Update statistics
            self.stats['total_detections'] += len(boxes)
            
            # Update active tracks and unique IDs
            if track_ids is not None:
                active_track_ids = set(track_ids)
                self.stats['active_tracks'] = len(active_track_ids)
                self.stats['unique_track_ids'].update(active_track_ids)
            else:
                self.stats['active_tracks'] = 0
            
            # Update detected classes count
            for cls_idx in classes:
                class_name = self.detector.names.get(cls_idx, f"class_{cls_idx}")
                self.stats['detected_classes'][class_name] = self.stats['detected_classes'].get(class_name, 0) + 1
            
            # Process each detection
            for i, (box, conf, cls_idx) in enumerate(zip(boxes, confs, classes)):
                track_id = track_ids[i] if track_ids is not None else None
                
                # Extract crop for classification
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 > x1 and y2 > y1:  # Valid box
                    crop = processed_frame[y1:y2, x1:x2]
                    
                    cls_result, cls_conf = None, None
                    if self.classifier and crop.size > 0:
                        cls_result, cls_conf = self.classifier.classify_object(crop)
                        if cls_conf > config.CLASSIFICATION_THRESHOLD:
                            self.stats['classifications_made'] += 1
                            # Update classification states count
                            if cls_result:
                                self.stats['classification_states'][cls_result] = self.stats['classification_states'].get(cls_result, 0) + 1
                    
                    # Save detection log for ID tracking
                    if track_id is not None:
                        detection_class = self.detector.names.get(cls_idx, f"class_{cls_idx}")
                        self.update_detection_log(track_id, detection_class, cls_result, crop, conf, cls_conf)
                    
                    # Draw detection with visualization
                    processed_frame = self.visualizer.draw_detection(
                        processed_frame, box, conf, cls_idx, self.detector.names,
                        cls_result, cls_conf, track_id
                    )
                    
                    # Draw trajectory
                    if track_id is not None:
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        processed_frame = self.visualizer.draw_trajectory(
                            processed_frame, track_id, center, cls_idx)
            
            # Add comprehensive status overlay
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Background for text - make it larger for more info
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (5, 5), (width-5, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
            
            # Status text
            cv2.putText(processed_frame, f"Time: {timestamp}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detection and tracking info
            detection_info = f"Current: {len(boxes)} detections | Active Tracks: {self.stats['active_tracks']}"
            cv2.putText(processed_frame, detection_info, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Unique tracking IDs count
            unique_ids_info = f"Total Unique IDs: {len(self.stats['unique_track_ids'])}"
            cv2.putText(processed_frame, unique_ids_info, (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Model status
            model_info = f"Detection: {'ON' if self.detector else 'OFF'} | Classification: {'ON' if self.classifier else 'OFF'}"
            cv2.putText(processed_frame, model_info, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Classes and states summary
            classes_summary = f"Classes: {len(self.stats['detected_classes'])} | States: {len(self.stats['classification_states'])}"
            cv2.putText(processed_frame, classes_summary, (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Total statistics
            total_info = f"Total: {self.stats['total_detections']} detections, {self.stats['classifications_made']} classifications"
            cv2.putText(processed_frame, total_info, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            # Return frame with error message
            cv2.putText(processed_frame, f"Processing Error: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return processed_frame
    
    def stop(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Monitoring system stopped")
    
    def save_feedback_with_image(self, track_id, correct_class):
        """Save feedback and cropped image for training"""
        if track_id not in self.detection_log:
            return False, "Track ID not found in detection log"
        
        detection_data = self.detection_log[track_id]
        crop_image = detection_data.get('crop')
        
        if crop_image is None:
            return False, "No image data available for this ID"
        
        try:
            # Create filename with timestamp and ID
            timestamp = detection_data['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"id_{track_id}_{timestamp}.jpg"
            
            # Save to correct class folder
            class_folder = self.feedback_data_path / correct_class
            class_folder.mkdir(exist_ok=True)
            
            image_path = class_folder / filename
            cv2.imwrite(str(image_path), crop_image)
            
            # Update detection log
            self.detection_log[track_id]['corrected'] = True
            self.detection_log[track_id]['correct_class'] = correct_class
            
            # Update stats
            self.stats['feedback_received'] += 1
            
            logger.info(f"Saved feedback image: {image_path}")
            
            # Check if we should retrain
            self.check_and_retrain()
            
            return True, f"Image saved to {image_path}"
            
        except Exception as e:
            logger.error(f"Error saving feedback image: {e}")
            return False, str(e)
    
    def check_and_retrain(self):
        """Check if conditions are met for retraining"""
        try:
            # Count total feedback samples
            total_samples = 0
            for class_folder in self.feedback_data_path.iterdir():
                if class_folder.is_dir():
                    total_samples += len(list(class_folder.glob('*.jpg')))
            
            # Check time since last retrain
            time_since_retrain = datetime.now() - self.last_retrain_time
            hours_since_retrain = time_since_retrain.total_seconds() / 3600
            
            if (total_samples >= self.min_feedback_samples and 
                hours_since_retrain >= self.retrain_interval_hours):
                
                logger.info(f"Triggering retrain: {total_samples} samples, {hours_since_retrain:.1f} hours since last retrain")
                self.trigger_retrain()
                
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
    
    def trigger_retrain(self):
        """Trigger model retraining with feedback data"""
        try:
            logger.info("=== AUTO RETRAIN TRIGGERED ===")
            logger.info(f"Training data path: {self.feedback_data_path}")
            
            # Count samples per class
            for class_folder in self.feedback_data_path.iterdir():
                if class_folder.is_dir():
                    sample_count = len(list(class_folder.glob('*.jpg')))
                    logger.info(f"Class '{class_folder.name}': {sample_count} samples")
            
            # Update last retrain time
            self.last_retrain_time = datetime.now()
            
            # NOTE: Integration point for actual model training
            # Example: subprocess.run(['python', 'train_yolo_cls.py', '--data', str(self.feedback_data_path)])
            
            logger.info("=== RETRAIN COMPLETED ===")
            
        except Exception as e:
            logger.error(f"Error during retrain: {e}")

    def pause_video(self):
        """Pause video playback (only for video files)"""
        if self.is_video_file:
            self.video_paused = True
            logger.info("Video paused")
            return True
        return False
    
    def resume_video(self):
        """Resume video playback (only for video files)"""
        if self.is_video_file:
            self.video_paused = False
            logger.info("Video resumed")
            return True
        return False
    
    def seek_video(self, frame_number):
        """Seek to specific frame (only for video files)"""
        if self.is_video_file and self.cap:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = max(0, min(frame_number, total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.video_position = frame_number
            logger.info(f"Seeked to frame {frame_number}")
            return True
        return False
    
    def get_video_info(self):
        """Get video information"""
        if not self.cap:
            return None
            
        info = {
            'is_video_file': self.is_video_file,
            'is_paused': self.video_paused,
            'current_frame': self.video_position if self.is_video_file else 0,
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_video_file else 0,
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        if self.is_video_file and info['total_frames'] > 0 and info['fps'] > 0:
            info['duration_seconds'] = info['total_frames'] / info['fps']
            info['current_time'] = info['current_frame'] / info['fps']
        else:
            info['duration_seconds'] = 0
            info['current_time'] = 0
            
        return info
    
    def update_detection_log(self, track_id, detection_class, classification_state, crop_image, det_conf, cls_conf):
        """Update detection log for tracking individual objects"""
        current_time = datetime.now()
        
        # Update or create log entry
        self.detection_log[track_id] = {
            'detect': detection_class,
            'state': classification_state or 'unknown',
            'timestamp': current_time,
            'crop': crop_image.copy() if crop_image is not None else None,
            'det_confidence': float(det_conf) if det_conf is not None else 0.0,
            'cls_confidence': float(cls_conf) if cls_conf is not None else 0.0,
            'corrected': False  # Flag for user feedback
        }
        
        # Limit log size - keep only recent entries
        if len(self.detection_log) > self.log_max_entries:
            # Remove oldest entries
            oldest_ids = sorted(self.detection_log.keys(), 
                              key=lambda x: self.detection_log[x]['timestamp'])[:5]
            for old_id in oldest_ids:
                del self.detection_log[old_id]
    
    def get_detection_log(self, limit=20):
        """Get recent detection log entries"""
        # Sort by timestamp, most recent first
        sorted_entries = sorted(
            self.detection_log.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Format for API response
        log_data = []
        for track_id, data in sorted_entries[:limit]:
            # Convert numpy types to native Python types for JSON serialization
            det_conf = data.get('det_confidence', 0)
            cls_conf = data.get('cls_confidence', 0)
            
            log_data.append({
                'id': int(track_id),
                'detect': str(data['detect']),
                'state': str(data['state']),
                'timestamp': data['timestamp'].strftime('%H:%M:%S'),
                'det_confidence': round(float(det_conf), 2) if det_conf is not None else 0.0,
                'cls_confidence': round(float(cls_conf), 2) if cls_conf is not None else 0.0,
                'corrected': bool(data.get('corrected', False))
            })
        
        return log_data

# Global monitoring system instance
monitoring_system = DispatchMonitoringSystem()

# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            if monitoring_system.latest_frame is not None:
                try:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', monitoring_system.latest_frame, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        logger.warning("Failed to encode frame")
                except Exception as e:
                    logger.error(f"Error encoding frame: {e}")
            else:
                # Send a placeholder frame when no video is available
                placeholder = create_placeholder_frame()
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def create_placeholder_frame():
    """Create a placeholder frame when no video is available"""
    import numpy as np
    
    # Create a black frame with text
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    text = "No Video Signal"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    
    # Get text size and center it
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    uptime = datetime.now() - monitoring_system.stats['uptime_start']
    stats = monitoring_system.stats.copy()
    
    # Convert set to count for JSON serialization
    stats['unique_track_ids_count'] = len(stats['unique_track_ids'])
    del stats['unique_track_ids']  # Remove set (not JSON serializable)
    
    stats['uptime_seconds'] = int(uptime.total_seconds())
    stats['uptime_formatted'] = str(uptime).split('.')[0]  # Remove microseconds
    
    return jsonify(stats)

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring with specified source"""
    try:
        data = request.json or {}
        source = data.get('source', 0)  # Default to webcam
        
        success = monitoring_system.start_video_capture(source)
        
        if success:
            return jsonify({'status': 'success', 'message': 'Monitoring started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start monitoring'}), 500
            
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    try:
        monitoring_system.stop()
        return jsonify({'status': 'success', 'message': 'Monitoring stopped'})
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/video/info')
def get_video_info():
    """Get video information"""
    try:
        info = monitoring_system.get_video_info()
        if info:
            return jsonify({'status': 'success', 'data': info})
        else:
            return jsonify({'status': 'error', 'message': 'No video source active'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/video/pause', methods=['POST'])
def pause_video():
    """Pause video playback"""
    try:
        success = monitoring_system.pause_video()
        if success:
            return jsonify({'status': 'success', 'message': 'Video paused'})
        else:
            return jsonify({'status': 'error', 'message': 'Cannot pause (not a video file)'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/video/resume', methods=['POST'])
def resume_video():
    """Resume video playback"""
    try:
        success = monitoring_system.resume_video()
        if success:
            return jsonify({'status': 'success', 'message': 'Video resumed'})
        else:
            return jsonify({'status': 'error', 'message': 'Cannot resume (not a video file)'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/video/seek', methods=['POST'])
def seek_video():
    """Seek to specific frame"""
    try:
        data = request.json
        frame_number = data.get('frame', 0)
        success = monitoring_system.seek_video(frame_number)
        if success:
            return jsonify({'status': 'success', 'message': f'Seeked to frame {frame_number}'})
        else:
            return jsonify({'status': 'error', 'message': 'Cannot seek (not a video file)'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detection_log')
def get_detection_log():
    """Get detection log"""
    try:
        log_data = monitoring_system.get_detection_log()
        return jsonify({'status': 'success', 'data': log_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/feedback_correction', methods=['POST'])
def submit_feedback_correction():
    """Submit feedback correction for specific track ID"""
    try:
        data = request.json
        track_id = data.get('track_id')
        correct_class = data.get('correct_class')
        
        if not track_id or not correct_class:
            return jsonify({'status': 'error', 'message': 'Missing track_id or correct_class'}), 400
        
        success, message = monitoring_system.save_feedback_with_image(track_id, correct_class)
        
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

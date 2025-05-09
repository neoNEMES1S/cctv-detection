import os
import cv2
import torch
import logging
import requests
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple
import warnings
import numpy as np
import time

# ------------------------------
# CONFIGURATION & LOGGING
# ------------------------------

warnings.filterwarnings('ignore')

class Config:
    VIDEO_PATH: str = "95.mp4"
    OUTPUT_DIR: str = "output_clips"
    FRAME_SKIP: int = 5  # Reduced from 15 to catch more frames
    DETECTION_CLASSES: List[str] = [
        'person', 'truck', 'car', 'bus', 'motorcycle', 
        # Include more vehicle types to catch ram raids
    ]
    CLIP_DURATION_SECONDS: int = 10
    FPS: int = 30
    UPLOAD_URL: Optional[str] = "https://your-api-endpoint.com/upload"
    YOLO_MODEL: str = 'yolov5m'  # Use medium model for better accuracy
    CONFIDENCE_THRESHOLD: float = 0.15  # Lowered from 0.25 for better detection
    
    # Add time-based detection parameters
    SUSPICIOUS_HOUR_START: int = 22
    SUSPICIOUS_HOUR_END: int = 5
    
    # Add specific scenarios
    SUSPICIOUS_SCENARIOS = {
        'multiple_people_night': {'min_people': 1, 'confidence': 0.2},  # Lowered thresholds
        'vehicle_near_atm': {
            'vehicle_types': ['truck', 'car', 'van'],
            'confidence': 0.2
        },
        'masked_person': {'confidence': 0.2},
        'tool_carrier': {
            'person_with_objects': ['backpack', 'suitcase', 'handbag'],
            'confidence': 0.2
        }
    }

    # Performance optimization
    PROCESSING_RESOLUTION: Tuple[int, int] = (640, 480)

    # Add ATM-specific scenarios
    ATM_SCENARIOS = {
        'ram_raid': {
            'required_object': 'car',
            'movement_threshold': 30,  # Lowered from 50
            'confidence': 0.2
        },
        # 'multiple_people_atm': {
        #     'min_people': 1,  # Even a single person at night is suspicious
        #     'confidence': 0.4
        # }
    }

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------------
# THREAT DETECTION MODULE
# ------------------------------

class ThreatDetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.previous_frame = None
        self.previous_detections = []
        self.motion_threshold = 2000  # Lowered from 5000 for more sensitive motion detection
    
    def _load_model(self):
        logger.info("Loading YOLO model...")
        model = torch.hub.load('ultralytics/yolov5', self.config.YOLO_MODEL, pretrained=True)
        model.conf = self.config.CONFIDENCE_THRESHOLD
        return model

    def detect_threat(self, frame) -> Tuple[bool, np.ndarray, str]:
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error(f"Invalid frame: {type(frame)}")
            return False, frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8), ""
            
        start_time = time.time()
        
        try:
            # Check for sudden motion first (ram raid detection)
            motion_detected, motion_score = self._detect_motion(frame)
            
            # Resize frame for faster processing
            h, w = frame.shape[:2]
            new_w, new_h = self.config.PROCESSING_RESOLUTION
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            # Convert BGR to RGB for the model
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Process frame with model
            results = self.model(frame_rgb)
            detections = results.pandas().xyxy[0]
            
            # Log all detections for debugging
            logger.info(f"Detections: {detections[['name', 'confidence']].to_dict(orient='records')}")
            
            # Create a copy of frame for annotation
            annotated_frame = frame.copy()
            
            # If significant motion was detected, check for vehicles
            if motion_detected:
                vehicles = detections[detections['name'].isin(['car', 'truck', 'bus'])]
                if not vehicles.empty:
                    self._draw_detections(annotated_frame, detections)
                    logger.warning(f"RAM RAID DETECTED - Motion score: {motion_score}")
                    return True, annotated_frame, "ram_raid_atm"
            
            # Check for other suspicious scenarios
            threat_detected, threat_type = self._analyze_scenarios(detections)
            
            if threat_detected:
                self._draw_detections(annotated_frame, detections)
                logger.warning(f"Suspicious activity ({threat_type}) detected")
                return True, annotated_frame, threat_type
                
            process_time = time.time() - start_time
            logger.info(f"Frame processing time: {process_time:.3f} seconds")
            
            return False, frame, ""
            
        except Exception as e:
            logger.error(f"Error in detect_threat: {e}")
            return False, frame, ""
    
    def _detect_motion(self, frame) -> Tuple[bool, float]:
        """Detect sudden large motion like a vehicle impact"""
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False, 0
            
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray, self.previous_frame)
        
        # Apply threshold
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion score (sum of all pixel differences)
        motion_score = np.sum(thresh) / 255
        
        # Update previous frame
        self.previous_frame = gray
        
        # Check if motion exceeds threshold
        significant_motion = motion_score > self.motion_threshold
        
        if significant_motion:
            logger.info(f"Significant motion detected: {motion_score}")
        
        return significant_motion, motion_score

    def _draw_detections(self, frame, detections):
        for _, detection in detections.iterrows():
            if detection['confidence'] > self.config.CONFIDENCE_THRESHOLD:
                # Extract coordinates and class
                x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
                class_name = detection['name']
                confidence = detection['confidence']

                # Draw box
                color = (0, 255, 0) if class_name == 'person' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add label with confidence
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _analyze_scenarios(self, detections) -> Tuple[bool, str]:
        # # Check for any people near ATM (simplified)
        # people = detections[detections['name'] == 'person']
        # if not people.empty:
        #     return True, "person_at_atm"
        
        # Check for vehicles near ATM
        vehicles = detections[detections['name'].isin(['car', 'truck', 'bus'])]
        if not vehicles.empty:
            return True, "vehicle_near_atm"
            
        return False, ""

# ------------------------------
# VIDEO PROCESSING MODULE
# ------------------------------

class VideoProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.cap = cv2.VideoCapture(config.VIDEO_PATH)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {config.VIDEO_PATH}")
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.buffer = deque(maxlen=config.CLIP_DURATION_SECONDS * config.FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0

    def read_next_frame(self):
        # Read a single frame
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Add frame to buffer regardless of skipping
        self.buffer.append(frame)
        
        # Increment counter and skip frames as needed
        self.frame_count += 1
        if self.frame_count % self.config.FRAME_SKIP != 0:
            return None
            
        # We're processing this frame, return it
        return frame

    def save_clip(self, clip_index: int, threat_type: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"threat_{threat_type}_{timestamp}_{clip_index}.mp4"
        output_path = os.path.join(self.config.OUTPUT_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.config.FPS, 
                              (self.frame_width, self.frame_height))
        
        for frame in self.buffer:
            # Add timestamp overlay
            cv2.putText(frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Add threat type overlay
            cv2.putText(frame, f"Threat: {threat_type}",
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            writer.write(frame)
            
        writer.release()
        logger.info(f"Saved suspicious clip to {output_path}")
        return output_path

    def release(self):
        self.cap.release()

# ------------------------------
# MAIN FUNCTION
# ------------------------------

def main():
    config = Config()
    detector = ThreatDetector(config)
    processor = VideoProcessor(config)
    
    clip_index = 0
    start_time = time.time()
    logger.info("Beginning video analysis...")

    try:
        while True:
            # Get next frame to process
            frame = processor.read_next_frame()
            
            # Check if we reached the end
            if frame is None:
                if processor.cap.get(cv2.CAP_PROP_POS_FRAMES) >= processor.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    logger.info("End of video stream.")
                    break
                continue
                
            # Detect threats with motion detection
            is_threat, annotated_frame, threat_type = detector.detect_threat(frame)
            
            # Update buffer with annotated frame if there was a threat
            if is_threat:
                processor.buffer[-1] = annotated_frame  
                clip_path = processor.save_clip(clip_index, threat_type)
                clip_index += 1

    except Exception as ex:
        logger.exception(f"Unhandled exception: {ex}")
    finally:
        processor.release()
        total_time = time.time() - start_time
        logger.info(f"Processing complete. Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
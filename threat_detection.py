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
import torch.amp as amp
import numpy as np
import time

# ------------------------------
# CONFIGURATION & LOGGING
# ------------------------------

warnings.filterwarnings('ignore')

class Config:
    VIDEO_PATH: str = "2.mp4"
    OUTPUT_DIR: str = "output_clips"
    FRAME_SKIP: int = 90  # Increased to process even fewer frames
    # Enhanced detection classes for ATM/Bank scenarios
    # Reduced list to only essential classes
    DETECTION_CLASSES: List[str] = [
        'person',        # To detect presence of people
        'truck',         # For vehicle-based ATM theft
        'car',          # Getaway vehicles
        'backpack',     # Potential tool carriers
        'suitcase',     # Suspicious large containers
        'handbag',      # Potential tool carriers
        'scissors',     # Common tampering tool
        'knife',        # Potential tampering tool
        'screwdriver',  # Common tampering tool
        'laptop',       # Potential skimming device
        'cell phone',   # Potential recording device
    ]
    CLIP_DURATION_SECONDS: int = 15  # Reduced for faster processing
    FPS: int = 30
    UPLOAD_URL: Optional[str] = "https://your-api-endpoint.com/upload"
    YOLO_MODEL: str = 'yolov5s'  # Using small model for better accuracy
    CONFIDENCE_THRESHOLD: float = 0.3  # Increased to reduce false positives
    
    # Add time-based detection parameters
    SUSPICIOUS_HOUR_START: int = 22  # 10 PM
    SUSPICIOUS_HOUR_END: int = 5     # 5 AM
    
    # Add specific scenarios - reduced to key scenarios only
    SUSPICIOUS_SCENARIOS = {
        'tool_carrier': {
            'person_with_objects': ['backpack', 'suitcase', 'handbag'],
            'confidence': 0.3
        },
        'atm_tampering': {
            'required_objects': ['person'],
            'suspicious_tools': ['scissors', 'knife', 'screwdriver'],
            'min_tools': 1,
            'confidence': 0.3,
            'max_distance': 300,  # Increased distance threshold
            'min_duration': 2  # Reduced minimum seconds of suspicious activity
        },
        'manual_atm_tampering': {
            'detection_zone': {
                'top_percent': 0.3,      # Top 30% of frame is ATM zone
                'bottom_percent': 0.7,   # Bottom 70% of frame is person zone
                'left_percent': 0.1,     # Left margin
                'right_percent': 0.9,    # Right margin
            },
            'hand_in_atm_time': 1,       # Reduced time requirement
            'confidence': 0.25,
            'min_duration': 1           # Reduced time requirement
        }
    }

    # Add performance optimization
    PROCESSING_RESOLUTION: Tuple[int, int] = (320, 240)  # Reduced resolution for faster processing
    ENABLE_BATCH_PROCESSING: bool = False
    ENABLE_CUDA: bool = True
    BATCH_SIZE: int = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ------------------------------
# THREAT DETECTION MODULE
# ------------------------------

class ThreatDetector:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Enable GPU if available
        if self.config.ENABLE_CUDA and torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()  # Set to evaluation mode
        self.previous_detections = []
        self.current_threat_type = None  # Store the type of threat detected
    
    def _load_model(self):
        logger.info("Loading YOLO model...")
        model = torch.hub.load('ultralytics/yolov5', self.config.YOLO_MODEL, pretrained=True)
        model.conf = self.config.CONFIDENCE_THRESHOLD
        return model

    def detect_threat(self, frame) -> Tuple[bool, np.ndarray, str]:
        # Track processing time
        start_time = time.time()
        
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, self.config.PROCESSING_RESOLUTION)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Process frame
        with torch.no_grad():  # Disable gradient calculation for inference
            results = self.model(frame_rgb)
        
        detections = results.pandas().xyxy[0]
        
        # Log processing time
        process_time = time.time() - start_time
        logger.info(f"Frame processing time: {process_time:.3f} seconds")
        
        # Create a copy of frame for annotation
        annotated_frame = frame.copy()
        
        # Check for suspicious scenarios and annotate frame
        threat_detected, threat_type = self._analyze_scenarios(detections)
        
        if threat_detected:
            # Annotate detected objects
            self._draw_detections(annotated_frame, detections)
            self.current_threat_type = threat_type
            logger.warning(f"Suspicious activity ({threat_type}) detected at {datetime.now()}")
            return True, annotated_frame, threat_type
            
        return False, frame, ""
    
    def _draw_detections(self, frame, detections):
        for _, detection in detections.iterrows():
            if detection['confidence'] > self.config.CONFIDENCE_THRESHOLD:
                # Extract coordinates and class
                x1, y1, x2, y2 = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
                class_name = detection['name']
                confidence = detection['confidence']

                # Draw box
                color = (0, 255, 0) if class_name == 'person' else (0, 0, 255)  # Green for people, Red for objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add label with confidence
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _analyze_scenarios(self, detections) -> Tuple[bool, str]:
        # Simplified detection history tracking
        if not hasattr(self, 'detection_history'):
            self.detection_history = {
                'atm_tampering': {'start_time': None, 'count': 0},
                'manual_atm_tampering': {'start_time': None, 'count': 0, 'hand_in_zone_start': None}
            }

        current_time = time.time()
        # Get frame dimensions for zone calculations
        frame_height, frame_width = 240, 320  # Default from PROCESSING_RESOLUTION

        # Quick exit if no people detected
        people = detections[detections['name'] == 'person']
        if people.empty:
            return False, ""
            
        # 1. Person with suspicious objects - quick check
        suspicious_objects = detections[detections['name'].isin(self.config.SUSPICIOUS_SCENARIOS['tool_carrier']['person_with_objects'])]
        if not suspicious_objects.empty:
            return True, "suspicious_objects"

        # 2. ATM Tampering Detection - only check if relevant tools detected
        tools = detections[detections['name'].isin(self.config.SUSPICIOUS_SCENARIOS['atm_tampering']['suspicious_tools'])]
        
        if not tools.empty:
            # Check if any person is close to any tool
            for _, person in people.iterrows():
                person_center = ((person['xmin'] + person['xmax']) / 2, (person['ymin'] + person['ymax']) / 2)
                
                for _, tool in tools.iterrows():
                    tool_center = ((tool['xmin'] + tool['xmax']) / 2, (tool['ymin'] + tool['ymax']) / 2)
                    
                    # Calculate distance between person and tool
                    distance = np.sqrt((person_center[0] - tool_center[0])**2 + (person_center[1] - tool_center[1])**2)
                    
                    if distance <= self.config.SUSPICIOUS_SCENARIOS['atm_tampering']['max_distance']:
                        # Update detection history
                        if self.detection_history['atm_tampering']['start_time'] is None:
                            self.detection_history['atm_tampering']['start_time'] = current_time
                        self.detection_history['atm_tampering']['count'] += 1
                        
                        # Check if the suspicious activity has lasted long enough
                        duration = current_time - self.detection_history['atm_tampering']['start_time']
                        if duration >= self.config.SUSPICIOUS_SCENARIOS['atm_tampering']['min_duration']:
                            return True, "atm_tampering"
                    else:
                        # Reset detection history if distance is too large
                        self.detection_history['atm_tampering'] = {'start_time': None, 'count': 0}

        # 3. Manual ATM Tampering Detection (no tools required)
        # Define ATM zone
        zone = self.config.SUSPICIOUS_SCENARIOS['manual_atm_tampering']['detection_zone']
        atm_zone_top = int(frame_height * zone['top_percent'])
        atm_zone_bottom = int(frame_height * zone['bottom_percent'])
        atm_zone_left = int(frame_width * zone['left_percent'])
        atm_zone_right = int(frame_width * zone['right_percent'])
        
        for _, person in people.iterrows():
            # Check if person's hands (upper part of bounding box) are in ATM zone
            person_hands_y = person['ymin'] + (person['ymax'] - person['ymin']) * 0.3  # Approximate hand position
            person_center_x = (person['xmin'] + person['xmax']) / 2
            
            hands_in_atm_zone = (
                person_hands_y <= atm_zone_bottom and 
                person_hands_y >= atm_zone_top and
                person_center_x >= atm_zone_left and 
                person_center_x <= atm_zone_right
            )
            
            if hands_in_atm_zone:
                # Check if this is the first time hands are detected in zone
                if self.detection_history['manual_atm_tampering']['hand_in_zone_start'] is None:
                    self.detection_history['manual_atm_tampering']['hand_in_zone_start'] = current_time
                
                # Start tracking suspicious behavior if hands have been in ATM zone long enough
                hands_duration = current_time - self.detection_history['manual_atm_tampering']['hand_in_zone_start']
                if hands_duration >= self.config.SUSPICIOUS_SCENARIOS['manual_atm_tampering']['hand_in_atm_time']:
                    if self.detection_history['manual_atm_tampering']['start_time'] is None:
                        self.detection_history['manual_atm_tampering']['start_time'] = current_time
                    
                    self.detection_history['manual_atm_tampering']['count'] += 1
                    duration = current_time - self.detection_history['manual_atm_tampering']['start_time']
                    
                    if duration >= self.config.SUSPICIOUS_SCENARIOS['manual_atm_tampering']['min_duration']:
                        return True, "manual_atm_tampering"
            else:
                # Reset hand zone tracking if hands are no longer in zone
                self.detection_history['manual_atm_tampering']['hand_in_zone_start'] = None
        
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
        
        # Get video properties
        self.original_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Calculate actual frame skip based on original video FPS
        self.effective_frame_skip = int(self.original_fps / self.config.FPS)
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        # Increase buffer size to capture more context before and after the event
        self.buffer = deque(maxlen=config.CLIP_DURATION_SECONDS * self.original_fps * 2)  # Double the buffer size
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Add pre-buffer to store frames before detection
        self.pre_buffer = deque(maxlen=self.original_fps * 5)  # Store 5 seconds before detection

    def read_next_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def save_clip(self, clip_index: int, threat_type: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"threat_{threat_type}_{timestamp}_{clip_index}.mp4"
        output_path = os.path.join(self.config.OUTPUT_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.original_fps,
                               (self.frame_width, self.frame_height))
        
        # First write pre-buffer frames
        for frame in self.pre_buffer:
            writer.write(frame)
        
        # Then write main buffer frames
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
# UPLOADER MODULE
# ------------------------------

class Uploader:
    def __init__(self, config: Config):
        self.config = config

    def upload_clip(self, filepath: str) -> bool:
        if not self.config.UPLOAD_URL:
            logger.info("Upload URL not provided, skipping upload.")
            return False

        try:
            with open(filepath, 'rb') as f:
                files = {'file': (Path(filepath).name, f, 'video/mp4')}
                response = requests.post(self.config.UPLOAD_URL, files=files)
                if response.status_code == 200:
                    logger.info(f"Successfully uploaded {filepath}")
                    return True
                else:
                    logger.warning(f"Failed to upload {filepath}: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.exception(f"Exception during upload: {e}")
            return False


# ------------------------------
# MAIN ORCHESTRATOR
# ------------------------------

def main():
    config = Config()
    detector = ThreatDetector(config)
    processor = VideoProcessor(config)
    uploader = Uploader(config)

    frame_count = 0
    clip_index = 0
    last_frame_time = time.time()
    threat_detected = False
    threat_start_time = None
    last_threat_time = None
    threat_type = None
    consecutive_threat_frames = 0
    post_threat_frames = 0

    logger.info("Beginning video analysis...")
    start_time = time.time()

    try:
        while True:
            frame = processor.read_next_frame()
            if frame is None:
                # Save any remaining threat footage before ending
                if threat_detected and processor.buffer:
                    clip_path = processor.save_clip(clip_index, threat_type or "suspicious_activity")
                    uploader.upload_clip(clip_path)
                logger.info(f"End of video stream. Total processing time: {time.time() - start_time:.2f} seconds")
                break

            frame_count += 1
            
            # Skip frames to improve performance
            if frame_count % config.FRAME_SKIP != 0:
                # Still add to buffer but don't process
                processor.buffer.append(frame.copy())
                continue

            # Smaller pre-buffer for efficiency
            if len(processor.pre_buffer) < processor.pre_buffer.maxlen:
                processor.pre_buffer.append(frame.copy())
            
            # Add to main buffer only if we're close to detecting a threat or already detected one
            if threat_detected or consecutive_threat_frames > 0:
                processor.buffer.append(frame.copy())

            # Process frame for threat detection
            is_threat, annotated_frame, current_threat_type = detector.detect_threat(frame)
            
            if is_threat:
                if not threat_detected:
                    threat_detected = True
                    threat_start_time = time.time()
                    threat_type = current_threat_type
                    logger.warning(f"Threat detected: {threat_type}")
                    # Start buffering frames
                    processor.buffer.append(frame.copy())
                
                last_threat_time = time.time()
                consecutive_threat_frames += 1
                
                # Replace the last frame in buffer with annotated version
                if processor.buffer:
                    processor.buffer[-1] = annotated_frame
                
                # Save clip after sufficient confirmation
                if consecutive_threat_frames >= 2:
                    clip_path = processor.save_clip(clip_index, threat_type)
                    uploader.upload_clip(clip_path)
                    processor.buffer.clear()
                    processor.pre_buffer.clear()
                    clip_index += 1
                    threat_detected = False
                    threat_start_time = None
                    consecutive_threat_frames = 0
            else:
                # If we haven't seen a threat for more than 2 seconds, reset the threat state
                if threat_detected and last_threat_time and (time.time() - last_threat_time) > 2:
                    # Save the clip before resetting
                    if processor.buffer:
                        clip_path = processor.save_clip(clip_index, threat_type)
                        uploader.upload_clip(clip_path)
                        processor.buffer.clear()
                        processor.pre_buffer.clear()
                        clip_index += 1
                    
                    threat_detected = False
                    threat_start_time = None
                    consecutive_threat_frames = 0

            # Progress reporting
            if frame_count % 300 == 0:  # Report every ~300 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({fps:.2f} fps)")

    except Exception as ex:
        logger.exception(f"Unhandled exception: {ex}")

    finally:
        processor.release()
        total_time = time.time() - start_time
        logger.info(f"Processing complete. Total time: {total_time:.2f} seconds for {frame_count} frames.")
        if frame_count > 0 and total_time > 0:
            logger.info(f"Average processing rate: {frame_count/total_time:.2f} fps")


if __name__ == "__main__":
    main()


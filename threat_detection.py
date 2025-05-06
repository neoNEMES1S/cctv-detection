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
    VIDEO_PATH: str = "94.mp4"
    OUTPUT_DIR: str = "output_clips"
    FRAME_SKIP: int = 60  # Process one frame every 2 seconds
    # Enhanced detection classes for ATM/Bank scenarios
    DETECTION_CLASSES: List[str] = [
        'person',        # To detect presence of people
        'truck',         # For vehicle-based ATM theft
        'car',          # Getaway vehicles
        'backpack',     # Potential tool carriers
        'suitcase',     # Suspicious large containers
        'handbag',      # Potential tool carriers
        'mask',         # Face coverings
        'tie',          # To detect formal wear (to spot unusual formal wear at odd hours)
    ]
    CLIP_DURATION_SECONDS: int = 10  # Increased to capture more context
    FPS: int = 30
    UPLOAD_URL: Optional[str] = "https://your-api-endpoint.com/upload"
    YOLO_MODEL: str = 'yolov5s'  # Using small model for better accuracy
    CONFIDENCE_THRESHOLD: float = 0.3  # Lowered threshold for better detection
    
    # Add time-based detection parameters
    SUSPICIOUS_HOUR_START: int = 22  # 10 PM
    SUSPICIOUS_HOUR_END: int = 5     # 5 AM
    
    # Add specific scenarios
    SUSPICIOUS_SCENARIOS = {
        'multiple_people_night': {
            'min_people': 2,
            'time_range': 'night',
            'confidence': 0.4
        },
        'vehicle_near_atm': {
            'distance_threshold': 100,  # pixels
            'vehicle_types': ['truck', 'car', 'van'],
            'confidence': 0.45
        },
        'masked_person': {
            'required_objects': ['person', 'mask'],
            'confidence': 0.5
        },
        'tool_carrier': {
            'person_with_objects': ['backpack', 'suitcase', 'handbag'],
            'confidence': 0.4
        }
    }

    # Add performance optimization
    PROCESSING_RESOLUTION: Tuple[int, int] = (480, 360)  # Even smaller resolution
    ENABLE_BATCH_PROCESSING: bool = False  # Disable batch processing for faster single frame analysis
    ENABLE_CUDA: bool = True  # Enable GPU if available
    BATCH_SIZE: int = 10  # Added batch size

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
        # 1. Multiple people at night
        if datetime.now().hour >= self.config.SUSPICIOUS_HOUR_START or            datetime.now().hour <= self.config.SUSPICIOUS_HOUR_END:
            people_count = len(detections[detections['name'] == 'person'])
            if people_count >= self.config.SUSPICIOUS_SCENARIOS['multiple_people_night']['min_people']:
                return True, "multiple_people_night"
        
        # 2. Vehicle near ATM
        vehicles = detections[detections['name'].isin(self.config.SUSPICIOUS_SCENARIOS['vehicle_near_atm']['vehicle_types'])]
        if not vehicles.empty:
            return True, "vehicle_near_atm"
        
        # 3. Masked person detection
        masked_people = set(detections[detections['name'] == 'person'].index) & set(detections[detections['name'] == 'mask'].index)
        if masked_people:
            return True, "masked_person"
            
        # 4. Person with suspicious objects
        people = detections[detections['name'] == 'person']
        suspicious_objects = detections[detections['name'].isin(self.config.SUSPICIOUS_SCENARIOS['tool_carrier']['person_with_objects'])]
        if not people.empty and not suspicious_objects.empty:
            return True, "suspicious_objects"
            
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
        self.effective_frame_skip = self.original_fps  # Skip frames to process 1 FPS
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.buffer = deque(maxlen=config.CLIP_DURATION_SECONDS * config.FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_next_frame(self) -> Optional[Tuple[bool, any]]:
        frames = []
        for _ in range(self.config.BATCH_SIZE):
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
        return frames if frames else None

    def save_clip(self, clip_index: int, threat_type: str) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Include threat type in filename
        filename = f"threat_{threat_type}_{timestamp}_{clip_index}.mp4"
        output_path = os.path.join(self.config.OUTPUT_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.config.FPS, 
                               (self.frame_width, self.frame_height))
        
        # Add timestamp and threat type as overlay
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

    logger.info("Beginning video analysis...")

    try:
        while True:
            frame = processor.read_next_frame()
            if frame is None:
                logger.info("End of video stream.")
                break

            frame_count += 1
            if frame_count % config.FRAME_SKIP != 0:
                continue

            is_threat, annotated_frame, threat_type = detector.detect_threat(frame)
            processor.buffer[-1] = annotated_frame  # Replace the last frame with annotated one

            if is_threat:
                clip_path = processor.save_clip(clip_index, threat_type)
                uploader.upload_clip(clip_path)
                processor.buffer.clear()
                clip_index += 1

    except Exception as ex:
        logger.exception(f"Unhandled exception: {ex}")

    finally:
        processor.release()
        logger.info("Processing complete.")


if __name__ == "__main__":
    main()


import os
import cv2
import torch
import logging
import requests
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

# ------------------------------
# CONFIGURATION & LOGGING
# ------------------------------

class Config:
    VIDEO_PATH: str = "video_input.mp4"
    OUTPUT_DIR: str = "output_clips"
    FRAME_SKIP: int = 5
    DETECTION_CLASSES: List[str] = ['knife', 'gun']
    CLIP_DURATION_SECONDS: int = 5
    FPS: int = 30  # Set manually or auto-inferred
    UPLOAD_URL: Optional[str] = "https://your-api-endpoint.com/upload"
    YOLO_MODEL: str = 'yolov5n'
    CONFIDENCE_THRESHOLD: float = 0.5

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
    
    def _load_model(self):
        logger.info("Loading YOLO model...")
        model = torch.hub.load('ultralytics/yolov5', self.config.YOLO_MODEL, pretrained=True)
        model.conf = self.config.CONFIDENCE_THRESHOLD
        return model

    def detect_threat(self, frame) -> bool:
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        threats = detections[detections['name'].isin(self.config.DETECTION_CLASSES)]
        if not threats.empty:
            logger.warning(f"Threat detected: {threats[['name', 'confidence']].to_dict(orient='records')}")
            return True
        return False


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

    def read_next_frame(self) -> Optional[Tuple[bool, any]]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.buffer.append(frame)
        return frame

    def save_clip(self, clip_index: int) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"suspicious_clip_{timestamp}_{clip_index}.mp4"
        output_path = os.path.join(self.config.OUTPUT_DIR, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.config.FPS, (self.frame_width, self.frame_height))
        for frame in self.buffer:
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

            if detector.detect_threat(frame):
                clip_path = processor.save_clip(clip_index)
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


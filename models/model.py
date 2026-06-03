# Independent instance of YOLOv11n for inference
# The upload to GCP storage bucket should be integrated into both
#   client and server scripts using ZeroMQ

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
from google.cloud import storage

# Google Cloud service account configuration
GCP_KEY_PATH = "configs/key.json"
BUCKET_NAME = "cs131-detections"


def init_gcs_bucket():
    # Authenticates and returns bucket object
    storage_client = storage.Client.from_service_account_json(GCP_KEY_PATH)
    return storage_client.bucket(BUCKET_NAME)


def upload_to_gcs(bucket, frame, filename):
    # Encodes an OpenCV frame in-memory and uploads it directly to storage
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print(f"Failed to encode {filename}")
        return

    blob = bucket.blob(filename)
    blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")
    print(f"Uploaded: {filename}")


# Inference on YOLOv11n
MODEL_PATH = Path(__file__).parents[1] / "runs/detect/train-9/weights/best.pt"

# Initialize connection to storage bucket
bucket = init_gcs_bucket()

model = YOLO(MODEL_PATH)

results = model.predict(
    source=0,  # Logitech Brio 101 on /dev/video0
    show=True,  # Set to False to prevent X11 display errors on headless Jetsons
    device=0,  # GPU found on device can be enabled with CUDA on Orin Nano
    half=True,  # Model quantized to FP16 precision
    conf=0.5,  # Confidence level
    iou=0.4,  # Intersection on union threshold
    stream=True,
)

# Identify localised objects based on labeled bounding boxes
for idx, r in enumerate(results):
    names = [r.names[int(c)] for c in r.boxes.cls]

    # If the list 'names' is not empty, something was detected
    if names:
        print(f"Detected: {names}")

        # r.plot() returns the numpy array of the image with the bounding boxes drawn on it
        annotated_frame = r.plot()

        # Generate a unique filename using a timestamp and the frame index
        timestamp = int(time.time())
        filename = f"detections/frame_{timestamp}_{idx}.jpg"

        # Upload the frame to Google Cloud
        upload_to_gcs(bucket, annotated_frame, filename)

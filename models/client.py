import cv2
import json
import time
from pathlib import Path
import threading
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import zmq
from google.cloud import storage

# Google Cloud service account configuration
GCP_KEY_PATH = "configs/keys.json"
BUCKET_NAME = "cs131-detections"


def init_gcs_bucket():
    storage_client = storage.Client.from_service_account_json(GCP_KEY_PATH)
    return storage_client.bucket(BUCKET_NAME)


def upload_to_gcs(bucket, frame, filename):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return
    blob = bucket.blob(filename)
    blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")
    print(f"Uploaded to GCS: {filename}")


# Initialize bucket connection
bucket = init_gcs_bucket()

model_path = Path(__file__).parents[1] / "runs/detect/train-9/weights/best.pt"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at: {model_path}")
model = YOLO(str(model_path))

# --- camera: cap resolution + 1-frame buffer so V4L2 doesn't queue frames ---
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

TARGET_FPS = 12
FRAME_INTERVAL = 1.0 / TARGET_FPS

nano1_ip = input("Enter Nano 1's IP address: ")
context = zmq.Context()


def make_socket(context, ip):
    """PAIR socket with CONFLATE so only the newest msg is ever queued each way.
    CONFLATE must be set before connect()."""
    socket = context.socket(zmq.PAIR)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 500)
    socket.setsockopt(zmq.SNDTIMEO, 500)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{ip}:5555")
    return socket


def get_best_detection(results):
    best, best_conf = None, 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = {"label": model.names[int(box.cls[0])], "confidence": conf}
    return best


socket = make_socket(context, nano1_ip)
fail_count = 0

while True:
    loop_start = time.time()

    ret, frame = camera.read()
    if not ret:
        time.sleep(0.01)
        continue

    results = model(frame, imgsz=640, half=True, verbose=False)
    my_detection = get_best_detection(results)

    annotated = results[0].plot()
    cv2.imshow(
        "Nano 2 - Live Detection", annotated
    )  # needs a display; remove if headless
    cv2.waitKey(1)

    # receive Nano 1's latest (CONFLATE -> always newest, never a backlog)
    # On the server and client when testing with Orin Nanos, ZeroMQ is broken
    #   when Google Cloud upload per detection is integrate. Only detections from
    #   the Nano running the client are looked at
    try:
        other_detection = json.loads(socket.recv_string())
        fail_count = 0
    except zmq.Again:
        other_detection = None
        fail_count += 1
        if fail_count >= 20:  # ~10s of silence: reconnect
            socket.close(0)
            socket = make_socket(context, nano1_ip)
            fail_count = 0

    # always push our latest detection up, even if the recv missed this round
    try:
        socket.send_string(json.dumps(my_detection or {}))
    except zmq.Again:
        pass

    my_conf = my_detection["confidence"] if my_detection else 0.0
    other_conf = other_detection.get("confidence", 0.0) if other_detection else 0.0

    if my_conf == 0.0 and other_conf == 0.0:
        final = "No produce detected"
    elif my_conf >= other_conf:
        final = f"[Nano 1 wins] {my_detection['label']} ({my_conf:.2f})"

        # Save to GCP storage if Nano 1 wins the consensus
        timestamp = int(time.time())
        filename = f"detections/nano1_frame_{timestamp}.jpg"
        upload_to_gcs(bucket, annotated, filename)
    else:
        final = f"[Nano 2 wins] {other_detection['label']} ({other_conf:.2f})"

    # throttle to TARGET_FPS
    elapsed = time.time() - loop_start
    if elapsed < FRAME_INTERVAL:
        time.sleep(FRAME_INTERVAL - elapsed)

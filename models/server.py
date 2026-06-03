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

# camera: cap resolution + 1-frame buffer so V4L2 doesn't queue frames ---
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

TARGET_FPS = 12  # throttle inference so the Orin isn't pegged
FRAME_INTERVAL = 1.0 / TARGET_FPS

app = Flask(__name__)
latest_frame = None
latest_result = "Waiting for detection..."
frame_lock = threading.Lock()

HTML = """
<!DOCTYPE html><html>
<head>
  <title>Produce Detector</title>
  <script>
    setInterval(() => {
      fetch('/result')
        .then(r => r.json())
        .then(data => {
          document.getElementById('result').innerText = data.final;
        });
    }, 1000);
  </script>
</head>
<body style="background:#111;color:white;text-align:center;font-family:Arial">
  <h2>Live Produce Detection - Nano 1</h2>
  <img src="/video" width="800"/>
  <h3 id="result" style="color:#00ff99;margin-top:20px;">Waiting for detection...</h3>
</body></html>
"""


def get_best_detection(results):
    best, best_conf = None, 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = {"label": model.names[int(box.cls[0])], "confidence": conf}
    return best


def make_socket(context):
    # PAIR socket with CONFLATE so only the newest msg is ever queued each way.
    # CONFLATE must be set before bind().
    socket = context.socket(zmq.PAIR)
    socket.setsockopt(zmq.CONFLATE, 1)  # no backlog can build -> no desync
    socket.setsockopt(zmq.RCVTIMEO, 500)
    socket.setsockopt(zmq.SNDTIMEO, 500)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind("tcp://*:5555")
    return socket


def zmq_loop():
    global latest_frame, latest_result

    context = zmq.Context()
    socket = make_socket(context)
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
        with frame_lock:
            latest_frame = annotated

        # exchange latest detection; CONFLATE guarantees no pileup ---
        other_detection = None
        try:
            socket.send_string(json.dumps(my_detection or {}))
            other_detection = json.loads(socket.recv_string())
            fail_count = 0
        except zmq.Again:
            other_detection = None  # Nano 2 quiet this round -> go solo
            fail_count += 1
            if fail_count >= 20:  # ~20s of silence: rebuild the pair
                socket.close(0)
                socket = make_socket(context)
                fail_count = 0

        my_conf = my_detection["confidence"] if my_detection else 0.0
        other_conf = other_detection.get("confidence", 0.0) if other_detection else 0.0

        if other_detection is None:
            result = (
                f"[Nano 1 only] {my_detection['label']} ({my_conf:.2f})"
                if my_detection
                else "No produce detected"
            )
        elif my_conf == 0.0 and other_conf == 0.0:
            result = "No produce detected"
        elif my_conf >= other_conf:
            result = f"[Nano 1 wins] {my_detection['label']} ({my_conf:.2f})"

            # Save to cloud bucket if Nano 1 wins the consensus
            timestamp = int(time.time())
            filename = f"detections/nano1_frame_{timestamp}.jpg"
            upload_to_gcs(bucket, annotated, filename)
        else:
            result = f"[Nano 2 wins] {other_detection['label']} ({other_conf:.2f})"

        with frame_lock:
            latest_result = result

        # throttle to TARGET_FPS
        elapsed = time.time() - loop_start
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)


def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/result")
def result():
    with frame_lock:
        res = latest_result
    return json.dumps({"final": res})


if __name__ == "__main__":
    threading.Thread(target=zmq_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)

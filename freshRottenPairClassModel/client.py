import zmq
import json
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train-9/weights/best.pt")      #!!!replace with model path
camera = cv2.VideoCapture(0)

nano1_ip = input("Enter Nano 1's IP address: ")
NANO1_IP = f"tcp://{nano1_ip}:5555"

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect(NANO1_IP)

print("Nano 2 connected to Nano 1")

def get_best_detection(results):
    best = None
    best_conf = 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = {
                    "label": model.names[int(box.cls[0])],
                    "confidence": conf
                }
    return best

while True:
    ret, frame = camera.read()
    if not ret:
        continue

    # Receive Nano 1's result first
    msg = socket.recv_string()
    other_detection = json.loads(msg)

    # Run Nano 2's own inference
    results = model(frame, verbose=False)
    my_detection = get_best_detection(results)

    # Send Nano 2's result back
    socket.send_string(json.dumps(my_detection or {}))

    # --- Consensus logic (same as Nano 1) ---
    my_conf = my_detection["confidence"] if my_detection else 0.0
    other_conf = other_detection.get("confidence", 0.0)

    if my_conf == 0.0 and other_conf == 0.0:
        final = "No produce detected"
    elif my_conf >= other_conf:
        final = f"[Nano 2 have higher confidence] {my_detection['label']} ({my_conf:.2f})"
    else:
        final = f"[Nano 1 have higher confidence] {other_detection['label']} ({other_conf:.2f})"

    print(f"Final result: {final}")


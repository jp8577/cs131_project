from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "runs/detect/train-4/weights/best.pt"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=0,       # Logitech Brio 101 on /dev/video0
    show=True,
    device=0,       # CUDA GPU 0 on Jetson Orin
    half=True,      # FP16 — faster on Jetson Orin's GPU
    conf=0.5,
    iou=0.4,
    stream=True,
)

for r in results:
    names = [r.names[int(c)] for c in r.boxes.cls]
    if names:
        print(names)
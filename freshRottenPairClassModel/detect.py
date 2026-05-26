from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "runs/detect/train-4/weights/best.pt"

model = YOLO(MODEL_PATH)

results = model.predict(
    source=0,
    show=True,
    device="mps",
    conf=0.5,
    iou=0.4,
    max_det=50,
    imgsz=640,
    stream=True,
)

for r in results:
    names = [r.names[int(c)] for c in r.boxes.cls]
    if names:
        print(names)
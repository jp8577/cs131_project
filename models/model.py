from ultralytics import YOLO
import kagglehub

# Load a pretrained YOLOv11n model (model weights from 100 epochs trained on LVIS)
model = YOLO("models/best.pt")

# Set confidence and iou (intersection on union) threshold
conf = 0.6
iou = 0.1
# show_results = False

# Run batched inference on a list of images
results = model.predict(
    source=0, show=True, conf=conf, iou=iou, stream=True, device="mps", half=True
)  # return a list of Results objects

# Save results and show detections (inference not configured for saving results or annotating bounding boxes)
for r in results:
    # r.show()
    pass
    # # save locally
    # result.save(filename=f'Example_Results/{os.path.basename(result.path)}.png')

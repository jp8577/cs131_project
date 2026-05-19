from ultralytics import YOLO
import kagglehub
import os
import yaml
from IPython.display import Image, display

# Load dataset (without annotations) from Kaggle
path = kagglehub.dataset_download("muhriddinmuxiddinov/fruits-and-vegetables-dataset")

# Load a pretrained YOLOv8m model (model weights)
model = YOLO('models/yolo_fruits_and_vegetables_v1.pt')

# Set confidence and iou threshold
conf = 0.6
iou = 0.1
# show_results = False

# Run batched inference on a list of images
results = model.predict(source=0,show=True,conf=0.6,iou=0.1,stream=True)  # return a list of Results objects

# if show_results:
#     for result in results:
#         result.show()

# create a directory to save the results
# if not os.path.exists('Example_Results'):
#     os.makedirs('Example_Results')

# Save results and show detections
for r in results:
    r.show()
    # # save locally
    # result.save(filename=f'Example_Results/{os.path.basename(result.path)}.png')

# Show images

# for i in range(2):
#     display(Image(f'Example_Results/{os.path.basename(images[i])}.png', width=200))

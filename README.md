<img width="1920" height="1280" alt="val_batch2_pred" src="https://github.com/user-attachments/assets/30c0e940-917e-4797-8b3c-04d261d65a38" />


# cs131_project

Group Member: 

Jake Wang

Junwen Yu

Sweden Agunenye

Jooahn Park

## ✏️ Use Case

As a food processing or distribution center operator, I would like to automatically identify produce type and detect whether fruits or vegetables are fresh or defective, so I can remove unsafe or low-quality produce faster, more consistently, and with less reliance on manual inspection.

As a business owner, I would like to collect and analyze produce quality data over time, so I can identify patterns related to suppliers, storage, transportation, or processing issues.

---

## 📍 Purpose for This Project

The purpose of this project is to address the problem of slow and inconsistent manual produce inspection in the Food & Beverage industry, specifically in food processing and distribution centers.

Currently, many facilities rely on human workers to inspect produce before it reaches the market. This creates several challenges:

- Human judgment can vary between workers.
- Manual inspection is slower and harder to scale.
- Produce can vary widely in color, shape, size, and texture.
- Defects such as discoloration, scratches, dents, mold, or rotten areas can be difficult to detect accurately.
- Large processing centers may handle hundreds of thousands of items, requiring fast real-time detection.

This project proposes an intelligent produce inspection system that uses cameras, edge computing, and cloud computing to conduct real-time image classification and defect detection.

---

## 🪜 Task Distribution

### 🖥️ Edge Layer

The edge layer uses two NVIDIA Jetson Nano devices connected to cameras mounted around a conveyor belt. One camera can be placed above the produce, while another camera can be placed on the side to capture multiple angles.

The edge devices are responsible for:

- Capturing live images of produce.
- Running a lightweight CNN model locally.
- Identifying the type of produce.
- Detecting whether the produce is fresh or defective.
- Making real-time decisions without depending on the cloud.
- Sending a signal to a robotic arm to remove defective produce.
- Forwarding captured images and detection metadata to the fog layer.

This allows the system to respond quickly with low latency, which is important for real-time conveyor belt inspection.

### ☁️ Cloud Layer

The cloud layer is rperesented by Google Cloud. We used a VM instance via the Compute Engine API that utilized a NVIDIA T4 GPU. Alongside a storage bucket, the cloud layer is collectively responsible for long-term storage, model training, and improvements on inference through hyperparameter tuning during retraining on new images of produce.

The cloud layer is also used for:

- Storing historical detection results.
- Sending updated model versions back to the edge devices.
- Supporting long-term analysis of produce quality trends.

In the future,  data can help businesses identify quality problems related to suppliers, storage conditions, or transportation.

---

## 🎥 Final Demo

For the final demo, we will build a proof-of-concept version of the produce inspection system.

The demo will show:

- Two cameras capturing live images of fruits or vegetables.
- Dual lightweight convolutional neural network (CNN) models (trained on YOLOv11n) running on two edge devices (NVIDIA Jetson Orin).
- The system classifying the type of produce.
- The system detecting whether the produce is fresh or defective.
- A live output showing the prediction result, such as `Fresh Apple`, `Rotten Banana`, or `Defective Tomato`.
- A cloud-hosted version of the model that represents how retraining and updates on weights would work in the full-scale system.

Physically, the demo can be shown by placing produce in front of the camera or moving it through a small mock conveyor setup. The camera will capture the produce, the model will process the image, and the result will be displayed in real time.

This proof of concept demonstrates how the full system could be used in a real food processing center to automatically inspect produce and remove defective items before they reach consumers.

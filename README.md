# cs131_project

Group Member: 

Jake Wang

Junwen Yu

Sweden Agunenye

Jooahn Park

## Use Case

As a food processing or distribution center operator, I would like to automatically identify produce type and detect whether fruits or vegetables are fresh or defective, so I can remove unsafe or low-quality produce faster, more consistently, and with less reliance on manual inspection.

As a business owner, I would like to collect and analyze produce quality data over time, so I can identify patterns related to suppliers, storage, transportation, or processing issues.

---

## Purpose for This Project

The purpose of this project is to address the problem of slow and inconsistent manual produce inspection in the Food & Beverage industry, specifically in food processing and distribution centers.

Currently, many facilities rely on human workers to inspect produce before it reaches the market. This creates several challenges:

- Human judgment can vary between workers.
- Manual inspection is slower and harder to scale.
- Produce can vary widely in color, shape, size, and texture.
- Defects such as discoloration, scratches, dents, mold, or rotten areas can be difficult to detect accurately.
- Large processing centers may handle hundreds of thousands of items, requiring fast real-time detection.

This project proposes an intelligent produce inspection system that uses cameras, edge computing, fog computing, and cloud computing to classify produce and detect defects in real time.

---

## Task Distribution

### Edge Layer

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

### Fog Layer

The fog layer uses a local gateway or on-site server located near the processing area.

The fog layer is responsible for:

- Receiving detection data from multiple edge devices.
- Filtering unnecessary or duplicate data.
- Aggregating useful image and detection results.
- Coordinating data from multiple camera angles.
- Reducing bandwidth usage before sending data to the cloud.
- Preventing storage and network bottlenecks.

This layer acts as the middle layer between the edge devices and the cloud.

### Cloud Layer

The cloud layer is hosted on AWS and is responsible for long-term storage, training, and system improvement.

The cloud layer is responsible for:

- Storing historical detection results.
- Training a larger and more advanced model using collected data.
- Retraining the model with new produce images over time.
- Improving classification and defect detection accuracy.
- Sending updated model versions back to the edge devices.
- Supporting long-term analysis of produce quality trends.

In the future, this data can help businesses identify quality problems related to suppliers, storage conditions, or transportation.

---

## Final Demo

For the final demo, we will build a proof-of-concept version of the produce inspection system.

The demo will show:

- A camera capturing live images of fruits or vegetables.
- A lightweight CNN model running on an edge device.
- The system classifying the type of produce.
- The system detecting whether the produce is fresh or defective.
- A live output showing the prediction result, such as `Fresh Apple`, `Rotten Banana`, or `Defective Tomato`.
- A cloud-hosted version of the model that represents how retraining and updates would work in the full-scale system.

Physically, the demo can be shown by placing produce in front of the camera or moving it through a small mock conveyor setup. The camera will capture the produce, the model will process the image, and the result will be displayed in real time.

This proof of concept demonstrates how the full system could be used in a real food processing center to automatically inspect produce and remove defective items before they reach consumers.

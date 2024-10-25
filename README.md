# YOLO Object Detection Streamlit App

This is a simple object detection app built using the YOLOv8 model and Streamlit. It allows users to detect objects in real-time using a webcam or by uploading images. The app processes the input image or video, detects objects, and displays both the bounding boxes and class names of the detected objects.

## Features
- **Webcam Object Detection**: Stream real-time video from your webcam and detect objects.
- **Image Upload Detection**: Upload an image and detect objects present in the image.
- **YOLOv8 Model**: Utilizes the lightweight YOLOv8 model for fast object detection.
- **Detected Object Information**: Displays the detected objects with bounding box coordinates, class name, and confidence score.

## Demo
![YOLO Object Detection App](demo_screenshot.png)

## How to Run
```bash
1. Clone the Repository

2. Install the Requirements
Make sure you have Python 3.10 or higher installed. Then, install the required Python packages by running:
pip install -r requirements.txt

3. Run the Streamlit App
You can run the app using Streamlit:

streamlit run app.py
This will start the app locally, and you can access it via your browser at http://localhost:8501.

Requirements
Python 3.10+
OpenCV
Streamlit
Pillow (for image handling)
YOLOv8 (via the ultralytics package)
Numpy
All necessary packages are included in the requirements.txt file.

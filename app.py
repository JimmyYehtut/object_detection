import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can replace with other YOLO models like yolov8m.pt or yolov8s.pt

st.title("YOLOv8 Object Detection: Laptop Webcam and Mobile Camera")

st.write("""
    ### Instructions:
    1. **For Laptop Webcam**: Select the "Laptop Webcam" option and allow access to your webcam.
    2. **For Mobile Camera**: Select the "Mobile Phone Camera" option, open the app on your phone via your local network (IP address), and allow access to your phone's camera.
    3. The video stream will be processed using YOLOv8 for object detection.
""")

# Toggle switch to choose between laptop webcam or mobile camera
camera_option = st.selectbox(
    "Select Camera Source", 
    ("Laptop Webcam", "Mobile Phone Camera")
)

class YOLOObjectDetection(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection using YOLO
        results = self.model(img)

        # Annotate the frame with detected bounding boxes
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].numpy().astype(int)
                confidence = box.conf[0]
                cls = int(box.cls[0])

                # Label: class name and confidence score
                label = f"{self.model.names[cls]} {confidence:.2f}"

                # Draw bounding boxes and label
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video based on the selected camera option
if camera_option == "Laptop Webcam":
    st.write("Using Laptop Webcam")
    webrtc_streamer(
        key="laptop-webcam",
        video_transformer_factory=YOLOObjectDetection,
        media_stream_constraints={"video": True, "audio": False}
    )
else:
    st.write("Using Mobile Phone Camera")
    webrtc_streamer(
        key="mobile-camera",
        video_transformer_factory=YOLOObjectDetection,
        media_stream_constraints={"video": True, "audio": False}
    )

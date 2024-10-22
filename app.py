import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import torch
from ultralytics import YOLO

# Load YOLOv5 model (automatically downloads if not available)
model = YOLO('yolov8n.pt')  # Use a small YOLOv8 model for speed

st.title("Object Detection")

# Sidebar for selecting the source of input
st.sidebar.title("Input Options")
source_option = st.sidebar.selectbox("Choose input type", ("Webcam", "Upload Image"))
stop_webcam = st.sidebar.button("Stop Yolo Webcam",key="stop_yolo_webcam")
# Function to detect objects and display results
def detect_objects(image):
    # Convert the image to a format suitable for YOLO
    img_array = np.array(image)
    results = model.predict(img_array)  # Detect objects in the image
    
    # Convert results to OpenCV format for displaying boxes
    result_img = results[0].plot()
    
    prediction_data = []
    for result in results[0].boxes.data.tolist():
        # result = [x_min, y_min, x_max, y_max, confidence, class]
        x_min, y_min, x_max, y_max, confidence, class_id = result[:6]
        class_name = model.names[int(class_id)]
        prediction_data.append(
            {
                "class": class_name,
                "confidence": confidence,
                "box": [x_min, y_min, x_max, y_max],
            }
        )


    return result_img, prediction_data
def main():
    # Webcam input
    if source_option == "Webcam":
        st.write("Using webcam for object detection.")
        
        # Open the webcam feed using OpenCV
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam.")
        
        # Stream the webcam feed
        FRAME_WINDOW = st.image([])
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Error reading webcam feed.")
                break
            
            # Convert the frame from BGR to RGB (Streamlit works with RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect objects in the frame
            result_frame, prediction_data = detect_objects(frame_rgb)
            
            # Display the result
            FRAME_WINDOW.image(result_frame)
            detection_info_placeholder = st.empty()  # Placeholder for text results
           # Prepare and update detected objects as text
            detected_objects_text = "Detected Objects:\n"
            for pred in prediction_data:
                # st.empty()
                # st.write(f"Class: {pred['class']}, Confidence: {pred['confidence']}")
                detected_objects_text += (f"Class: {pred['class']}, "
                                      f"Confidence: {pred['confidence']:.2f}, "
                                      f"Box: {pred['box']}\n"
                                     )
        
                # Update the text in the placeholder to avoid component creation in each loop
                detection_info_placeholder = st.empty()  # Placeholder for text results
                
            # Add a way to stop the webcam feed
            if stop_webcam:
                break
        detection_info_placeholder.text(detected_objects_text)
        cap.release()

    # Image upload input
    elif source_option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Convert the uploaded image to PIL format
            image = Image.open(uploaded_image)

            # Detect objects in the uploaded image
            result_img = detect_objects(image)

            # Display the original and the result
            st.image(result_img, caption="Detected objects", use_column_width=True)

if __name__ == "__main__":
    main()
import streamlit as st
import cv2
from ultralytics import YOLO
import os
import numpy as np
from PIL import Image

# Streamlit UI Setup
st.set_page_config(page_title="Bone Fracture Detection", layout="centered", page_icon="🦴")
st.markdown(
    """
    <style>
    body { background-color: black; color: white; }
    h1 { text-align: center; color: white; }
    .upload-label { color: white; font-size: 18px; }
    </style>
    """, unsafe_allow_html=True
)
st.markdown("<h1>Bone Fracture Detection 🦴</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    imgPredict = img.copy()

    # Load YOLO model
    model_path = os.path.join(
        "Yolov8 Models", "m_model_epochs_100_imgsize_640_batchsize_32",
        "runs", "detect", "yolov8n_custom", "weights", "best.pt"
    )

    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Run prediction
    threshold = 0.5
    results = model(imgPredict)[0]
    fracture_detected = False

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            fracture_detected = True
            class_name = results.names[int(class_id)].upper()
            cv2.rectangle(imgPredict, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(imgPredict, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    imgPredict_rgb = cv2.cvtColor(imgPredict, cv2.COLOR_BGR2RGB)

    # Show image (FIXED: updated to use_container_width)
    st.image(imgPredict_rgb, caption="Prediction", use_container_width=True)

    # Show result text
    if fracture_detected:
        st.markdown("<h3 style='color: red;'>Fractured ✅</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>Normal 🦴</h3>", unsafe_allow_html=True)









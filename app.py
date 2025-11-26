import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "0"

import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import cv2

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("🦾 YOLO Object Detection")
st.write("Upload an image to detect objects using a YOLO model.")

# ------------------------------------------------------------
# Load YOLO model FROM HUGGINGFACE URL
# ------------------------------------------------------------

HF_MODEL_URL = "https://huggingface.co/arbajshaikh880/yolo_model/resolve/main/best.pt"

@st.cache_resource
def load_model():
    return YOLO(HF_MODEL_URL)  # loads from HuggingFace

model = load_model()

# ------------------------------------------------------------
# File upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    if st.button("🚀 Detect Objects"):
        with st.spinner("Detecting..."):
            # Save uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                results = model.predict(source=tmp.name, save=False, conf=0.25)

            # Plot detection results
            result_img = results[0].plot()  # BGR
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            st.image(result_img_rgb, caption="🎯 Detection Result", use_column_width=True)
            st.success("✅ Detection complete!")

else:
    st.info("👆 Please upload an image to start detection.")

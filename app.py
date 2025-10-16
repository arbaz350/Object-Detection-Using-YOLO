import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.title("🦾 YOLO Object Detection Web App")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # load your trained model
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting..."):
            # Create a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                results = model.predict(source=tmp.name, save=False, conf=0.25)

            # Display result
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detected Objects", use_column_width=True)

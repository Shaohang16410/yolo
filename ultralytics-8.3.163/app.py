# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import traceback
import requests
from tqdm import tqdm
from ultralytics import YOLO

# Set page configuration
st.set_page_config(
    page_title="License Plate Recognizer",
    page_icon="🚗",
    layout="wide"
)


def download_file(url, filename):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        progress_bar = st.progress(0)
        status_text = st.empty()

        with open(filename, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                downloaded += size
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading {filename}: {downloaded / total_size * 100:.1f}%")

        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"Failed to download {filename}: {e}")
        return False


# Try to import util functions with error handling
try:
    from util import get_car, read_license_plate
except ImportError as e:
    st.error(f"❌ Failed to import util functions: {e}")
    st.error("Please make sure util.py exists and has get_car and read_license_plate functions")
    st.stop()


# Load models with caching
@st.cache_resource
def load_models():
    try:
        st.info("Loading models, please wait...")

        # Check if model files exist, offer to download if not
        models_to_download = []

        if not os.path.exists('yolov8n.pt'):
            models_to_download.append(('yolov8n.pt',
                                       'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'))

        if not os.path.exists('license_plate_detector.pt'):
            st.warning("⚠️ License plate detector model not found.")
            st.info("You can:")
            st.info("1. Upload your own model file")
            st.info("2. Use a placeholder for testing")
            st.info("3. Download a pre-trained model")

            option = st.radio("Choose an option:",
                              ["Upload model", "Use placeholder", "Download model"])

            if option == "Upload model":
                uploaded_model = st.file_uploader("Upload license_plate_detector.pt", type=['pt'])
                if uploaded_model:
                    with open("license_plate_detector.pt", "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    st.success("✅ Model uploaded successfully!")

            elif option == "Download model":
                if st.button("Download License Plate Detector"):
                    # This is a placeholder URL - you'll need to find a real license plate detection model
                    success = download_file(
                        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        "license_plate_detector.pt"
                    )
                    if success:
                        st.success("✅ Model downloaded successfully!")

            elif option == "Use placeholder":
                # Use YOLOv8n as a placeholder for both detection tasks
                if not os.path.exists('license_plate_detector.pt'):
                    os.symlink('yolov8n.pt', 'license_plate_detector.pt')
                    st.info("⚠️ Using vehicle detection model for license plates (placeholder)")

        # Load models
        if os.path.exists('yolov8n.pt'):
            coco_model = YOLO('yolov8n.pt')
        else:
            st.error("Vehicle detection model not found. Please download it first.")
            st.stop()

        if os.path.exists('license_plate_detector.pt'):
            license_plate_detector = YOLO('license_plate_detector.pt')
        else:
            st.error("License plate detector model not found.")
            st.stop()

        st.success("✅ Models loaded successfully!")
        return coco_model, license_plate_detector

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")
        st.stop()


# ... [rest of your process_image and main functions remain the same] ...

def main():
    st.title("🚗 License Plate Recognition App")
    st.markdown("Upload an image to detect and recognize license plates")

    # First, check if basic requirements are met
    if not os.path.exists('yolov8n.pt'):
        st.warning("⚠️ Required model files not found.")
        if st.button("Download Basic Models"):
            with st.spinner("Downloading models..."):
                # Download YOLOv8n model
                yolo_success = download_file(
                    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                    "yolov8n.pt"
                )
                if yolo_success:
                    st.success("✅ Downloaded vehicle detection model!")
                else:
                    st.error("❌ Failed to download vehicle detection model")
                    return

    try:
        # Load models
        coco_model, license_plate_detector = load_models()

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("❌ Failed to read the image. Please try another file.")
                return

            # Display original image
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Process the image when button is clicked
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    start_time = time.time()
                    processed_image, results = process_image(
                        image, coco_model, license_plate_detector
                    )
                    end_time = time.time()
                    processing_time = end_time - start_time

                # Display results
                st.subheader("Processed Image")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Display detection results
                st.subheader("Detection Results")
                if results:
                    for i, res in enumerate(results):
                        st.success(f"**License Plate {i + 1}:** {res['text']}")
                        st.write(f"- Car Detection Confidence: {res['car_score'] * 100:.2f}%")
                        st.write(f"- License Plate Detection Confidence: {res['plate_bbox_score'] * 100:.2f}%")
                        st.write(f"- OCR Confidence: {res['ocr_score'] * 100:.2f}%")
                else:
                    st.warning("No license plates detected in the image.")

                st.info(f"Processing time: {processing_time:.2f} seconds")

    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        st.error(f"Full error details: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
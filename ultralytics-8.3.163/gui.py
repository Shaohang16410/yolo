import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# Import the necessary functions from your util.py file
# Ensure util.py is in the same directory as app.py
from util import get_car, read_license_plate

# --- MODEL LOADING ---
# Use st.cache_resource to load models only once and cache them
@st.cache_resource
def load_models():
    """Loads the YOLO models for vehicle and license plate detection."""
    print("Loading models...")
    coco_model = YOLO('yolo11n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    print("Models loaded successfully.")
    return coco_model, license_plate_detector

# --- CORE PROCESSING LOGIC ---
# This is the same function from your gui.py, slightly adapted
def process_image(frame, coco_model, license_plate_detector):
    """
    Processes a single image frame to detect cars, license plates, and read the plate number.
    It then draws the results on the image.

    Args:
        frame (np.array): The image to process.
        coco_model (YOLO): The pre-loaded YOLO model for vehicle detection.
        license_plate_detector (YOLO): The pre-loaded model for license plate detection.

    Returns:
        tuple: A tuple containing:
            - The annotated image (np.array).
            - A list of dictionaries, where each dict contains detected text and scores.
    """
    # 1. Detect Vehicles
    vehicles = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # 2. Detect License Plates
    license_plates = license_plate_detector(frame)[0]

    detection_results = []

    # 3. Match plates to cars and read them
    for license_plate in license_plates.boxes.data.tolist():
        lp_x1, lp_y1, lp_x2, lp_y2, plate_bbox_score, class_id = license_plate
        car_x1, car_y1, car_x2, car_y2, car_score = get_car(license_plate, detections_)

        if car_x1 != -1:  # If a car was found for this plate
            # Crop the license plate
            license_plate_crop = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2), :]

            # Read the license plate text
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

            if license_plate_text:
                result_info = {
                    'text': license_plate_text,
                    'car_score': car_score,
                    'plate_bbox_score': plate_bbox_score,
                    'ocr_score': license_plate_text_score
                }
                detection_results.append(result_info)

                # --- Draw visualizations directly on the frame ---
                cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 3)
                cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                text_label = f"{license_plate_text}"
                (w, h), _ = cv2.getTextSize(text_label, font, 1.5, 4)
                text_x = int(lp_x1)
                text_y = int(lp_y1) - 10
                
                # Draw background and text
                cv2.rectangle(frame, (text_x, text_y - h - 10), (text_x + w, text_y), (255, 255, 255), -1)
                cv2.putText(frame, text_label, (text_x, text_y - 5), font, 1.5, (0, 0, 0), 4)

    return frame, detection_results


# --- STREAMLIT APP ---

st.set_page_config(page_title="License Plate Recognizer", layout="wide")
st.title("License Plate Recognition with YOLO")

# Load models
coco_model, license_plate_detector = load_models()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(opencv_image, channels="BGR", caption="Original Image", use_column_width=True)

    with col2:
        # Process the image
        with st.spinner('Processing image...'):
            start_time = time.time()
            annotated_frame, detection_results = process_image(opencv_image.copy(), coco_model, license_plate_detector)
            end_time = time.time()
            processing_time = end_time - start_time

        st.image(annotated_frame, channels="BGR", caption="Processed Image", use_column_width=True)

    # Display results below the images
    st.markdown("---")
    st.subheader("Detection Results")
    st.info(f"Processing Time: {processing_time:.2f} seconds")

    if detection_results:
        for res in detection_results:
            st.success(f"**Plate:** `{res['text']}`")
            # Display scores in columns
            score_col1, score_col2, score_col3 = st.columns(3)
            with score_col1:
                st.metric(label="Car Confidence", value=f"{res['car_score']*100:.1f}%")
            with score_col2:
                st.metric(label="Plate Confidence", value=f"{res['plate_bbox_score']*100:.1f}%")
            with score_col3:
                st.metric(label="OCR Confidence", value=f"{res['ocr_score']*100:.1f}%")
            st.markdown("---")
    else:
        st.warning("No license plates were found in the image.")

else:
    st.info("Please upload an image to begin processing.")
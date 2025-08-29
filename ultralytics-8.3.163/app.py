# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import traceback
from ultralytics import YOLO

# Set page configuration
st.set_page_config(
    page_title="License Plate Recognizer",
    page_icon="🚗",
    layout="wide"
)

# Try to import util functions with error handling
try:
    from util import get_car, read_license_plate

    st.success("✅ Successfully imported util functions")
except ImportError as e:
    st.error(f"❌ Failed to import util functions: {e}")
    st.stop()


# Load models with caching
@st.cache_resource
def load_models():
    try:
        st.info("Loading models, please wait...")

        # Check if model files exist
        if not os.path.exists('yolo11n.pt'):
            st.error("❌ yolo11n.pt not found! Please make sure it's in your project directory.")
            st.stop()

        if not os.path.exists('license_plate_detector.pt'):
            st.error("❌ license_plate_detector.pt not found! Please make sure it's in your project directory.")
            st.stop()

        coco_model = YOLO('yolo11n.pt')
        license_plate_detector = YOLO('license_plate_detector.pt')

        st.success("✅ Models loaded successfully!")
        return coco_model, license_plate_detector

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")
        st.stop()


def process_image(frame, coco_model, license_plate_detector):
    """
    Processes a single image frame to detect cars, license plates, and read the plate number.
    """
    try:
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
        st.write(f"Found {len(license_plates.boxes.data.tolist())} license plates")

        detection_results = []
        annotated_frame = frame.copy()

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
                    # Store all relevant info in the results dictionary
                    result_info = {
                        'text': license_plate_text,
                        'car_score': car_score,
                        'plate_bbox_score': plate_bbox_score,
                        'ocr_score': license_plate_text_score
                    }
                    detection_results.append(result_info)

                    # Draw visualizations directly on the frame
                    # Draw car bounding box
                    cv2.rectangle(annotated_frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0),
                                  3)

                    # Draw license plate bounding box
                    cv2.rectangle(annotated_frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 3)

                    # Prepare and display the license plate text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_thickness = 4
                    text_color = (0, 0, 0)
                    bg_color = (255, 255, 255)

                    (text_width, text_height), baseline = cv2.getTextSize(
                        license_plate_text, font, font_scale, font_thickness
                    )
                    text_x = int(car_x1)
                    text_y = int(car_y1) - 10

                    # Draw a white background for the text for better readability
                    cv2.rectangle(
                        annotated_frame,
                        (text_x, text_y - text_height - baseline),
                        (text_x + text_width, text_y),
                        bg_color, -1
                    )
                    # Put the text on the image
                    cv2.putText(
                        annotated_frame,
                        license_plate_text,
                        (text_x, text_y - baseline),
                        font, font_scale, text_color, font_thickness
                    )

        return annotated_frame, detection_results

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")
        return frame, []


def main():
    st.title("🚗 License Plate Recognition App")
    st.markdown("Upload an image to detect and recognize license plates")

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
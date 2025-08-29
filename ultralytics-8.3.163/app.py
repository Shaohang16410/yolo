# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# Import the necessary functions from util.py
from util import get_car, read_license_plate, detect_car_brand, detect_car_color

# Set page configuration
st.set_page_config(
    page_title="License Plate Recognizer",
    page_icon="🚗",
    layout="wide"
)


# Load models with caching
@st.cache_resource
def load_models(coco_model_file, license_plate_model_file):
    st.info("Loading models, please wait...")

    try:
        coco_model = YOLO(coco_model_file)  # Vehicle detection model
        license_plate_detector = YOLO(license_plate_model_file)  # License plate detection model
        st.success("✅ Models loaded successfully!")
        return coco_model, license_plate_detector
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()


def process_image(image, coco_model, license_plate_detector):
    """
    Detects cars and license plates, performs OCR, and overlays results on the image.
    Returns the annotated image and structured detection results.
    """
    results_list = []

    # Detect vehicles
    vehicle_results = coco_model(image)[0]
    vehicles = vehicle_results.boxes.data.tolist()

    # Detect license plates
    lp_results = license_plate_detector(image)[0]
    license_plates = lp_results.boxes.data.tolist()

    # Process each license plate
    for lp in license_plates:
        x1, y1, x2, y2, plate_score, _ = lp

        # Match plate to a car
        car = get_car(lp, vehicles)
        if car[0] == -1:
            continue

        xcar1, ycar1, xcar2, ycar2, car_score = car

        # Crop the license plate and car
        license_plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
        car_crop = image[int(ycar1):int(ycar2), int(xcar1):int(xcar2)]

        # Read license plate text
        text, ocr_score = read_license_plate(license_plate_crop)

        # Detect car brand and color
        car_brand, brand_confidence = detect_car_brand(car_crop)
        car_color, color_confidence = detect_car_color(car_crop)

        # Draw bounding boxes
        cv2.rectangle(image, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Prepare information text
        info_texts = []
        if text:
            info_texts.append(f"Plate: {text}")
        if car_brand:
            info_texts.append(f"Brand: {car_brand}")
        if car_color:
            info_texts.append(f"Color: {car_color}")

        # Draw information text
        y_offset = int(ycar1) - 10
        for i, info_text in enumerate(info_texts):
            cv2.putText(image, info_text, (int(xcar1), y_offset - (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results_list.append({
            "text": text,
            "car_score": car_score,
            "plate_bbox_score": plate_score,
            "ocr_score": ocr_score,
            "car_brand": car_brand,
            "brand_confidence": brand_confidence,
            "car_color": car_color,
            "color_confidence": color_confidence
        })

    return image, results_list


def main():
    st.title("🚗 Advanced Vehicle Recognition App")
    st.markdown("Upload an image to detect vehicles, recognize license plates, and identify car brands & colors")

    # Sidebar for model selection
    st.sidebar.header("Model Configuration")

    # Option 1: Use default models
    use_default_models = st.sidebar.checkbox("Use default models", value=True)

    if use_default_models:
        coco_model_path = 'yolo11n.pt'
        license_plate_model_path = 'license_plate_detector.pt'

        if not os.path.exists(license_plate_model_path):
            st.sidebar.warning("Default license plate model not found. Please upload one.")
            use_default_models = False

    if not use_default_models:
        # Option 2: Upload custom models
        st.sidebar.info("Upload your model files")
        uploaded_coco_model = st.sidebar.file_uploader("Upload YOLO vehicle model", type=['pt'])
        uploaded_lp_model = st.sidebar.file_uploader("Upload license plate model", type=['pt'])

        if uploaded_coco_model and uploaded_lp_model:
            with open("temp_coco.pt", "wb") as f:
                f.write(uploaded_coco_model.getbuffer())
            with open("temp_lp.pt", "wb") as f:
                f.write(uploaded_lp_model.getbuffer())

            coco_model_path = "temp_coco.pt"
            license_plate_model_path = "temp_lp.pt"
        else:
            st.info("Please upload model files to continue")
            return

    # Load models
    coco_model, license_plate_detector = load_models(coco_model_path, license_plate_model_path)

    # File uploader for images
    st.header("Image Processing")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Process button
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                start_time = time.time()
                processed_image, results = process_image(
                    image, coco_model, license_plate_detector
                )
                end_time = time.time()
                processing_time = end_time - start_time

            # Display processed image
            st.subheader("Processed Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Display results
            st.subheader("Detection Results")
            if results:
                for i, res in enumerate(results):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.success(f"**Vehicle {i + 1} Detection Results:**")
                        if res['text']:
                            st.write(f"🚗 **License Plate:** {res['text']}")
                        if res['car_brand']:
                            st.write(f"🏭 **Brand:** {res['car_brand']} ({res['brand_confidence'] * 100:.1f}%)")
                        if res['car_color']:
                            st.write(f"🎨 **Color:** {res['car_color']} ({res['color_confidence'] * 100:.1f}%)")

                    with col2:
                        st.write("**Confidence Scores:**")
                        st.write(f"- Car Detection: {res['car_score'] * 100:.2f}%")
                        st.write(f"- License Plate Detection: {res['plate_bbox_score'] * 100:.2f}%")
                        if res['text']:
                            st.write(f"- OCR Confidence: {res['ocr_score'] * 100:.2f}%")

                    st.divider()
            else:
                st.warning("No vehicles or license plates detected in the image.")

            st.info(f"Processing time: {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()
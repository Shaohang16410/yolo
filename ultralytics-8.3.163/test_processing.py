# test_processing.py
import cv2
import numpy as np
from ultralytics import YOLO
from util import get_car, read_license_plate, detect_car_brand, detect_car_color

# Load models
coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Load test image
image = cv2.imread('test_image.jpg')  # Use a small test image

print("Testing vehicle detection...")
vehicle_results = coco_model(image)[0]
print(f"Vehicle detection results: {len(vehicle_results.boxes.data.tolist())} detections")

print("Testing license plate detection...")
lp_results = license_plate_detector(image)[0]
print(f"License plate detection results: {len(lp_results.boxes.data.tolist())} detections")

print("Testing util functions...")
# Test get_car function
test_license_plate = [100, 100, 200, 200, 0.9, 0]  # Example coordinates
vehicles = [[50, 50, 250, 250, 0.8]]  # Example vehicle
car_result = get_car(test_license_plate, vehicles)
print(f"get_car result: {car_result}")

# Test read_license_plate function
test_crop = image[100:200, 100:200]  # Example crop
if test_crop.size > 0:
    text, score = read_license_plate(test_crop)
    print(f"read_license_plate result: {text}, {score}")
else:
    print("Empty crop")

# Test car brand detection
if test_crop.size > 0:
    brand, brand_conf = detect_car_brand(test_crop)
    print(f"Car brand detection: {brand}, {brand_conf}")

# Test car color detection
if test_crop.size > 0:
    color, color_conf = detect_car_color(test_crop)
    print(f"Car color detection: {color}, {color_conf}")
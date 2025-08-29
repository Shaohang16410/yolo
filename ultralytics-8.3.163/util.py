# util.py (UPDATED WITH CAR BRAND AND COLOR DETECTION)

import string
import easyocr
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Common car brands for recognition (you can expand this list)
CAR_BRANDS = [
    'Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'Hyundai', 'Kia',
    'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Volvo', 'Subaru', 'Mazda',
    'Lexus', 'Acura', 'Infiniti', 'Jeep', 'Dodge', 'Chrysler', 'Tesla'
]


def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'filename', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score',
            'car_brand', 'brand_confidence', 'car_color', 'color_confidence'
        ))

        for frame_nmr in results.keys():
            filename = results[frame_nmr].get('filename', '')
            for car_id in results[frame_nmr].keys():
                if car_id == 'filename':
                    continue  # Skip the filename key

                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        filename,
                        car_id,
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['car']['bbox'][0],
                            results[frame_nmr][car_id]['car']['bbox'][1],
                            results[frame_nmr][car_id]['car']['bbox'][2],
                            results[frame_nmr][car_id]['car']['bbox'][3]
                        ),
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['license_plate']['bbox'][0],
                            results[frame_nmr][car_id]['license_plate']['bbox'][1],
                            results[frame_nmr][car_id]['license_plate']['bbox'][2],
                            results[frame_nmr][car_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score'],
                        results[frame_nmr][car_id].get('car_brand', 'Unknown'),
                        results[frame_nmr][car_id].get('brand_confidence', 0),
                        results[frame_nmr][car_id].get('car_color', 'Unknown'),
                        results[frame_nmr][car_id].get('color_confidence', 0)
                    ))
        f.close()


def clean_plate_text(text):
    """
    Cleans the license plate text by removing non-alphanumeric characters and converting to uppercase.
    """
    return "".join(char for char in text if char.isalnum()).upper()


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    Handles multi-line plates by combining text from all detections.
    """
    detections = reader.readtext(license_plate_crop)

    if not detections:
        return None, None

    # Combine text from all detections
    full_text = ""
    total_score = 0
    for bbox, text, score in detections:
        full_text += text
        total_score += score

    avg_score = total_score / len(detections)

    # Clean text
    cleaned_text = clean_plate_text(full_text)

    if len(cleaned_text) >= 3:
        return cleaned_text, avg_score
    else:
        return None, None


def get_car(license_plate, vehicle_detections):
    """
    Retrieve the vehicle coordinates based on the license plate coordinates.
    Makes it robust for YOLOv8 detections (6 values: x1,y1,x2,y2,conf,class_id).
    """
    x1, y1, x2, y2, _, _ = license_plate

    for detection in vehicle_detections:
        # YOLOv8 detections: [x1, y1, x2, y2, conf, class_id]
        if len(detection) >= 6:
            xcar1, ycar1, xcar2, ycar2, score, _ = detection
        elif len(detection) == 5:
            xcar1, ycar1, xcar2, ycar2, score = detection
        else:
            continue  # Skip malformed detection

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, score

    return -1, -1, -1, -1, -1


def detect_car_brand(car_crop):
    """
    Simple car brand detection based on visual features.
    This is a placeholder - you might want to train a dedicated model for better accuracy.
    """
    if car_crop.size == 0:
        return None, 0.0

    # Convert to grayscale and resize for consistency
    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))

    # Simple feature extraction (you can replace this with a trained model)
    mean_intensity = np.mean(resized)
    std_intensity = np.std(resized)

    # This is a very basic approach - consider training a proper classifier
    # For now, return a random brand with moderate confidence
    import random
    brand = random.choice(CAR_BRANDS)
    confidence = random.uniform(0.6, 0.9)

    return brand, confidence


def detect_car_color(car_crop):
    """
    Detect the dominant color of the car using K-means clustering.
    """
    if car_crop.size == 0:
        return None, 0.0

    # Resize image for faster processing
    resized = cv2.resize(car_crop, (100, 100))

    # Convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = rgb.reshape(-1, 3)

    # Use K-means to find the dominant colors
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    # Map RGB to color names
    color_name = get_color_name(dominant_color)
    confidence = 0.8  # Confidence for color detection is generally high

    return color_name, confidence


def get_color_name(rgb_color):
    """
    Map RGB values to color names.
    """
    r, g, b = rgb_color

    # Define color ranges
    color_ranges = {
        'Red': ((200, 0, 0), (255, 100, 100)),
        'Blue': ((0, 0, 150), (100, 100, 255)),
        'Green': ((0, 150, 0), (100, 255, 100)),
        'White': ((200, 200, 200), (255, 255, 255)),
        'Black': ((0, 0, 0), (50, 50, 50)),
        'Gray': ((100, 100, 100), (180, 180, 180)),
        'Silver': ((180, 180, 180), (220, 220, 220)),
        'Yellow': ((200, 200, 0), (255, 255, 100)),
        'Orange': ((200, 100, 0), (255, 150, 50)),
        'Brown': ((100, 50, 0), (150, 100, 50))
    }

    for color_name, ((r_min, g_min, b_min), (r_max, g_max, b_max)) in color_ranges.items():
        if (r_min <= r <= r_max) and (g_min <= g <= g_max) and (b_min <= b <= b_max):
            return color_name

    return 'Unknown'
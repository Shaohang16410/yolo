# util.py (UPDATED, ROBUST VERSION)

import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)


def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'filename', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'
        ))

        for frame_nmr in results.keys():
            filename = results[frame_nmr].get('filename', '')
            for car_id in results[frame_nmr].keys():
                if car_id == 'filename':
                    continue  # Skip the filename key

                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():

                    f.write('{},{},{},{},{},{},{},{}\n'.format(
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
                        results[frame_nmr][car_id]['license_plate']['text_score']
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
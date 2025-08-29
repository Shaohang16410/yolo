from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np

from util import get_car, read_license_plate, write_csv

print("--- Starting Plate Recognition Script (Full Debug Mode) ---")

# Create debug directory if it doesn't exist
debug_dir = 'debug_crops'
os.makedirs(debug_dir, exist_ok=True)
print(f"Debug crops will be saved to '{debug_dir}/'")

results = {}

# ... (model loading remains the same) ...
print("Loading models...")
coco_model = YOLO('yolo11n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
print("Models loaded successfully.")

# ... (image finding remains the same) ...
input_dir = 'input_images'
image_paths = glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]')) + \
              glob.glob(os.path.join(input_dir, '*.[pP][nN][gG]')) + \
              glob.glob(os.path.join(input_dir, '*.[jJ][pP][eE][gG]'))
print(f"Found {len(image_paths)} images to process.")

vehicles = [2, 3, 5, 7]

for frame_nmr, image_path in enumerate(image_paths):
    print(f"\n--- Processing image ({frame_nmr + 1}/{len(image_paths)}): {image_path} ---")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  - Warning: Could not read image. Skipping.")
        continue

    results[frame_nmr] = {}
    results[frame_nmr]['filename'] = image_path

    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
    print(f"  - Detected {len(detections_)} vehicles.")

    license_plates = license_plate_detector(frame)[0]
    print(f"  - Detected {len(license_plates.boxes)} license plates.")

    if len(detections_) > 0 and len(license_plates.boxes) > 0:
        plate_counter = 0
        for license_plate in license_plates.boxes.data.tolist():
            lp_x1, lp_y1, lp_x2, lp_y2, score, class_id = license_plate

            car_x1, car_y1, car_x2, car_y2, car_score = get_car(license_plate, detections_)

            if car_x1 != -1:
                print(f"    - Found a license plate INSIDE a car bounding box.")

                # crop license plate
                license_plate_crop = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2), :]

                # --- SAVE THE CROP FOR DEBUGGING ---
                crop_filename = f"crop_{frame_nmr}_{plate_counter}.png"
                crop_filepath = os.path.join(debug_dir, crop_filename)
                cv2.imwrite(crop_filepath, license_plate_crop)
                print(f"      -> Saved license plate crop to '{crop_filepath}'")
                plate_counter += 1

                # read license plate number (using the color crop)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                print(f"    - OCR result: Text='{license_plate_text}', Score={license_plate_text_score}")

                if license_plate_text is not None:
                    # ... (rest of the logic is the same) ...
                    print(f"      -> SUCCESS: Valid license plate found: '{license_plate_text}'. Storing result.")
                    car_id = f'{frame_nmr}_{len(results[frame_nmr])}'
                    results[frame_nmr][car_id] = {'car': {'bbox': [car_x1, car_y1, car_x2, car_y2]},
                                                  'license_plate': {'bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                else:
                    print("      -> FAILED: OCR did not return a valid result from the crop.")
            else:
                print("    - Found a license plate, but it was NOT inside any detected car bounding box. Discarding.")

# ... (writing results remains the same) ...
print("\n--- Writing results to CSV ---")
final_results_count = sum(len(v) - 1 for v in results.values())
if final_results_count > 0:
    write_csv(results, './numberplate.csv')
    print(f"Processing complete. {final_results_count} results saved to numberplate.csv")
else:
    print("Processing complete. No valid license plates were found across all images to save.")
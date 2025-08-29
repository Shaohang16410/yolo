# visualize.py

import ast
import cv2
import pandas as pd
import os


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    # This function remains the same
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img


# Create output directory if it doesn't exist
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Load the results from the CSV file
results_df = pd.read_csv('./numberplate.csv')

# Group results by filename to process one image at a time
for filename, group in results_df.groupby('filename'):
    print(f"Processing {filename}...")

    # Load the original image
    frame = cv2.imread(filename)
    if frame is None:
        print(f"Warning: Could not read image {filename}. Skipping.")
        continue

    for index, row in group.iterrows():
        # Draw car bounding box
        car_bbox_str = row['car_bbox'].replace('[', '').replace(']', '').replace('  ', ' ')
        car_x1, car_y1, car_x2, car_y2 = map(float, car_bbox_str.split())
        draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                    line_length_x=100, line_length_y=100)

        # Draw license plate bounding box
        lp_bbox_str = row['license_plate_bbox'].replace('[', '').replace(']', '').replace('  ', ' ')
        x1, y1, x2, y2 = map(float, lp_bbox_str.split())
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

        # Prepare license plate text to display
        license_plate_number = str(row['license_number'])

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 6
        text_color = (0, 0, 0)
        bg_color = (255, 255, 255)

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(license_plate_number, font, font_scale, font_thickness)

        # Position for the text background
        text_x = int(car_x1)
        text_y = int(car_y1) - text_height - 20

        # Draw background rectangle for the text
        cv2.rectangle(frame, (text_x, text_y), (text_x + text_width, text_y + text_height + baseline), bg_color, -1)

        # Put the license plate text
        cv2.putText(frame, license_plate_number, (text_x, text_y + text_height), font, font_scale, text_color,
                    font_thickness)

    # Save the annotated image
    output_path = os.path.join(output_dir, os.path.basename(filename))
    cv2.imwrite(output_path, frame)

print(f"Visualization complete. Annotated images saved to '{output_dir}' directory.")
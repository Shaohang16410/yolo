# gui_app.py

import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import time  # Import the time module

# Import the necessary functions from your existing util.py file
from util import get_car, read_license_plate


# --- CORE PROCESSING LOGIC (Adapted from main.py) ---

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
                # Store all relevant info in the results dictionary
                result_info = {
                    'text': license_plate_text,
                    'car_score': car_score,
                    'plate_bbox_score': plate_bbox_score,
                    'ocr_score': license_plate_text_score
                }
                detection_results.append(result_info)

                # --- Draw visualizations directly on the frame ---

                # Draw car bounding box (simple rectangle for the UI)
                cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 3)

                # Draw license plate bounding box
                cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 3)

                # Prepare and display the license plate text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 4
                text_color = (0, 0, 0)
                bg_color = (255, 255, 255)

                (text_width, text_height), baseline = cv2.getTextSize(license_plate_text, font, font_scale,
                                                                      font_thickness)
                text_x = int(car_x1)
                text_y = int(car_y1) - 10

                # Draw a white background for the text for better readability
                cv2.rectangle(frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y), bg_color,
                              -1)
                # Put the text on the image
                cv2.putText(frame, license_plate_text, (text_x, text_y - baseline), font, font_scale, text_color,
                            font_thickness)

    return frame, detection_results


# --- TKINTER GUI APPLICATION ---

class PlateRecognizerApp:
    def __init__(self, root, coco_model, license_plate_detector):
        self.root = root
        self.root.title("License Plate Recognizer")
        self.root.geometry("800x700")

        # Store models
        self.coco_model = coco_model
        self.license_plate_detector = license_plate_detector

        # --- UI Elements ---

        # Frame for buttons and labels
        top_frame = Frame(root)
        top_frame.pack(pady=10)

        # Button to select image
        self.select_button = Button(top_frame, text="Select Image", command=self.load_and_process_image,
                                    font=("Helvetica", 12))
        self.select_button.pack(side=tk.LEFT, padx=10)

        # Frame for text results
        text_frame = Frame(top_frame)
        text_frame.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        # Label to show the main result (plate number and scores)
        self.result_label = Label(text_frame, text="Please select an image to begin.", font=("Helvetica", 12, "bold"), justify=tk.LEFT)
        self.result_label.pack(anchor='w')  # Align to the west/left

        # Label to show status like processing time
        self.status_label = Label(text_frame, text="", font=("Helvetica", 10), justify=tk.LEFT)
        self.status_label.pack(anchor='w')  # Align to the west/left

        # Label to display the image
        self.image_label = Label(root)
        self.image_label.pack(pady=10, padx=10, expand=True)

    def load_and_process_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return  # User cancelled the dialog

        self.result_label.config(text="Processing...", fg="blue")
        self.status_label.config(text="")
        self.root.update_idletasks()  # Update UI to show "Processing..." message

        # Read image with OpenCV
        frame = cv2.imread(file_path)

        # --- Time the processing ---
        start_time = time.time()
        # Process the image using our modified function
        annotated_frame, detection_results = process_image(frame, self.coco_model, self.license_plate_detector)
        end_time = time.time()
        processing_time = end_time - start_time

        # Update status label with processing time
        self.status_label.config(text=f"Processing Time: {processing_time:.2f} seconds")

        # Display the results
        if detection_results:
            # Format the results with scores
            display_texts = []
            for res in detection_results:
                text = res['text']
                # The scores are probabilities (0-1), multiply by 100 for percentage
                c_score = res['car_score'] * 100
                p_score = res['plate_bbox_score'] * 100
                o_score = res['ocr_score'] * 100
                display_texts.append(f"{text} (Car: {c_score:.1f}%, Plate: {p_score:.1f}%, OCR: {o_score:.1f}%)")

            result_text = "Detected: " + "\n".join(display_texts)
            self.result_label.config(text=result_text, fg="green")
        else:
            self.result_label.config(text="No license plates found.", fg="red")

        # Display the processed image in the UI
        self.display_image(annotated_frame)

    def display_image(self, frame):
        # OpenCV uses BGR, Pillow/Tkinter uses RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Resize image to fit in the window while maintaining aspect ratio
        max_width = 750
        max_height = 600
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage for Tkinter
        photo_image = ImageTk.PhotoImage(image=pil_image)

        # Update the image label
        self.image_label.config(image=photo_image)
        # Keep a reference to the image to prevent it from being garbage collected
        self.image_label.image = photo_image


if __name__ == "__main__":
    # --- Load Models ONCE at the start ---
    print("Loading models, please wait...")
    coco_model = YOLO('yolo11n.pt')  # Your vehicle detection model
    license_plate_detector = YOLO('license_plate_detector.pt')  # Your LP detection model
    print("Models loaded successfully.")

    # --- Create and run the Tkinter application ---
    root = tk.Tk()
    app = PlateRecognizerApp(root, coco_model, license_plate_detector)
    root.mainloop()
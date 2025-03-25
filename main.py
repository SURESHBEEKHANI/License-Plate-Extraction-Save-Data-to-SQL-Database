import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import math
import re
import json
import sqlite3
import sys
from datetime import datetime
from ultralytics import YOLO  # Make sure you have YOLOv10 installed or adjust accordingly
from paddleocr import PaddleOCR
from PIL import Image
import base64

# -------------------------------
# Utility Functions & Initialization
# -------------------------------

# Load the YOLOv10 model (cached for performance)
def load_model():
    return YOLO("weights/best.pt")  # Update the model path as needed

model = load_model()

# Initialize PaddleOCR (runs only once)
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Class names for detected objects (if needed)
className = ["License"]

def paddle_ocr(frame, x1, y1, x2, y2, min_score=60):
    """
    Crop the region from the frame and run PaddleOCR.
    Clean up the OCR result by removing unwanted characters.
    """
    cropped = frame[y1:y2, x1:x2]
    result = ocr.ocr(cropped, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile(r'[\W]')  # changed from '[\W]' to a raw string literal
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)

def save_json(license_plates, startTime, endTime):
    """
    Save detected license plates into a JSON file (for each 20-second interval)
    and update the cumulative JSON file. Also, call the database saving function.
    """
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    os.makedirs("json", exist_ok=True)
    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    existing_data.append(interval_data)
    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

    save_to_database(license_plates, startTime, endTime)

def save_to_database(license_plates, start_time, end_time):
    """
    Save detected license plates into an SQLite database.
    A UNIQUE constraint on (start_time, license_plate) prevents duplicates.
    """
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LicensePlates(
            start_time TEXT,
            end_time TEXT,
            license_plate TEXT,
            UNIQUE(start_time, license_plate)
        )
    ''')
    for plate in license_plates:
        cursor.execute('''
            INSERT OR IGNORE INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()

def get_base64_image(image_path):
    """
    Convert an image file to base64 (used to display a logo).
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# New main() function replacing the Streamlit UI
def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_or_image_file>")
        sys.exit(1)
    input_path = sys.argv[1]
    
    # Determine if input is a video or image file
    if input_path.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(input_path)
        startTime = datetime.now()
        license_plates = set()
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                endTime = datetime.now()
                if license_plates:
                    save_json(license_plates, startTime, endTime)
                break
            count += 1
            print(f"Frame Number: {count}")
            # ...existing video processing code...
            results = model.predict(frame, conf=0.45)
            for result in results:
                # ...existing code for iterating over boxes...
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)
                    cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Process image file
        frame = cv2.cvtColor(np.array(Image.open(input_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        results = model.predict(frame, conf=0.45)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = paddle_ocr(frame, x1, y1, x2, y2)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow("Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

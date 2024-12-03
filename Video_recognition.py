from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from utils import get_car, read_characters, write_csv, license_plate_show, validate_and_correct_plate
def load_models():
    vehicle_model = YOLO('yolov8n.pt')  # Path to your vehicle detection model
    license_plate_detector = YOLO('./models/license_plate_detector_v4.pt')  # Path to your license plate detection model
    character_recognition_model = YOLO('./models/last.pt')  # Path to your character recognition model
    return vehicle_model, license_plate_detector, character_recognition_model
def process_frame(frame, vehicle_model, license_plate_detector, character_recognition_model, mot_tracker, vehicles,
                  results, frame_nmr):
    detections = vehicle_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
    track_ids = mot_tracker.update(np.asarray(detections_))
    license_plates = license_plate_detector(frame)[0]
    license_plate_list = []
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        license_plate_list.append([x1, y1, x2, y2, score])

    for lp in license_plate_list:
        x1, y1, x2, y2, score = lp
        license_plate_bbox = [x1, y1, x2, y2]
        vehicle = get_car(license_plate_bbox, track_ids)
        if vehicle is not None:
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            # Recognize characters
            labels_list, x_list, y_list = read_characters(character_recognition_model, license_plate_crop)
            if labels_list:
                plate_predict, is_single_line = license_plate_show(labels_list, x_list, y_list)
                corrected_plate = validate_and_correct_plate(plate_predict, is_single_line)
                if corrected_plate:
                    print(f"Frame {frame_nmr}, Car ID {car_id}, Plate: {corrected_plate}")
                    if car_id not in results[frame_nmr]:
                        results[frame_nmr][car_id] = {
                            'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                            'license_plate_bbox': [x1, y1, x2, y2],
                            'license_plate_text': corrected_plate
                        }
                else:
                    print("Biển số không hợp lệ hoặc không đủ ký tự.")
            else:
                print("Không nhận dạng được ký tự trên biển số.")
def process_video(video_path, vehicle_model, license_plate_detector, character_recognition_model, mot_tracker,
                  vehicles):
    results = {}
    cap = cv2.VideoCapture(video_path)
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            process_frame(frame, vehicle_model, license_plate_detector, character_recognition_model, mot_tracker,
                          vehicles, results, frame_nmr)
    return results
def main(video_path):
    # Load models
    vehicle_model, license_plate_detector, character_recognition_model = load_models()
    # Initialize tracker
    mot_tracker = Sort()
    # Vehicle class IDs (depends on your model)
    vehicles = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
    # Process video
    results = process_video(video_path, vehicle_model, license_plate_detector, character_recognition_model, mot_tracker,
                            vehicles)
    # Write results to CSV
    write_csv(results, './output_video/results.csv')
if __name__ == "__main__":
    video_path = './sample3.mp4'  # Path to your video file
    main(video_path)

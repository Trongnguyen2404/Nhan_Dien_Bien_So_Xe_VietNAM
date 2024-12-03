import cv2
import os
import csv
from collections import Counter
from time import time
import torch
from ultralytics import YOLO
from IdentifiedByPhoto import license_plate_show, validate_and_correct_plate

# Đường dẫn lưu kết quả
output_folder = 'output_camera'
os.makedirs(output_folder, exist_ok=True)

# Đường dẫn lưu file CSV
output_csv_file = os.path.join(output_folder, 'camera_results.csv')
if os.path.exists(output_csv_file):
    os.remove(output_csv_file)
if not os.path.exists(output_csv_file):
    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'License Plate', 'Bounding Box'])

# Hàm kiểm tra biển số đã tồn tại trong CSV hay chưa
def is_plate_in_csv(file_path, plate):
    """Kiểm tra xem biển số đã tồn tại trong file CSV chưa."""

    with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if plate in row:
                return True
    return False

def put_plate_on_image(img, bbox, license_plate_predict):
    x1, y1, x2, y2 = bbox

    # Vẽ khung biển số (màu đỏ)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Chia biển số thành các dòng (nếu có ký tự xuống dòng)
    lines = license_plate_predict.split("\n")

    # Cấu hình font chữ và độ dày
    font_scale = 1.2
    font_thickness = 2
    text_color = (255, 255, 0)  # Màu xanh lá

    # Tính chiều cao dòng text
    text_height = 0
    for line in lines:
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_height = max(text_height, text_size[1])

    # Xác định vị trí hiển thị text (dưới hoặc trên biển số)
    y_text_start = y1 - len(lines) * (text_height + 10) if y1 > len(lines) * (text_height + 10) else y2 + 20

    for idx, line in enumerate(lines):
        # Tính vị trí từng dòng text
        y_text = y_text_start + idx * (text_height + 10)
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Vẽ nền màu đen mờ phía sau văn bản
        text_width = text_size[0]
        text_height = text_size[1]
        padding = 10  # Khoảng cách giữa văn bản và nền
        background_y1 = y_text - text_height - padding
        background_y2 = y_text + padding

        # Vẽ nền màu đen mờ
        overlay = img.copy()  # Tạo bản sao của ảnh để overlay
        cv2.rectangle(
            overlay,
            (x1 - padding, background_y1),
            (x1 + text_width + padding, background_y2),
            (0, 0, 0),  # Màu nền (đen)
            -1  # Độ dày âm để tô đầy hình chữ nhật
        )
        alpha = 0.5  # Độ trong suốt của nền
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Hiển thị từng dòng text lên ảnh
        cv2.putText(
            img,
            line,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

    return img

# Hàm khởi tạo mô hình YOLO
def init_models(model_detect_path='./models/license_plate_detector_v4.pt', model_rec_path='./models/last.pt'):
    # Sử dụng GPU nếu có, nếu không dùng CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  # In ra thiết bị đang sử dụng

    model_detect = YOLO(model_detect_path).to(device)
    model_rec = YOLO(model_rec_path).to(device)
    return model_detect, model_rec

# Hàm nhận diện biển số qua camera
def recognize_license_plate_from_camera():
    cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định
    model_detect, model_rec = init_models()

    plate_counts = []  # Danh sách lưu trữ biển số nhận diện trong mỗi 0.5 giây
    start_time = time()  # Thời gian bắt đầu tính
    plate_time_threshold = 2  # Thời gian kiểm tra (0.5 giây)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break

        results_detect = model_detect(frame, conf=0.4)
        for result in results_detect:
            boxes = result.boxes
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int)
                crop = frame[r[1]:r[3], r[0]:r[2]]
                results_rec = model_rec(crop, conf=0.4)

                labels_list, x_list, y_list = [], [], []
                for rec_result in results_rec:
                    rec_boxes = rec_result.boxes
                    for rec_box in rec_boxes:
                        cls = int(rec_box.cls.cpu().numpy().item())
                        x_center = rec_box.xywh[0][0].cpu().numpy().item()
                        y_center = rec_box.xywh[0][1].cpu().numpy().item()
                        labels_list.append(cls)
                        x_list.append(x_center)
                        y_list.append(y_center)

                if labels_list:
                    plate_predict, is_single_line = license_plate_show(labels_list, x_list, y_list)
                    corrected_plate = validate_and_correct_plate(plate_predict, is_single_line)

                    if corrected_plate:
                        # Thêm biển số vào danh sách
                        plate_counts.append(corrected_plate)

                        # Kiểm tra nếu đã qua 0.5 giây
                        if time() - start_time >= plate_time_threshold:
                            # Kiểm tra các biển số trong danh sách plate_counts
                            plate_count = sum(1 for plate in plate_counts if plate == corrected_plate)

                            # Kiểm tra nếu biển số xuất hiện quá 5 lần
                            if plate_count > 5 and not is_plate_in_csv(output_csv_file, corrected_plate):
                                # Ghi vào file CSV nếu chưa có trong đó
                                with open(output_csv_file, mode='a', newline='', encoding='utf-8') as csv_file:
                                    csv_writer = csv.writer(csv_file)
                                    csv_writer.writerow([cap.get(cv2.CAP_PROP_POS_FRAMES), corrected_plate, r])

                            # Đặt lại danh sách và thời gian
                            plate_counts = []
                            start_time = time()

                        # Hiển thị biển số lên khung hình
                        frame = put_plate_on_image(frame, r, corrected_plate)
        cv2.putText(frame, "stop=q", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                    cv2.LINE_AA)
        # Hiển thị ảnh lên cửa sổ
        cv2.imshow("Camera - Nhận diện biển số xe", frame)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

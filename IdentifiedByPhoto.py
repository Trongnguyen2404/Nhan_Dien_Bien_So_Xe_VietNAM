from ultralytics import YOLO
import numpy as np
import cv2
import os
import csv

# Nhập đường dẫn cần thiết
source_folder = 'C:\\Users\\nguye\\PycharmProjects\\Vietnamese-license-plate-recognition'
input_media = None

# Thư mục lưu kết quả
output_folder = os.path.join(source_folder, 'output_img')
os.makedirs(output_folder, exist_ok=True)

# Đường dẫn lưu file CSV
output_csv_file = os.path.join(output_folder, 'license_plate_results.csv')

# Hàm chuyển đổi số thành chữ cái cho các vị trí cụ thể
def number_to_letter(num):
    mapping = {
        '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
        '5': 'F', '6': 'G', '7': 'H', '8': 'K', '9': 'L'
    }
    return mapping.get(num, num)

def letter_to_number(letter):
    mapping = {
        'A': '4', 'B': '8', 'E': '3', 'G': '6', 'I': '1',
        'O': '0', 'Q': '0', 'S': '5', 'Z': '2'
    }
    return mapping.get(letter, letter)

# Hàm hiển thị biển số
def license_plate_show(labels, x, y):
    labels_order = []
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z']

    predict_table = np.column_stack((labels, x, y))
    y_sorted = sorted(y)
    row_below = []
    row_above = []
    c = y_sorted[0]

    for num in y_sorted:
        if num > 1.8 * c:
            row_below.append(num)
        else:
            row_above.append(num)

    is_single_line = not row_below or not row_above

    if is_single_line:
        x_all = [predict_table[i][1] for i in range(len(predict_table))]
        x_all_sorted = sorted(x_all)
        labels_order = [predict_table[np.where(predict_table[:, 1] == x_val)][0][0] for x_val in x_all_sorted]
    else:
        x_above = [predict_table[np.where(predict_table[:, 2] == y_val)][0][1] for y_val in row_above]
        x_below = [predict_table[np.where(predict_table[:, 2] == y_val)][0][1] for y_val in row_below]

        x_above_sorted = sorted(x_above)
        x_below_sorted = sorted(x_below)
        labels_order_above = [predict_table[np.where(predict_table[:, 1] == x_val)][0][0] for x_val in x_above_sorted]
        labels_order_below = [predict_table[np.where(predict_table[:, 1] == x_val)][0][0] for x_val in x_below_sorted]
        labels_order = labels_order_above + labels_order_below

    plate_chars = ''
    for label_idx in labels_order:
        label_idx = int(label_idx)
        if 0 <= label_idx < len(characters):
            plate_chars += characters[label_idx]

    return plate_chars, is_single_line

# Hàm xác thực và chỉnh sửa biển số
def validate_and_correct_plate(plate, is_single_line):
    plate_clean = plate.replace('-', '').replace('\n', '').replace('.', '')

    # Kiểm tra nếu số ký tự lớn hơn 9
    if len(plate_clean) > 9:
        return None

    # Nếu số ký tự nhỏ hơn 8, trả về None
    if len(plate_clean) < 8:
        return None

    plate_list = list(plate_clean)

    # Kiểm tra và sửa ký tự tại vị trí 2 nếu cần
    if not plate_list[2].isalpha():
        plate_list[2] = number_to_letter(plate_list[2])

    # Duyệt qua các ký tự trong biển số và sửa nếu cần
    for idx in range(len(plate_list)):
        if idx != 2 and idx != 3:
            if not plate_list[idx].isdigit():
                plate_list[idx] = letter_to_number(plate_list[idx])

    # Kiểm tra lại nếu sau khi sửa mà ký tự khác 2 và 3 vẫn là chữ cái
    for idx in range(len(plate_list)):
        if idx != 2 and idx != 3:
            if plate_list[idx].isalpha():  # Nếu có ký tự là chữ cái ở các vị trí khác ngoài 2 và 3
                return None

    # Định dạng lại biển số theo kiểu một dòng hoặc nhiều dòng
    if is_single_line:
        if len(plate_list) == 8:
            formatted_plate = ''.join(plate_list[:3]) + '-' + ''.join(plate_list[3:])
        elif len(plate_list) == 9:
            formatted_plate = ''.join(plate_list[:3]) + '-' + ''.join(plate_list[3:6]) + '.' + ''.join(plate_list[6:])
        else:
            formatted_plate = ''.join(plate_list)
    else:
        if len(plate_list) == 8:
            formatted_plate = ''.join(plate_list[:2]) + '-' + ''.join(plate_list[2:4]) + '\n' + ''.join(plate_list[4:])
        elif len(plate_list) == 9:
            formatted_plate = ''.join(plate_list[:2]) + '-' + ''.join(plate_list[2:4]) + '\n' + ''.join(
                plate_list[4:7]) + '.' + ''.join(plate_list[7:])
        else:
            formatted_plate = ''.join(plate_list)

    return formatted_plate


# Hàm vẽ biển số lên hình ảnh
def put_plate_on_image(img, bbox, license_plate_predict):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    lines = license_plate_predict.split("\n")

    font_scale = 1.2
    font_thickness = 2
    y_show = y1 - 35 if y1 > 20 else y2 + 10

    for line in lines:
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_width, text_height = text_size

        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x1, y_show - text_height - 8),
            (x1 + text_width, y_show + 8),
            (0, 0, 0), -1
        )
        alpha = 0.5
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv2.putText(
            img,
            line,
            (x1, y_show),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 0),
            font_thickness,
            cv2.LINE_AA
        )

        y_show += text_height + 10

    return img

# Hàm kiểm tra xem dòng dữ liệu đã có trong CSV chưa
def is_row_in_csv(file_path, new_row):
    if not os.path.exists(file_path):
        return False
    with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row == new_row:
                return True
    return False

# Hàm chính để xử lý hình ảnh
# Hàm chính để xử lý hình ảnh
def process_image(input_media, source_folder, output_folder, output_csv_file):
    # Kiểm tra nếu không có ảnh đầu vào
    if input_media is None:
        return

    media_path = os.path.join(source_folder, input_media)
    model_detect = YOLO(os.path.join(source_folder, 'models/license_plate_detector_v4.pt'))
    model_rec = YOLO(os.path.join(source_folder, 'models/last.pt'))

    frame = cv2.imread(media_path)
    if frame is None:
        print("Không thể đọc hình ảnh.")
        return
    frame = cv2.resize(frame, (640, 640))
    results_detect = model_detect(frame, conf=0.4)
    annotated_frame = frame.copy()

    original_image_path = os.path.join(output_folder, f"original_{os.path.basename(input_media)}")
    cv2.imwrite(original_image_path, frame)

    with open(output_csv_file, mode='a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if os.stat(output_csv_file).st_size == 0:
            csv_writer.writerow(['File Name', 'Original Image Path', 'License Plate', 'Annotated Image Path'])

        for result in results_detect:
            boxes = result.boxes
            for box in boxes:
                r = box.xyxy[0].cpu().numpy().astype(int)
                crop = frame[r[1]:r[3], r[0]:r[2]]  # Crop the plate area
                results_rec = model_rec(crop, conf=0.4)  # Recognize characters on the plate
                labels_list = []
                x_list = []
                y_list = []

                # Collect the predicted labels and their corresponding x, y coordinates
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
                    # Process the license plate characters
                    plate_predict, is_single_line = license_plate_show(labels_list, x_list, y_list)
                    corrected_plate = validate_and_correct_plate(plate_predict, is_single_line)
                    if corrected_plate:
                        # Annotate the image with the detected plate
                        annotated_frame = put_plate_on_image(annotated_frame, r, corrected_plate)
                        annotated_image_path = os.path.join(output_folder, f"annotated_{os.path.basename(input_media)}")
                        cv2.imwrite(annotated_image_path, annotated_frame)  # Save the annotated image

                        # Check if the row exists in the CSV, if not, add it
                        new_row = [os.path.basename(input_media), original_image_path, corrected_plate,
                                   annotated_image_path]
                        if not is_row_in_csv(output_csv_file, new_row):
                            with open(output_csv_file, mode='a', newline='', encoding='utf-8') as csv_file:
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow(new_row)
                            print(f"Đã thêm dòng: {new_row}")
                        else:
                            print("Dòng dữ liệu đã tồn tại, không thêm.")
                    else:
                        print("Biển số không hợp lệ hoặc không đủ ký tự.")
                else:
                    print("Không nhận dạng được ký tự trên biển số.")

    print(f"Kết quả đã được lưu vào {output_csv_file}")


# Gọi hàm chính để xử lý
process_image(input_media, source_folder, output_folder, output_csv_file)
# utils.py

import numpy as np
import cv2
import csv
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Function to map numbers to letters for specific positions
def number_to_letter(num):
    mapping = {
        '0': 'A',
        '1': 'B',
        '2': 'C',
        '3': 'D',
        '4': 'E',
        '5': 'F',
        '6': 'G',
        '7': 'H',
        '8': 'K',
        '9': 'L'
    }
    return mapping.get(num, num)

def letter_to_number(letter):
    mapping = {
        'A': '4',
        'B': '8',
        'E': '3',
        'G': '6',
        'I': '1',
        'O': '0',
        'Q': '0',
        'S': '5',
        'Z': '2'
    }
    return mapping.get(letter, letter)

def license_plate_show(labels, x, y):
    labels_order = []
    x_above = []
    x_below = []
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z']

    # Combine labels and positions into a table
    predict_table = np.column_stack((labels, x, y))

    # Separate characters into rows based on y position
    y_sorted = sorted(y)
    row_below = []
    row_above = []
    c = y_sorted[0]
    for num in y_sorted:
        if num > 1.8 * c:
            row_below.append(num)
        else:
            row_above.append(num)

    # Determine if the plate is single-line or double-line
    is_single_line = not row_below or not row_above

    # Get x positions for characters in each row
    if is_single_line:
        x_all = [predict_table[i][1] for i in range(len(predict_table))]
        x_all_sorted = sorted(x_all)
        labels_order = [predict_table[np.where(predict_table[:, 1] == x_val)][0][0] for x_val in x_all_sorted]
    else:
        x_above = [predict_table[np.where(predict_table[:, 2] == y_val)][0][1] for y_val in row_above]
        x_below = [predict_table[np.where(predict_table[:, 2] == y_val)][0][1] for y_val in row_below]

        # Sort x positions and get corresponding labels
        x_above_sorted = sorted(x_above)
        x_below_sorted = sorted(x_below)
        labels_order_above = [predict_table[np.where(predict_table[:, 1] == x_val)][0][0] for x_val in x_above_sorted]
        labels_order_below = [predict_table[np.where(predict_table[:, 1] == x_val)][0][0] for x_val in x_below_sorted]
        labels_order = labels_order_above + labels_order_below

    # Build the license plate string without formatting
    plate_chars = ''
    for label_idx in labels_order:
        label_idx = int(label_idx)
        if 0 <= label_idx < len(characters):
            plate_chars += characters[label_idx]

    return plate_chars, is_single_line

def validate_and_correct_plate(plate, is_single_line):
    # Remove formatting characters for validation
    plate_clean = plate.replace('-', '').replace('\n', '').replace('.', '')
    if len(plate_clean) < 8:
        return None  # Không đủ 8 ký tự
    plate_list = list(plate_clean)

    # Ký tự thứ 3 bắt buộc là chữ cái
    if not plate_list[2].isalpha():
        plate_list[2] = number_to_letter(plate_list[2])

    # Ký tự thứ 4 có thể là số hoặc chữ cái, không cần xử lý

    # Các ký tự khác bắt buộc là số
    for idx in range(len(plate_list)):
        if idx != 2 and idx != 3:
            if not plate_list[idx].isdigit():
                plate_list[idx] = letter_to_number(plate_list[idx])

    # Thêm định dạng vào biển số dựa trên độ dài và số dòng
    if is_single_line:
        if len(plate_list) == 8:
            # Định dạng cho biển số 8 ký tự một dòng: XXA-1234
            formatted_plate = ''.join(plate_list[:3]) + '-' + ''.join(plate_list[3:])
        elif len(plate_list) == 9:
            # Định dạng cho biển số 9 ký tự một dòng: XXA-123.45
            formatted_plate = ''.join(plate_list[:3]) + '-' + ''.join(plate_list[3:6]) + '.' + ''.join(plate_list[6:])
        else:
            formatted_plate = ''.join(plate_list)
    else:
        if len(plate_list) == 8:
            # Định dạng cho biển số 8 ký tự hai dòng: XX-A1\n2345
            formatted_plate = ''.join(plate_list[:2]) + '-' + ''.join(plate_list[2:4]) + '\n' + ''.join(plate_list[4:])
        elif len(plate_list) == 9:
            # Định dạng cho biển số 9 ký tự hai dòng: XX-A1\n234.56
            formatted_plate = ''.join(plate_list[:2]) + '-' + ''.join(plate_list[2:4]) + '\n' + ''.join(plate_list[4:7]) + '.' + ''.join(plate_list[7:])
        else:
            formatted_plate = ''.join(plate_list)

    return formatted_plate

def read_characters(model_rec, crop):
    # Recognize characters in the cropped plate
    results_rec = model_rec(crop, conf=0.4)

    # Process recognition results
    labels_list = []
    x_list = []
    y_list = []

    for rec_result in results_rec:
        rec_boxes = rec_result.boxes
        for rec_box in rec_boxes:
            cls = int(rec_box.cls.cpu().numpy().item())
            x_center = rec_box.xywh[0][0].cpu().numpy().item()
            y_center = rec_box.xywh[0][1].cpu().numpy().item()
            labels_list.append(cls)
            x_list.append(x_center)
            y_list.append(y_center)

    return labels_list, x_list, y_list

def get_car(license_plate_bbox, vehicle_track_ids):
    x1, y1, x2, y2 = license_plate_bbox

    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle
        if x1 >= xcar1 and y1 >= ycar1 and x2 <= xcar2 and y2 <= ycar2:
            return vehicle  # Trả về phương tiện chứa biển số
    return None

def write_csv(results, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                data = results[frame_nmr][car_id]
                writer.writerow({
                    'frame_nmr': frame_nmr,
                    'car_id': car_id,
                    'car_bbox': data['car_bbox'],
                    'license_plate_bbox': data['license_plate_bbox'],
                    'license_plate_text': data['license_plate_text']
                })

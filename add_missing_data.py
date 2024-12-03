# add_missing_data.py
import csv
import numpy as np
from scipy.interpolate import interp1d
def load_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data
def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split(','))) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split(','))) for row in data])
    license_plate_texts = np.array([row['license_plate_text'] for row in data])
    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_data = car_bboxes[car_mask]
        license_plate_bboxes_data = license_plate_bboxes[car_mask]
        license_plate_texts_data = license_plate_texts[car_mask]
        full_frame_numbers = np.arange(car_frame_numbers.min(), car_frame_numbers.max() + 1)
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []
        license_plate_texts_interpolated = []
        if len(car_frame_numbers) == 1:
            car_bboxes_interpolated = np.tile(car_bboxes_data[0], (len(full_frame_numbers), 1))
            license_plate_bboxes_interpolated = np.tile(license_plate_bboxes_data[0], (len(full_frame_numbers), 1))
        else:
            # Interpolate bounding boxes
            for i in range(4):  # For each coordinate in bbox
                interp_func = interp1d(car_frame_numbers, car_bboxes_data[:, i], kind='linear',
                                       fill_value='extrapolate')
                car_bboxes_interpolated.append(interp_func(full_frame_numbers))
                interp_func_lp = interp1d(car_frame_numbers, license_plate_bboxes_data[:, i], kind='linear',
                                          fill_value='extrapolate')
                license_plate_bboxes_interpolated.append(interp_func_lp(full_frame_numbers))

            car_bboxes_interpolated = np.stack(car_bboxes_interpolated, axis=1)
            license_plate_bboxes_interpolated = np.stack(license_plate_bboxes_interpolated, axis=1)

        # Interpolate license plate texts with the most frequent value
        unique_texts, counts = np.unique(license_plate_texts_data, return_counts=True)
        most_common_text = unique_texts[np.argmax(counts)]
        license_plate_texts_interpolated = [most_common_text] * len(full_frame_numbers)
        # Append interpolated data for each frame
        for idx, frame_number in enumerate(full_frame_numbers):
            row = {
                'frame_nmr': str(int(frame_number)),
                'car_id': str(car_id),
                'car_bbox': ','.join(map(str, car_bboxes_interpolated[idx])),
                'license_plate_bbox': ','.join(map(str, license_plate_bboxes_interpolated[idx])),
                'license_plate_text': license_plate_texts_interpolated[idx]
            }
            interpolated_data.append(row)
    return interpolated_data
def write_csv(data, output_path):
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_text']
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)
def main(input_file, output_file):
    data = load_csv(input_file)
    interpolated_data = interpolate_bounding_boxes(data)
    write_csv(interpolated_data, output_file)
if __name__ == "__main__":
    input_file = './output_video/results.csv'  # Đường dẫn tệp CSV đầu vào
    output_file = './output_video/results_interpolated.csv'  # Đường dẫn tệp CSV đầu ra
    main(input_file, output_file)

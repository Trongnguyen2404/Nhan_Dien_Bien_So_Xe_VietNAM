import cv2
import pandas as pd
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=2, line_length=20):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)
    return img
def apply_overlay(frame, x1_lp, y1_lp, x2_lp, y2_lp, color=(0, 0, 255), alpha=0.4):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1_lp, y1_lp), (x2_lp, y2_lp), color, -1)  # Màu đỏ
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
def put_license_plate_text(frame, plate_text, x1_lp, y1_lp):
    lines = plate_text.split('\n')
    font_scale = 1.5
    thickness = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(lines[0], font, font_scale, thickness)
    line_height = text_height + baseline
    max_text_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines])
    total_text_height = len(lines) * line_height
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x1_lp, y1_lp - total_text_height - 10),
                  (x1_lp + max_text_width, y1_lp + 10),
                  (0, 0, 0),
                  -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    # Vẽ văn bản lên hình ảnh
    for i, line in enumerate(lines):
        y_position = y1_lp - 10 - (len(lines) - i - 1) * line_height
        cv2.putText(frame, line, (x1_lp, y_position), font, font_scale, (255, 255, 0), thickness)

    return frame


def process_frame(frame, results, frame_nmr):
    df_ = results[results['frame_nmr'] == frame_nmr]
    for idx, row in df_.iterrows():
        car_bbox = list(map(float, row['car_bbox'].split(',')))
        x1, y1, x2, y2 = map(int, car_bbox)
        lp_bbox = list(map(float, row['license_plate_bbox'].split(',')))
        x1_lp, y1_lp, x2_lp, y2_lp = map(int, lp_bbox)
        frame = apply_overlay(frame, x1_lp, y1_lp, x2_lp, y2_lp)
        draw_border(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.rectangle(frame, (x1_lp, y1_lp), (x2_lp, y2_lp), (0, 0, 255), 2)
        plate_text = row['license_plate_text']
        frame = put_license_plate_text(frame, plate_text, x1_lp, y1_lp)
    return frame
def process_video(input_video_path, output_video_path, results):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0 or width == 0 or height == 0:
        print("Error: Invalid video properties.")
        return
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_nmr = -1
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            frame = process_frame(frame, results, frame_nmr)
            out.write(frame)
        else:
            print(f"Failed to read frame {frame_nmr}")
    out.release()
    cap.release()
def load_results(csv_path):
    return pd.read_csv(csv_path, dtype={'frame_nmr': int})
def main(input_video_path, output_video_path, results_csv_path):
    results = load_results(results_csv_path)
    process_video(input_video_path, output_video_path, results)
if __name__ == "__main__":
    input_video_path = './sample3.mp4'  # Đường dẫn video đầu vào
    output_video_path = './output_video/output_video.mp4'  # Đường dẫn video đầu ra
    results_csv_path = './output_video/results_interpolated.csv'  # Đường dẫn tệp CSV chứa kết quả
    main(input_video_path, output_video_path, results_csv_path)

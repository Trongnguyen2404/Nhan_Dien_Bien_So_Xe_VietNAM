import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk
import csv
import os
import threading

from IdentifiedByPhoto import process_image
from IdentifiedViaCamera import recognize_license_plate_from_camera
from Video_recognition import main as process_video_from_file
from add_missing_data import main as add_missing_data_main
from visualize import main as visualize_main

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Biển Số Xe")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)
        self.refresh_task_id = None  # Biến lưu ID của tác vụ after
        self.video_running = False  # Biến để theo dõi trạng thái video

        # Tạo giao diện
        self.create_widgets()

    def create_widgets(self):
        """Tạo giao diện của ứng dụng"""
        # Tiêu đề
        title_label = tk.Label(self.root, text="Hệ thống nhận diện biển số xe phát triển bởi nhóm 10",
                               font=("Arial", 16, "bold"), bg="black", fg="white")
        title_label.place(x=0, y=0, relwidth=1, height=40)

        # Nút chọn phương thức nhận diện (Ảnh, Camera, Video)
        self.btn_camera = tk.Button(self.root, text="Nhận diện biển số xe qua camera", width=30,
                                    command=self.process_camera)
        self.btn_video = tk.Button(self.root, text="Nhận diện biển số xe trong video", width=30,
                                   command=self.process_video)
        self.btn_image = tk.Button(self.root, text="Nhận diện biển số xe trong ảnh", width=30,
                                   command=self.process_image)

        self.btn_camera.place(x=10, y=50)
        self.btn_video.place(x=350, y=50)
        self.btn_image.place(x=690, y=50)

        # Khung hiển thị ảnh/video đã nhận diện
        self.image_label = tk.Label(self.root, text="Khu vực hiển thị kết quả", bg="white", width=90, height=20,
                                    relief="solid")
        self.image_label.place(x=10, y=100, width=650, height=400)

        # Danh sách biển số đã nhận diện
        list_label = tk.Label(self.root, text="Danh sách biển số đã nhận diện", font=("Arial", 14))
        list_label.place(x=680, y=100)

        self.license_plate_listbox = tk.Listbox(self.root, font=("Arial", 12), width=30, height=20)
        self.license_plate_listbox.place(x=680, y=140, width=300, height=360)

        # Thêm sự kiện khi chọn item trong Listbox
        self.license_plate_listbox.bind("<<ListboxSelect>>", self.display_annotated_image)

        # Tiến trình và nút hiển thị video
        self.progress_label = tk.Label(self.root, text="Tiến trình: ", font=("Arial", 12), fg="blue")
        self.show_video_button = tk.Button(self.root, text="Hiển thị Video", command=self.display_full_video)

    def toggle_progress_widgets(self, show):
        """Hiển thị hoặc ẩn các widget tiến trình và hiển thị video"""
        if show:
            self.progress_label.place(x=10, y=520, width=650, height=30)  # Vị trí của label tiến trình
            self.show_video_button.place(x=10, y=560, width=150, height=30)  # Nút hiển thị video
        else:
            self.progress_label.place_forget()
            self.show_video_button.place_forget()

    def display_full_video(self):
        self.stop_refresh_task()
        """Hiển thị toàn bộ video chạy"""
        output_video_path = './output_video/output_video.mp4'
        self.display_output_video(output_video_path)

    def stop_refresh_task(self):
        """Hủy tác vụ refresh danh sách biển số"""
        if self.refresh_task_id:
            self.root.after_cancel(self.refresh_task_id)
            self.refresh_task_id = None  # Đặt lại ID tác vụ

    def display_output_video(self, video_file):
        """Hiển thị video output trên giao diện Tkinter"""
        cap = cv2.VideoCapture(video_file)
        self.video_running = True  # Đánh dấu video đang chạy

        def update_frame():
            if not self.video_running:
                cap.release()
                return  # Dừng video nếu video_running là False

            ret, frame = cap.read()
            if not ret:
                cap.release()
                return  # Video đã kết thúc

            # Chuyển đổi frame thành ảnh RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((650, 400))  # Điều chỉnh kích thước hiển thị

            # Chuyển ảnh thành ImageTk để hiển thị trên Tkinter
            self.displayed_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.displayed_image)
            self.image_label.image = self.displayed_image  # Giữ ảnh để tránh bị thu hồi bộ nhớ

            # Tiếp tục cập nhật từng frame trong video
            self.root.after(30, update_frame)  # Cập nhật mỗi 30ms

        update_frame()  # Bắt đầu hiển thị video

    def process_camera(self):
        if self.video_running:
            # Nếu video đang chạy, dừng video
            self.stop_video()
        self.toggle_progress_widgets(False)
        """Xử lý nhận diện biển số qua camera"""

        # Xóa ảnh đã nhận diện hiển thị
        self.image_label.config(image='', text="Khu vực hiển thị kết quả")
        self.image_label.image = None

        # Xóa danh sách biển số đã nhận diện
        self.license_plate_listbox.delete(0, tk.END)

        threading.Thread(target=self.run_camera_recognition).start()

    def run_camera_recognition(self):
        """Chạy nhận diện biển số qua camera và tải kết quả"""
        self.load_license_plates_for_camera()  # Tải biển số ngay khi bắt đầu
        self.refresh_task_id = self.root.after(3000, self.refresh_license_plate_list)  # Lưu ID của tác vụ after

        # Bắt đầu nhận diện biển số qua camera
        recognize_license_plate_from_camera()

    def refresh_license_plate_list(self):
        """Tải lại danh sách biển số từ CSV sau mỗi 3s"""
        self.load_license_plates_for_camera()  # Tải lại biển số từ file CSV

        # Tiếp tục gọi lại sau 3 giây
        self.refresh_task_id = self.root.after(3000, self.refresh_license_plate_list)

    def process_image(self):
        self.stop_refresh_task()
        """Xử lý ảnh và hiển thị kết quả"""
        if self.video_running:
            # Nếu video đang chạy, dừng video
            self.stop_video()
        # Ẩn các widget tiến trình khi chọn ảnh
        self.toggle_progress_widgets(False)
        # Mở hộp thoại chọn ảnh
        input_file = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if input_file:
            # Đặt thư mục đầu ra và tệp CSV
            output_folder = "C:\\Users\\nguye\\PycharmProjects\\Vietnamese-license-plate-recognition\\output_img"
            output_csv_file = os.path.join(output_folder, "license_plate_results.csv")

            # Xử lý ảnh và nhận diện biển số
            process_image(input_file, "C:\\Users\\nguye\\PycharmProjects\\Vietnamese-license-plate-recognition",
                          output_folder, output_csv_file)

            # Đọc ảnh đã nhận diện (ảnh đã được chú thích)
            annotated_image_path = os.path.join(output_folder, f"annotated_{os.path.basename(input_file)}")

            # Kiểm tra xem ảnh đã nhận diện có tồn tại không
            if os.path.exists(annotated_image_path):
                # Mở ảnh đã chú thích
                img = Image.open(annotated_image_path)
                img.thumbnail((650, 400))  # Điều chỉnh kích thước hiển thị

                # Hiển thị ảnh lên giao diện
                self.displayed_image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.displayed_image)
                self.image_label.image = self.displayed_image  # Giữ ảnh để tránh bị thu hồi bộ nhớ

                # Tải lại danh sách biển số
                self.load_license_plates_for_photo()
            else:
                messagebox.showerror("Lỗi", "Không thể tìm thấy ảnh đã chú thích.")

    def stop_video(self):
        """Dừng video nếu nó đang hiển thị"""
        self.video_running = False
        self.image_label.config(image=None)  # Tắt video bằng cách xóa ảnh hiện tại

    def process_video(self):
        """Xử lý nhận diện biển số qua video"""
        video_file = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video files", "*.mp4 *.avi")])
        threading.Thread(target=self.run_video_recognition, args=(video_file,)).start()

    def run_video_recognition(self, video_file):
        if video_file:
            # Hiển thị các widget tiến trình khi chọn video
            self.toggle_progress_widgets(True)
        """Chạy nhận diện biển số qua video"""
        self.stop_refresh_task()
        # Tạo đối tượng VideoCapture để đọc video
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video.")
            return

        # Lấy frame đầu tiên từ video
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể đọc frame đầu tiên của video.")
            cap.release()
            return

        # Chuyển đổi frame thành ảnh RGB để hiển thị
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((650, 400))  # Điều chỉnh kích thước hiển thị

        # Chuyển ảnh thành ImageTk để hiển thị trên Tkinter
        self.displayed_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.displayed_image)
        self.image_label.image = self.displayed_image  # Giữ ảnh để tránh bị thu hồi bộ nhớ

        # Cập nhật tiến trình: 10% - Đang đọc video
        self.progress_label.config(text="Đang xử lý video... Tiến trình: 10%")

        # Bắt đầu xử lý video
        # 1. Chạy process_video_from_file (Ví dụ nhận diện biển số)

        results = process_video_from_file(video_file)
        self.progress_label.config(text="Đang nhận diện biển số trong video... Tiến trình: 30%")
        # 2. Sau khi nhận diện xong, gọi add_missing_data.py để xử lý dữ liệu

        intermediate_csv_file = './output_video/results.csv'
        output_csv_file = './output_video/results_interpolated.csv'
        add_missing_data_main(intermediate_csv_file, output_csv_file)
        self.progress_label.config(text="Đang xử lý dữ liệu... Tiến trình: 60%")
        # 3. Cuối cùng, visualize video và lưu kết quả
        self.progress_label.config(text="Đang tạo video đầu ra... Tiến trình: 90%")
        self.load_license_plates_for_video()
        output_video_path = './output_video/output_video.mp4'
        visualize_main(video_file, output_video_path, output_csv_file)
        # Bắt đầu hiển thị video đầu ra trên Tkinter
        self.display_output_video(output_video_path)
        # Cập nhật tiến trình cuối cùng sau khi hoàn thành
        self.progress_label.config(text="Nhận diện biển số xong! Đã hoàn thành.")

        cap.release()

    def load_license_plates_for_photo(self):
        """Tải danh sách biển số từ file CSV của ảnh"""
        self.license_plate_listbox.delete(0, tk.END)
        self.license_plate_images = {}  # Dictionnary lưu trữ ảnh đã nhận diện tương ứng với biển số

        output_csv_file = 'C:\\Users\\nguye\\PycharmProjects\\Vietnamese-license-plate-recognition\\output_img\\license_plate_results.csv'

        if os.path.exists(output_csv_file):
            with open(output_csv_file, mode='r', newline='', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Bỏ qua header
                for row in csv_reader:
                    license_plate = row[2]  # Cột License Plate
                    original_image_path = row[1]  # Cột Đường dẫn ảnh gốc
                    annotated_image_path = row[3]  # Cột Đường dẫn ảnh đã nhận diện

                    # Thay thế dấu xuống dòng bằng dấu cách trong biển số
                    license_plate = license_plate.replace('\n', ' ')

                    # Lưu thông tin ảnh tương ứng với biển số
                    self.license_plate_images[license_plate] = annotated_image_path

                    # Thêm biển số vào danh sách
                    self.license_plate_listbox.insert(tk.END, license_plate)

    def load_license_plates_for_camera(self):
        """Tải danh sách biển số từ file CSV của camera"""
        self.license_plate_listbox.delete(0, tk.END)

        output_csv_file = "output_camera/camera_results.csv"

        if os.path.exists(output_csv_file):
            with open(output_csv_file, mode='r', newline='', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)
                for row in csv_reader:
                    license_plate = row[1]
                    license_plate = license_plate.replace('\n', ' ')
                    self.license_plate_listbox.insert(tk.END, license_plate)

    def load_license_plates_for_video(self):
        """Tải danh sách biển số và khung hình từ kết quả nhận diện video"""
        self.license_plate_listbox.delete(0, tk.END)
        output_csv_file = "output_video/results_interpolated.csv"

        if os.path.exists(output_csv_file):
            unique_license_plates = set()  # Tập hợp để lưu biển số duy nhất
            with open(output_csv_file, mode='r', newline='', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Bỏ qua dòng tiêu đề
                for row in csv_reader:
                    frame_nmr = row[0]  # Cột đầu tiên là frame_nmr
                    license_plate = row[4]
                    license_plate = license_plate.replace('\n', ' ')

                    # Kết hợp frame_nmr và biển số
                    entry = f"{frame_nmr}: {license_plate}"

                    # Chỉ thêm vào nếu biển số chưa có trong tập hợp
                    if license_plate not in unique_license_plates:
                        unique_license_plates.add(license_plate)
                        self.license_plate_listbox.insert(tk.END, entry)

    def display_annotated_image(self, event):
        """Hiển thị ảnh biển số đã nhận diện khi người dùng click vào danh sách"""
        selected_idx = self.license_plate_listbox.curselection()
        if not selected_idx:
            return

        selected_license_plate = self.license_plate_listbox.get(selected_idx)

        # Nếu chọn biển số từ video (có khung hình)
        if ":" in selected_license_plate:
            frame_number = selected_license_plate.split(":")[0].strip()
            self.display_frame_from_video(frame_number)
        else:
            # Nếu chọn biển số từ ảnh
            annotated_image_path = self.license_plate_images.get(selected_license_plate)
            if annotated_image_path and os.path.exists(annotated_image_path):
                img = Image.open(annotated_image_path)
                img.thumbnail((650, 400))
                self.displayed_image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.displayed_image)
                self.image_label.image = self.displayed_image

    def display_frame_from_video(self, frame_number):
        if self.video_running:
            # Nếu video đang chạy, dừng video
            self.stop_video()
        """Hiển thị video tại khung hình đã chọn từ danh sách biển số video"""
        video_file = './output_video/output_video.mp4'
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video.")
            return

        # Tìm đến khung hình cần hiển thị
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()

        if not ret:
            messagebox.showerror("Lỗi", f"Không thể đọc khung hình {frame_number}.")
            cap.release()
            return

        # Chuyển đổi frame thành ảnh RGB để hiển thị
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((650, 400))  # Điều chỉnh kích thước hiển thị

        # Chuyển ảnh thành ImageTk để hiển thị trên Tkinter
        self.displayed_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.displayed_image)
        self.image_label.image = self.displayed_image  # Giữ ảnh để tránh bị thu hồi bộ nhớ

        cap.release()  # Giải phóng video sau khi hiển thị


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()

# 🚗 YOLOv8 Obstacle Detection Dashboard

Đây là dự án cuối kỳ môn **Chuyên đề Học máy**. Ứng dụng web này sử dụng mô hình học sâu **YOLOv8** kết hợp với **Flask** và **OpenCV** để nhận diện vật cản giao thông theo thời gian thực qua Camera và Video.

## ✨ Tính năng nổi bật

Dự án được thiết kế với giao diện Dark Mode hiện đại (Bootstrap 5) và chia thành 3 tính năng chính:

1. **📷 Live Camera Detection (Nhận diện Camera trực tiếp)**
   - Sử dụng Web API để lấy luồng camera từ thiết bị người dùng (hỗ trợ cả Laptop và Camera trước/sau của Điện thoại).
   - Truyền tải dữ liệu frame-by-frame (Canvas Blob) tối ưu băng thông.
   - **Tích hợp OpenCV:** Cung cấp tùy chọn bật/tắt bộ lọc tiền xử lý (Gaussian Blur & Contrast Enhancement) giúp cải thiện chất lượng ảnh thiếu sáng trước khi đưa vào YOLO.

2. **🎬 Video Processing & Object Tracking (Phân tích Video)**
   - Cho phép tải lên video MP4 và phát luồng nhận diện (MJPEG Stream).
   - Hỗ trợ tính năng tự động lặp (Auto-loop) và nút Phát lại (Replay).
   - **Trích xuất vật thể:** Tự động cắt (crop) và lưu lại hình ảnh của các vật thể được phát hiện với độ tin cậy (confidence) cao nhất, cập nhật trực tiếp lên giao diện theo thời gian thực (Polling).

3. **📊 Training Metrics Dashboard (Báo cáo Huấn luyện)**
   - Hiển thị trực quan các KPI của mô hình: **Precision, Recall, mAP50, mAP50-95**.
   - Trình bày các biểu đồ phân tích (Confusion Matrix, F1-Curve, Loss) với tính năng Click để phóng to (Bootstrap Modal Lightbox).

## 📁 Cấu trúc thư mục

```text
yolo_web_local/
│
├── app.py                  # Mã nguồn chính của server Flask
├── train_model.py          # Script huấn luyện YOLO trên Google Colab
├── obstacle_run_1/         # Thư mục chứa trọng số mô hình
│   └── weights/
│       └── best.pt         # Model YOLOv8 đã được train
│
├── static/
│   ├── metrics/            # Chứa hình ảnh biểu đồ huấn luyện (results.png,...)
│   └── uploads/            # Thư mục lưu tạm video và ảnh cắt vật thể
│
├── templates/              # Thư mục chứa giao diện HTML (Jinja2)
│   ├── base.html           # Layout gốc, chứa Navbar và thư viện
│   ├── camera.html         # Giao diện Tab Camera
│   ├── video.html          # Giao diện Tab Video
│   └── metrics.html        # Giao diện Tab Báo cáo
│
└── README.md
```

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Yêu cầu hệ thống
Đảm bảo máy tính của bạn đã cài đặt Python 3.8+ và các thư viện cần thiết:
```bash
pip install flask opencv-python ultralytics numpy
```

### 2. Khởi chạy ứng dụng
Mở Terminal/Command Prompt tại thư mục dự án và chạy lệnh:
```bash
python app.py
```

### 3. Truy cập ứng dụng
- **Trên máy tính cục bộ:** Mở trình duyệt và truy cập `http://localhost:5000`
- **Trên điện thoại (qua mạng LAN/Internet):** 
  Do yêu cầu bảo mật của trình duyệt, camera điện thoại không thể mở qua `http` IP LAN. Để giải quyết, hãy sử dụng **Ngrok** để tạo đường hầm HTTPS:
  ```bash
  # Mở terminal mới và chạy:
  ngrok http 5000
  ```
  Sau đó copy đường link `https://xxxx.ngrok.app` gửi qua điện thoại để trải nghiệm đầy đủ tính năng Camera.

## 🛠 Tech Stack
* **AI/Machine Learning:** YOLOv8 (Ultralytics)
* **Computer Vision:** OpenCV (`cv2`), Numpy
* **Backend:** Python, Flask
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API, Canvas), Bootstrap 5


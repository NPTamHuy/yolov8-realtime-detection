import cv2
import numpy as np
import os
import uuid
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO

app = Flask(__name__)

# Cấu hình thư mục lưu trữ
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO('obstacle_run_1/weights/best.pt')

# Biến toàn cục lưu trữ DANH SÁCH các vật thể có độ chính xác cao nhất
# Cấu trúc: {"CAR": [det_1, det_2, ...], "PERSON": [det_1, det_2, ...]}
class_detections = {}
MAX_HISTORY = 6  # Số lượng ảnh tối đa lưu lại cho mỗi class (1 ảnh gốc + 5 ảnh lịch sử)


def process_frame(frame, apply_filter=False, is_video=False):
    global class_detections

    if apply_filter:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

    results = model(frame, conf=0.5, imgsz=416)

    # Nếu là luồng Video, tiến hành trích xuất danh sách hình ảnh
    if is_video:
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name not in class_detections:
                    class_detections[cls_name] = []

                history = class_detections[cls_name]

                # Lưu ảnh nếu danh sách chưa đầy, hoặc độ chính xác mới cao hơn ảnh kém nhất trong danh sách
                if len(history) < MAX_HISTORY or conf > history[-1]['confidence']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        crop_img = frame[y1:y2, x1:x2]
                        img_filename = f"{cls_name}_{uuid.uuid4().hex[:8]}.jpg"
                        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
                        cv2.imwrite(img_path, crop_img)

                        new_det = {
                            "class_name": cls_name,
                            "confidence": conf,
                            "confidence_display": round(conf * 100, 2),
                            "bbox": [x1, y1, x2, y2],
                            "image_url": f"/static/uploads/{img_filename}"
                        }

                        history.append(new_det)
                        # Sắp xếp danh sách giảm dần theo độ chính xác
                        history.sort(key=lambda x: x['confidence'], reverse=True)
                        # Chỉ giữ lại top MAX_HISTORY ảnh tốt nhất
                        class_detections[cls_name] = history[:MAX_HISTORY]

    annotated_frame = results[0].plot()
    return annotated_frame


# --- CÁC ĐỊNH TUYẾN GIAO DIỆN ---
@app.route('/')
def index():
    return render_template('camera.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/image')
def image():
    return render_template('image.html')


@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/metrics')
def metrics():
    return render_template('metrics.html')


# --- CÁC API XỬ LÝ ---
@app.route('/detect', methods=['POST'])
def detect():
    try:
        use_filter = request.form.get('use_filter') == 'true'
        file = request.files['image'].read()
        nparr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Dữ liệu ảnh không hợp lệ'}), 400

        annotated_frame = process_frame(frame, apply_filter=use_filter, is_video=False)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global class_detections
    class_detections.clear()  # Reset lại lịch sử khi có video mới

    if 'video' not in request.files:
        return jsonify({'error': 'Không tìm thấy file video'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
    file.save(filepath)
    return jsonify({'message': 'Tải lên thành công', 'video_url': '/video_feed'})


def generate_video_frames():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
    cap = cv2.VideoCapture(filepath)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        annotated_frame = process_frame(frame, apply_filter=False, is_video=True)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_best_detections')
def get_best_detections():
    # Trả về toàn bộ dictionary (chứa list các ảnh)
    return jsonify(class_detections)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'File ảnh không hợp lệ'}), 400

        annotated_frame = process_frame(frame, apply_filter=False, is_video=False)

        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, annotated_frame)

        return jsonify({
            'message': 'Thành công',
            'image_url': f"/static/uploads/{result_filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
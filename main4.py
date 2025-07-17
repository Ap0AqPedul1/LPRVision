from ultralytics import YOLO
import cv2
import threading
import queue

# Load model YOLOv8 untuk kendaraan dan detektor plat nomor
vehicle_model = YOLO("yolov8s.pt")
license_plate_model = YOLO("license_plate_detector.pt")

vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
license_plate_classes = ['license_plate']  # sesuaikan jika nama kelas berbeda di model plat nomor

# rtsp_url = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
rtsp_url = 0

frame_queue = queue.Queue(maxsize=5)
stop_flag = False

def read_frames():
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buffer kecil untuk real-time
    if not cap.isOpened():
        print("Gagal membuka RTSP stream")
        global stop_flag
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari RTSP stream")
            break

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put(frame)

    cap.release()

def process_frames():
    global stop_flag
    while not stop_flag:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        small_frame = cv2.resize(frame, (640, 360))

        # Deteksi kendaraan
        vehicle_results = vehicle_model(small_frame)
        vehicle_detections = []
        for result in vehicle_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = vehicle_model.names[cls_id]
                if cls_name in vehicle_classes:
                    vehicle_detections.append(box)

        # Deteksi plat nomor
        license_plate_results = license_plate_model(small_frame)
        license_plate_detections = []
        for result in license_plate_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = license_plate_model.names[cls_id]
                if cls_name in license_plate_classes:
                    license_plate_detections.append(box)

        annotated_frame = small_frame.copy()

        # Gambar bounding box kendaraan
        for box in vehicle_detections:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            cls_name = vehicle_model.names[cls_id]
            conf = box.conf[0].cpu().numpy()
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Gambar bounding box plat nomor dengan warna berbeda
        for box in license_plate_detections:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            cls_name = license_plate_model.names[cls_id]
            conf = box.conf[0].cpu().numpy()
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("YOLO RTSP Vehicle and License Plate Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break

thread_read = threading.Thread(target=read_frames)
thread_process = threading.Thread(target=process_frames)

thread_read.start()
thread_process.start()

thread_read.join()
thread_process.join()

cv2.destroyAllWindows()

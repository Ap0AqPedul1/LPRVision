import cv2
from ultralytics import YOLO
import time

# URL RTSP kamera
rtsp_url  = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"

# Load model YOLOv8s untuk deteksi kendaraan
model_vehicle = YOLO("yolov8s.pt")

# Load model khusus untuk deteksi plat nomor
model_plate = YOLO("license_plate_detector.pt")

# Kelas kendaraan default dari model YOLOv8s
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Gagal membuka stream RTSP, periksa URL dan koneksi.")
    exit()

time.sleep(2)  # Tunggu stream stabil

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mendapatkan frame, mencoba ulang...")
        time.sleep(1)
        continue

    # Deteksi kendaraan pertama dengan YOLOv8s
    results_vehicle = model_vehicle(frame)
    detections_vehicle = results_vehicle[0]

    annotated_frame = frame.copy()

    # Loop deteksi kendaraan
    for box, cls in zip(detections_vehicle.boxes.xyxy.cpu().numpy(), detections_vehicle.boxes.cls.cpu().numpy()):
        label = model_vehicle.names[int(cls)]
        if label in vehicle_classes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Crop area kendaraan untuk deteksi plat nomor
            vehicle_crop = frame[y1:y2, x1:x2]

            # Deteksi plat nomor dalam crop kendaraan
            results_plate = model_plate(vehicle_crop)
            detections_plate = results_plate[0]

            for plate_box in detections_plate.boxes.xyxy.cpu().numpy():
                px1, py1, px2, py2 = map(int, plate_box)
                # Koordinat plat nomor relatif pada crop, sesuaikan ke frame utama
                cv2.rectangle(annotated_frame, (x1+px1, y1+py1), (x1+px2, y1+py2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "Plate", (x1+px1, y1+py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Deteksi Kendaraan dan Plat Nomor", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

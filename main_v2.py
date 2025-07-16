from ultralytics import YOLO
import cv2

# Load model YOLO Anda
license_plate_detector = YOLO("license_plate_detector.pt")

# Ganti URL ini dengan alamat stream RTSP kamera Anda
rtsp_url  = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"

# Buka video stream RTSP
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Gagal membuka RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari RTSP stream")
        break

    # Resize frame agar lebih kecil (misal width=640), tetap pertahankan aspect ratio
    width = 640
    height = int(frame.shape[0] * (width / frame.shape[1]))
    frame_resized = cv2.resize(frame, (width, height))

    # Lakukan deteksi dan tracking plat nomor pada frame yang sudah diresize
    results = license_plate_detector.track(frame_resized, persist=True)

    # Gambar bounding box di frame yang diresize
    for plate in results:
        for bbox in plate.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Tampilkan frame yang sudah diresize dengan deteksi realtime
    cv2.imshow("RTSP License Plate Detection", frame_resized)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

# Load model YOLOv8 yang sudah Anda train
model = YOLO('License Plate Recognition.v11i.yolov8/runs/detect/train/weights/best.pt')

# Fungsi untuk mendeteksi plate nomor pada gambar
def detect_plate(image_path):
    # Baca gambar
    img = cv2.imread(image_path)

    # Lakukan prediksi/deteksi menggunakan model
    results = model(img)

    # Ambil hasil deteksi (bounding boxes, confidence, class, dll)
    for result in results:
        boxes = result.boxes  # bounding boxes
        for box in boxes:
            # Bounding box koordinat
            xyxy = box.xyxy[0].cpu().numpy()  # format (xmin, ymin, xmax, ymax)
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()

            # Tampilkan bounding box dan confidence pada gambar
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img, f"Conf: {conf:.2f}", (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan hasil deteksi
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contoh panggilan fungsi dengan file gambar yang ingin diperiksa
# detect_plate('License Plate Recognition.v11i.yolov8/test/images/20-cdmx2017policia-c_jpg.rf.80df9245b5dd6f7e54d1d48ca2241e2f.jpg')
detect_plate('dsa.png')  # Ganti dengan path gambar yang sesuaia
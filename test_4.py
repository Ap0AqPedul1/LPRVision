from ultralytics import YOLO
import cv2

# Load model YOLOv8
model = YOLO("yolo/plate.pt")

# Daftar gambar
images = ["image/test.jpg", "image/test_3.jpg"]

for img_path in images:
    img = cv2.imread(img_path)

    # Deteksi
    results = model(img)

    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            # koordinat bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # gambar bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{model.names[cls]} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # simpan hasil crop
            crop = img[y1:y2, x1:x2]
            cv2.imwrite(f"hasil_crop_{img_path}_{i}.jpg", crop)

    # Simpan gambar dengan bounding box
    cv2.imwrite(f"hasil_deteksi_{img_path}", img)
    cv2.imshow(f"Hasil {img_path}", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

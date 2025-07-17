from ultralytics import YOLO
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
import cv2
import easyocr
import numpy as np

# Load model YOLO
model = YOLO('license_plate_detector.pt')

# Inisialisasi EasyOCR reader
reader = easyocr.Reader(['en'])  # Bisa ditambah bahasa lain jika perlu

# Contoh kelas custom dengan BaseSolution (optional, bisa disesuaikan)
class LicensePlateSolution(BaseSolution):
    def postprocess(self, results):
        return results

def detect_license_plate(image_path):
    # Baca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Jalankan deteksi
    results = model(img)

    # Gunakan Annotator untuk menggambar bounding box dengan warna berbeda
    annotator = Annotator(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            # Crop area plat nomor untuk OCR
            license_plate_img = img[y1:y2, x1:x2]

            # Konversi ke grayscale untuk pembacaan OCR optimal
            gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

            # Gunakan EasyOCR untuk membaca teks plat nomor
            ocr_result = reader.readtext(gray)
            text = ''
            if ocr_result:
                # Gabungkan semua hasil teks yang terbaca
                text = ' '.join([res[1] for res in ocr_result])

            # Warna dari palet
            c = colors(int(box.cls[0]), True)
            label = f'{text} {conf:.2f}' if text else f'Plate {conf:.2f}'

            annotator.box_label([x1, y1, x2, y2], label, color=c)

    # Hasil gambar dengan annotasi
    annotated_img = annotator.result()

    # Tampilkan hasil dengan OpenCV
    cv2.imshow('Deteksi dan OCR Plat Nomor', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_license_plate('capture_frame_10.jpg')

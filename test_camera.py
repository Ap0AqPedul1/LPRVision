import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import easyocr
import numpy as np
from inference_sdk import InferenceHTTPClient


RTSP_URL = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
MODEL_PATH = "yolo/plate.pt"  # ganti path jika perlu
OUTPUT_DIR = "output"

# ---------- Utilitas ----------
def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def draw_box_with_label(img, xyxy, label, conf=None, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    text = f"{label}"
    if conf is not None:
        text += f" {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
    cv2.putText(img, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

# ---------- Inisialisasi Model ----------
print("[INFO] Loading YOLO model...")
yolo = YOLO(MODEL_PATH)

print("[INFO] Loading EasyOCR (English)...")
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True jika kamu pakai GPU yang sudah ready

ensure_dir(OUTPUT_DIR)

# ---------- Buka RTSP ----------
print("[INFO] Opening RTSP stream...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)  # FFMPEG biasanya paling stabil untuk RTSP

if not cap.isOpened():
    # Coba sekali lagi setelah delay singkat
    time.sleep(1.0)
    cap.open(RTSP_URL)

if not cap.isOpened():
    raise RuntimeError("Gagal membuka RTSP stream. Cek URL/kredensial/koneksi kamera.")

print("[INFO] Stream terbuka. Tekan 'c' untuk capture, 'q' untuk keluar.")

# ---------- Loop Tampilan ----------
last_frame = None
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        # Coba reconnect ringan
        cap.release()
        time.sleep(0.5)
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        continue

    last_frame = frame.copy()
    cv2.imshow("RTSP View (press c to capture, q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        # --------- Capture & Proses ---------
        if last_frame is None:
            continue

        stamp = ts()
        raw_path = os.path.join(OUTPUT_DIR, f"capture_{stamp}.jpg")
        cv2.imwrite(raw_path, last_frame)
        print(f"[INFO] Captured: {raw_path}")

        # YOLO inference pada frame yang di-capture
        results = yolo(last_frame, verbose=False)

        annotated = last_frame.copy()
        crop_dir = os.path.join(OUTPUT_DIR, f"plates_{stamp}")
        ensure_dir(crop_dir)

        plate_count = 0
        recognized_texts = []

        for r in results:
            boxes = r.boxes
            names = r.names if hasattr(r, 'names') else {}
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else None
                cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else None
                cls_name = names.get(cls_id, "plate" if cls_id is not None else "obj")

                # Kalau model punya kelas lain, kita filter hanya 'plate' jika ada
                if "plate" in [n.lower() for n in names.values()] and cls_name.lower() != "plate":
                    continue

                # Crop plat
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(annotated.shape[1], x2), min(annotated.shape[0], y2)
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = annotated[y1c:y2c, x1c:x2c]

                plate_count += 1
                crop_path = os.path.join(crop_dir, f"plate_{i}_{stamp}.jpg")
                cv2.imwrite(crop_path, crop)

                # Preprocess ringan untuk OCR (opsional: grayscale + sedikit threshold)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # adaptif bisa membantu, tapi jangan terlalu agresif
                # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                #                              cv2.THRESH_BINARY, 31, 15)

                # EasyOCR
                ocr_result = reader.readtext(gray)
                # Gabungkan text yang terdeteksi jadi satu baris ringkas
                text_joined = " ".join([t[1] for t in ocr_result]) if len(ocr_result) > 0 else ""
                recognized_texts.append(text_joined.strip())

                # Gambar bounding box + label di annotated
                pretty = text_joined if text_joined else "PLATE"
                draw_box_with_label(annotated, (x1, y1, x2, y2), pretty, conf=conf)

        annotated_path = os.path.join(OUTPUT_DIR, f"annotated_{stamp}.jpg")
        cv2.imwrite(annotated_path, annotated)

        # Tampilkan hasil di jendela terpisah
        show_img = annotated.copy()
        info_lines = [
            f"Captured: {os.path.basename(raw_path)}",
            f"Detections: {plate_count}",
        ]
        if recognized_texts:
            info_lines += [f"OCR[{i+1}]: {txt}" for i, txt in enumerate(recognized_texts)]

        # Render info text di gambar
        y0 = 30
        for line in info_lines:
            cv2.putText(show_img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(show_img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 28

        cv2.imshow("Captured & Processed", show_img)
        print(f"[INFO] Annotated saved: {annotated_path}")
        if plate_count:
            for i, txt in enumerate(recognized_texts, start=1):
                print(f"[OCR {i}] {txt if txt else '(no text)'}")
        else:
            print("[INFO] Tidak ada plat terdeteksi.")

# ---------- Cleanup ----------
cap.release()
cv2.destroyAllWindows()
print("[INFO] Selesai.")

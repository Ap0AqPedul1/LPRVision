import cv2
import os
import time
import numpy as np
import re
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import easyocr


custom_configuration = InferenceConfiguration(confidence_threshold=0.1)
# ====== KONFIG ======
RTSP_URL = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
MODEL_ID = "license-plate-recognition-rxg4e/11"
MIN_CONF = 0.1           # abaikan deteksi di bawah ini
PAD = 6                   # padding crop biar huruf tidak kepotong

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="EheH0mexmA60NICeX1rp"
)
CLIENT.configure(custom_configuration)

# EasyOCR (gpu=False aman untuk banyak mesin Windows)
reader = easyocr.Reader(['en'], gpu=False)

os.makedirs("output", exist_ok=True)
last_annotated = None  # simpan hasil anotasi terakhir

def safe_crop(img, x1, y1, x2, y2, pad=0):
    H, W = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # kurangi noise tapi tetap jaga edge
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
    # adaptif threshold untuk kondisi cahaya beragam
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    return th

def ocr_text(crop):
    if crop is None:
        return ""
    proc = preprocess_for_ocr(crop)
    # EasyOCR: detail=0 langsung balikin list string
    texts = reader.readtext(proc, detail=0, paragraph=True)
    if not texts:
        # coba tanpa threshold (kadang lebih bagus)
        texts = reader.readtext(crop, detail=0, paragraph=True)
    if not texts:
        return ""
    # gabung & bersihkan hanya alfanumerik + dash (opsional)
    raw = " ".join(texts)
    clean = re.sub(r"[^A-Z0-9\- ]", "", raw.upper())
    # rapikan spasi
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def annotate_with_roboflow(img):
    """
    Infer Roboflow dari frame numpy (via file sementara),
    kembalikan (img_annot, predictions_list_dengan_text).
    """
    tmp_path = "output/frame_tmp.jpg"
    cv2.imwrite(tmp_path, img)

    result = CLIENT.infer(tmp_path, model_id=MODEL_ID)
    H, W = img.shape[:2]

    # Skala kalau ukuran result['image'] berbeda
    rf_w = result.get("image", {}).get("width", W)
    rf_h = result.get("image", {}).get("height", H)
    scale_x = W / rf_w if rf_w else 1.0
    scale_y = H / rf_h if rf_h else 1.0

    preds = result.get("predictions", [])
    # urutkan biar yang paling yakin dulu
    preds = sorted(preds, key=lambda p: p.get("confidence", 0.0), reverse=True)

    img_annot = img.copy()
    enriched = []

    for p in preds:
        conf = float(p.get("confidence", 0.0))
        if conf < MIN_CONF:
            continue

        # bbox center-based dari Roboflow: (x, y, w, h)
        cx = p.get("x", 0) * scale_x
        cy = p.get("y", 0) * scale_y
        w  = p.get("width", 0) * scale_x
        h  = p.get("height", 0) * scale_y

        x1 = int(cx - w/2); y1 = int(cy - h/2)
        x2 = int(cx + w/2); y2 = int(cy + h/2)
        cls = p.get("class", "obj")

        # crop + OCR
        crop = safe_crop(img, x1, y1, x2, y2, pad=PAD)
        text = ocr_text(crop)

        # gambar bbox
        color = (0, 255, 0)
        cv2.rectangle(img_annot, (x1, y1), (x2, y2), color, 2)

        # label: class + conf + (jika ada) teks plate
        label = f"{cls} {conf:.2f}"
        if text:
            label = f"{text} | {label}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(img_annot, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(img_annot, label, (x1 + 3, top + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

        # simpan versi enriched (tambahkan 'text')
        p_en = dict(p)
        p_en["text"] = text
        p_en["bbox_xyxy"] = [x1, y1, x2, y2]
        enriched.append(p_en)

    return img_annot, enriched

def main():
    global last_annotated

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka RTSP stream")

    print("[INFO] Tekan 'c' = capture+infer (tampilkan), 's' = simpan hasil, 'q' = keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame RTSP kosong, retry...")
            time.sleep(0.1)
            continue

        # tampilkan live
        try:
            cv2.imshow("RTSP Live", frame)
        except cv2.error:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('c'):
            print("[INFO] Capture… infer Roboflow + EasyOCR…")
            try:
                annotated, preds = annotate_with_roboflow(frame)
                last_annotated = annotated

                # log ringkas ke console
                # contoh: [{'x':..., 'y':..., ..., 'text': 'H 1234 AB'}]
                print("[INFO] Prediksi + OCR:", preds)

                # tampilkan hasil anotasi
                try:
                    cv2.imshow("Hasil Deteksi Plat", annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
            except Exception as e:
                print(f"[ERR] Gagal infer/tampil: {e}")

        elif key == ord('s'):
            if last_annotated is not None:
                out_path = "output/azhari_annotated.jpg"
                cv2.imwrite(out_path, last_annotated)
                print(f"[INFO] Hasil disimpan: {out_path}")
            else:
                print("[INFO] Belum ada hasil anotasi untuk disimpan (tekan 'c' dulu).")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

if __name__ == "__main__":
    main()

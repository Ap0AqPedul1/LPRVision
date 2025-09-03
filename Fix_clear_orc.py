import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
import easyocr

# Load YOLO model untuk deteksi plat
model_plate = YOLO("yolo/plate.pt")

# Load EasyOCR (bahasa Indonesia + Inggris)
reader = easyocr.Reader(['en'])

# RTSP URL
rtsp_url = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"

# Queue antar thread
frame_queue = queue.Queue(maxsize=5)

# Variabel global untuk simpan teks terakhir
last_text = ""

def rtsp_reader():
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def yolo_detector():
    global last_text
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            results = model_plate(frame, stream=True)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # Kotak deteksi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{model_plate.names[cls]} {conf:.2f}",
                                (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    # Crop plat
                    crop_plate = frame[y1:y2, x1:x2]
                    if crop_plate.size > 0:
                        # Grayscale
                        gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)
                        hsv = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2HSV)

                        # Threshold Otsu
                        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        inverted = cv2.bitwise_not(thresh)

                        # Deteksi warna dominan
                        lower_yellow = np.array([20, 100, 100])
                        upper_yellow = np.array([40, 255, 255])
                        lower_white = np.array([0, 0, 200])
                        upper_white = np.array([180, 30, 255])
                        lower_red1 = np.array([0, 100, 100])
                        upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([160, 100, 100])
                        upper_red2 = np.array([180, 255, 255])
                        lower_black = np.array([0, 0, 0])
                        upper_black = np.array([180, 255, 50])

                        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                        mask_white = cv2.inRange(hsv, lower_white, upper_white)
                        mask_red = cv2.bitwise_or(
                            cv2.inRange(hsv, lower_red1, upper_red1),
                            cv2.inRange(hsv, lower_red2, upper_red2)
                        )
                        mask_black = cv2.inRange(hsv, lower_black, upper_black)

                        colors = {
                            "hitam": cv2.countNonZero(mask_black),
                            "putih": cv2.countNonZero(mask_white),
                            "kuning": cv2.countNonZero(mask_yellow),
                            "merah": cv2.countNonZero(mask_red)
                        }
                        dominant_color = max(colors, key=colors.get)
                        print("Warna plat dominan:", dominant_color)

                        if dominant_color == "hitam":
                            final_plate = inverted
                        else:
                            final_plate = thresh

                        # EasyOCR hanya bisa baca dari file atau array RGB
                        final_plate_rgb = cv2.cvtColor(final_plate, cv2.COLOR_GRAY2RGB)
                        ocr_result = reader.readtext(final_plate_rgb, detail=0)

                        if ocr_result:
                            last_text = " ".join(ocr_result)
                            print("Plat Terdeteksi:", last_text)

                        # Tampilkan crop
                        cv2.imshow("Plat Gray", gray)
                        cv2.imshow("Plat Threshold", final_plate)

            # Tampilkan video deteksi
            cv2.imshow("Deteksi Plat", frame)

            # Buat tab baru untuk teks OCR
            ocr_display = np.ones((200, 400, 3), dtype=np.uint8) * 255
            cv2.putText(ocr_display, f"OCR: {last_text}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("OCR Text", ocr_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

# Jalankan thread
t1 = threading.Thread(target=rtsp_reader, daemon=True)
t2 = threading.Thread(target=yolo_detector, daemon=True)

t1.start()
t2.start()
t1.join()
t2.join()

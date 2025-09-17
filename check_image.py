import cv2
import numpy as np
import asyncio
from ultralytics import YOLO
import easyocr
import os
import json

class PlateDetectorOCR:
    def __init__(self, model_path="plate.pt", crop_config="crop_config.json"):
        # Load YOLOv8
        self.model = YOLO(model_path)
        # EasyOCR
        self.reader = easyocr.Reader(['en'])
        self.last_plate_crop = None
        self.last_text = ""

        # Buat folder output kalau belum ada
        os.makedirs("output", exist_ok=True)

        # Load konfigurasi crop jika ada
        self.crop_cfg = None
        if os.path.exists(crop_config):
            with open(crop_config, "r") as f:
                self.crop_cfg = json.load(f)
            print("[INFO] Crop config loaded:", self.crop_cfg)

    def apply_crop_config(self, img):
        """Crop manual berdasarkan crop_config.json"""
        if not self.crop_cfg:
            return img
        x, y, w, h = (
            self.crop_cfg["x"],
            self.crop_cfg["y"],
            self.crop_cfg["w"],
            self.crop_cfg["h"],
        )
        return img[y:y+h, x:x+w]

    def detect_plate(self, img_path):
        """Deteksi plat pada gambar dan simpan crop terakhir"""
        img = cv2.imread(img_path)

        # Jika nama file ada 'back', crop manual dulu
        if "back" in os.path.basename(img_path).lower() and self.crop_cfg:
            img = self.apply_crop_config(img)
            print("[INFO] Gambar dicrop sesuai crop_config.json")

        results = self.model(img)

        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Crop plat
                crop = img[y1:y2, x1:x2]
                self.last_plate_crop = crop

                # Simpan hasil crop
                crop_name = f"output/crop_{i}_{os.path.basename(img_path)}"
                cv2.imwrite(crop_name, crop)
                print(f"[SAVE] {crop_name}")

                # Gambar kotak di gambar asli
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        save_name = f"output/deteksi_{os.path.basename(img_path)}"
        cv2.imwrite(save_name, img)
        print(f"[SAVE] {save_name}")

        cv2.imshow("Deteksi Plat", img)
        cv2.waitKey(1)

    def process_ocr(self, img_name=""):
        """OCR + simpan hasil threshold + simpan crop OCR"""
        if self.last_plate_crop is None:
            print("Belum ada crop plat.")
            return
        
        gray = cv2.cvtColor(self.last_plate_crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.last_plate_crop, cv2.COLOR_BGR2HSV)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        inverted = cv2.bitwise_not(thresh)

        # Range warna untuk cek dominasi
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

        # Simpan hasil threshold
        threshold_name = f"output/threshold_{os.path.basename(img_name)}"
        cv2.imwrite(threshold_name, final_plate)
        print(f"[SAVE] {threshold_name}")

        # OCR
        final_plate_rgb = cv2.cvtColor(final_plate, cv2.COLOR_GRAY2RGB)
        ocr_result = self.reader.readtext(final_plate_rgb, detail=0)

        if ocr_result:
            self.last_text = " ".join(ocr_result)
            print("Plat Terdeteksi:", self.last_text)
            asyncio.run(self.send_to_clients(self.last_text))

            # Tambahkan teks hasil OCR ke crop asli
            ocr_crop = self.last_plate_crop.copy()
            cv2.putText(
                ocr_crop,
                self.last_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Simpan hasil OCR crop
            ocr_name = f"output/ocr_{os.path.basename(img_name)}"
            cv2.imwrite(ocr_name, ocr_crop)
            print(f"[SAVE] {ocr_name}")

            # Tampilkan hasil
            cv2.imshow("Plat dengan OCR", ocr_crop)

        cv2.imshow("Plat Threshold", final_plate)
        cv2.waitKey(1)

    async def send_to_clients(self, text):
        """Dummy async function (bisa diganti sesuai kebutuhan)"""
        print(f"[SEND] ke client: {text}")


if __name__ == "__main__":
    detector = PlateDetectorOCR("yolo/plate.pt")

    # Daftar gambar
    images = ["image/test_123.jpg"]

    for img in images:
        detector.detect_plate(img)
        detector.process_ocr(img_name=img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

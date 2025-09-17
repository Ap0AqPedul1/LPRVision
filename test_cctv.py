import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
import easyocr
import asyncio
import websockets

class PlateRecognizer:
    def __init__(self, rtsp_url, ws_port=8765, yolo_model_path="yolo/plate.pt"):
        self.rtsp_url = rtsp_url
        self.model_plate = YOLO(yolo_model_path)
        self.reader = easyocr.Reader(['en'])
        self.frame_queue = queue.Queue(maxsize=5)
        self.last_text = ""
        self.last_plate_crop = None
        self.running = True
        self.ws_port = ws_port
        self.clients = set()

    async def ws_handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                pass  # Server tidak menerima pesan dari client
        finally:
            self.clients.remove(websocket)

    async def send_to_clients(self, text):
        if self.clients:
            await asyncio.wait([client.send(text) for client in self.clients])

    def start_websocket_server(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(self.ws_handler, "0.0.0.0", self.ws_port)
        loop.run_until_complete(start_server)
        loop.run_forever()

    def rtsp_reader(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        cap.release()

    def process_ocr(self):
        if self.last_plate_crop is None:
            return
        
        gray = cv2.cvtColor(self.last_plate_crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.last_plate_crop, cv2.COLOR_BGR2HSV)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(thresh)

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

        final_plate_rgb = cv2.cvtColor(final_plate, cv2.COLOR_GRAY2RGB)
        ocr_result = self.reader.readtext(final_plate_rgb, detail=0)

        if ocr_result:
            self.last_text = " ".join(ocr_result)
            print("Plat Terdeteksi:", self.last_text)
            asyncio.run(self.send_to_clients(self.last_text))

        cv2.imshow("Plat Threshold", final_plate)

    def yolo_detector(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                results = self.model_plate(frame, stream=True)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{self.model_plate.names[cls]} {conf:.2f}",
                                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)

                        crop_plate = frame[y1:y2, x1:x2]
                        if crop_plate.size > 0:
                            self.last_plate_crop = crop_plate

                cv2.imshow("Deteksi Plat", frame)

                ocr_display = np.ones((200, 400, 3), dtype=np.uint8) * 255
                cv2.putText(ocr_display, f"OCR: {self.last_text}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow("OCR Text", ocr_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.process_ocr()
                elif key == ord('q'):
                    self.running = False
                    break
        cv2.destroyAllWindows()

    def run(self):
        t1 = threading.Thread(target=self.rtsp_reader, daemon=True)
        t2 = threading.Thread(target=self.yolo_detector, daemon=True)
        t3 = threading.Thread(target=self.start_websocket_server, daemon=True)
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()


if __name__ == "__main__":
    rtsp_url = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
    recognizer = PlateRecognizer(rtsp_url)
    recognizer.run()

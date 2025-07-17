import cv2
import threading
import re
import easyocr
from inference_sdk import InferenceHTTPClient

class InferenceWorker:
    def __init__(self, api_url, api_key, model_id):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id
        self.image = None
      
    def infer_image(self, image_path):
        result = self.client.infer(image_path, model_id=self.model_id)
        # print("Inferensi: ", result["predictions"])
        return result["predictions"]

    def read_image(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan di path: {image_path}")
        print(f"Gambar berhasil dibaca: {image_path}")

    def detection(self, points):
        if self.image is None:
            raise ValueError("Image belum dibaca. Panggil read_image() terlebih dahulu.")
        for detection in points:
            center_x, center_y = detection['x'], detection['y']
            width, height = detection['width'], detection['height']
           

            x_min = int(center_x - (width / 2))
            y_min = int(center_y - (height / 2))
            x_max = int(center_x + (width / 2))
            y_max = int(center_y + (height / 2))


            image_detection = self.image[y_min:y_max, x_min:x_max]
            cv2.imwrite('cropped_photo.jpg', image_detection)

class InferenceWorker_2():
    def __init__(self, api_url, api_key, model_id):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id
        self.image = None

    def read_image(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan di path: {image_path}")
        print(f"Gambar berhasil dibaca: {image_path}")

    def infer_image(self, image_path):
        result = self.client.infer(image_path, model_id="license-ocr-qqq6v/3")
        # print(result['predictions'])
        return result['predictions']

          
class RTSPStreamHandler:
    def __init__(self, rtsp_url, inference_worker, inference_worker_2):
        self.rtsp_url = rtsp_url
        self.inference_worker = inference_worker
        self.inference_worker_2 = inference_worker_2
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.read_stream, daemon=True).start()
        self.main_loop()

    def read_stream(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Gagal membaca frame dari RTSP")
                self.running = False
                break
            # Resize frame agar tidak terlalu besar
            frame = cv2.resize(frame, (640, 480))
            with self.lock:
                self.frame = frame


    def infer_frame_thread(self, frame):
        temp_img_path = "temp.jpg"
        cv2.imwrite(temp_img_path, frame)
        self.inference_worker.read_image(temp_img_path)
        predictions = self.inference_worker.infer_image(temp_img_path)
        self.inference_worker.detection(predictions)
        
        predictions_2 = self.inference_worker_2.infer_image("cropped_photo.jpg")
        # print("predictions OCR:", predictions_2)

        # Filter y <= 100, lalu urut berdasarkan x
        filtered_sorted = sorted([item for item in predictions_2 if item['y'] <= 100], key=lambda x: x['x'])

        # Ambil hanya class-nya
        sorted_classes = [item['class'] for item in filtered_sorted]
        joined_classes = ''.join(sorted_classes)

        # Output
        print(sorted_classes)   # ['A', 'D', '6', '1', '7', '4', 'C', 'U']
        print(joined_classes)   # AD6174CU

    def main_loop(self):
        while self.running:
            with self.lock:
                frame = self.frame.copy() if self.frame is not None else None
            if frame is None:
                continue

            cv2.imshow("RTSP Stream", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Jalankan inferensi pada thread terpisah agar UI tidak hang
                threading.Thread(target=self.infer_frame_thread, args=(frame,), daemon=True).start()

            if key == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rtsp_url = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
    
    api_url = "https://detect.roboflow.com"
    api_key = "EheH0mexmA60NICeX1rp"
    model_id = "license-plate-recognition-rxg4e/11"

    api_url_2="https://detect.roboflow.com"
    api_key_2="EheH0mexmA60NICeX1rp"
    model_id_2="license-ocr-qqq6v/3"

    worker = InferenceWorker(api_url, api_key, model_id)
    worker_2 = InferenceWorker_2(api_url_2, api_key_2, model_id_2)
    stream_handler = RTSPStreamHandler(rtsp_url, worker, worker_2)
    stream_handler.start()

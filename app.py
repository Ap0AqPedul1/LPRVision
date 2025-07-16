from inference_sdk import InferenceHTTPClient
import cv2
import sys
from config import *
import easyocr
import re

class ImageReader:

    def __init__(self,image):
        self.image_path = image
        self.image = None
        self.image_detection = []
        self.reader = easyocr.Reader(['en', 'id'])
        _ = self.set_client(self.image_path)
        self.read_image(self.image_path)
        self.detection(_)
        self.image_show()

    def set_client(self,image):
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="EheH0mexmA60NICeX1rp"
        )
        confidence_threshold = 0.5
        result = CLIENT.infer(image, model_id="license-plate-recognition-rxg4e/6", confidence_threshold=confidence_threshold)
        print(result[predictions])
        return result[predictions]

    def read_image(self,image):
        self.image = cv2.imread(image)
        if self.image is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan di path: {self.image_path}")
        print(f"Gambar berhasil dibaca: {self.image_path}")
        
    def detection(self,point):
        for detection in point:
            center_x, center_y = detection['x'], detection['y']
            width, height = detection['width'], detection['height']
            confidence = detection['confidence']
            label = detection['class']
            
            x_min = int(center_x - (width / 2))
            y_min = int(center_y - (height / 2))
            x_max = int(center_x + (width / 2))
            y_max = int(center_y + (height / 2))
          
            cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #self.image_detection.append(self.image[y_min:y_max, x_min:x_max])
            image_detection = self.image[y_min:y_max, x_min:x_max]
            text = self.read_image_to_text(image_detection)
            cv2.putText(self.image, text, (x_min, y_min - 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    def read_image_to_text(self,image):
        results = self.reader.readtext(image)
        check_plate=[]
        for (bbox, text, confidence) in results:
            check_plate.append(text)
            print(f"Text: {text}, Confidence: {confidence}")

        for plat in check_plate:
            if self.is_plat_nomor(plat):
                return plat
            else:
                return check_plate[0]
        return text
    
    def is_plat_nomor(self,plat):
        # Menghapus spasi ekstra
        plat = plat.replace(" ", "")  
        # Regex untuk mencocokkan plat nomor dengan format yang benar
        match = re.match(r"^([A-Z]{1,2})\d{1,4}[A-Z]{1,3}$", plat)
        if match:
            kode = match.group(1)
            return kode in plat_nomor_indonesia
        return False


    def image_show(self):
        if self.image is None:
            raise ValueError("Gambar belum dibaca. Gunakan 'read_image()' terlebih dahulu.")
        cv2.imshow("window_name", self.image)
        cv2.waitKey(0)  # Tunggu hingga tombol ditekan
        cv2.destroyAllWindows()
    



if __name__ == "__main__":
    image_path = "image/dsa.jpg"
    reader = ImageReader(image_path)
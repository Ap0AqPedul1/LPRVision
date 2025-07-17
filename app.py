from inference_sdk import InferenceHTTPClient
import cv2
import sys
from config import *
import easyocr
import re

class worker_plate:
    def __init__(self,image):
        self.image_path = image
        self.image = None
        self.image_detection = []
        _ = self.set_client(self.image_path)
   

    def set_client(self,image):
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="EheH0mexmA60NICeX1rp"
        )
      
        result = CLIENT.infer(image, model_id="license-plate-recognition-rxg4e/11")

        print(result[predictions])
        return result[predictions]

class worker_ocr:
    def __init__(self,image):
        self.image_path = image
        self.image = None
        self.image_detection = []
        _ = self.set_client(self.image_path)
   

    def set_client(self,image):
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="EheH0mexmA60NICeX1rp"
        )
      
        result = CLIENT.infer(image, model_id="license-ocr-qqq6v/3")

        print(result[predictions])
        return result[predictions]

if __name__ == "__main__":
    
    image_path = "temp.jpg"
    reader = worker_plate(image_path)

    image_path = "cropped_photo.jpg"
    reader = worker_ocr(image_path)
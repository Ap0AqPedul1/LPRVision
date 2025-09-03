import cv2
import numpy as np

# 1. Baca gambar
image = cv2.imread('test.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. Threshold dan inverted
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(thresh)

# 3. Rentang HSV warna
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

# 4. Masking
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(hsv, lower_white, upper_white)
mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                          cv2.inRange(hsv, lower_red2, upper_red2))
mask_black = cv2.inRange(hsv, lower_black, upper_black)

# 5. Hitung jumlah piksel masing-masing warna
colors = {
    "hitam": cv2.countNonZero(mask_black),
    "putih": cv2.countNonZero(mask_white),
    "kuning": cv2.countNonZero(mask_yellow),
    "merah": cv2.countNonZero(mask_red)
}

# 6. Tentukan warna dominan
dominant_color = max(colors, key=colors.get)
print("Warna plat dominan:", dominant_color)

# 7. Simpan hasil sesuai warna
if dominant_color == "hitam":
    cv2.imwrite("output.jpg", inverted)
    print("Disimpan: output.jpg (hasil terbalik karena plat hitam)")
else:
    cv2.imwrite("output.jpg", thresh)
    print("Disimpan: output.jpg (hasil threshold normal)")

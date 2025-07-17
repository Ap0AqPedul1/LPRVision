import cv2
import numpy as np

def nothing(x):
    pass

# Baca gambar
image = cv2.imread('cropped_photo.jpg')
if image is None:
    print("Error: File gambar 'cropped_photo.jpg' tidak ditemukan.")
    exit()

# Ubah ke ruang warna HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Buat window untuk trackbar
cv2.namedWindow('Trackbars')

# Buat trackbar untuk lower HSV
cv2.createTrackbar('LowerH', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('LowerS', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('LowerV', 'Trackbars', 0, 255, nothing)

# Buat trackbar untuk upper HSV
cv2.createTrackbar('UpperH', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('UpperS', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('UpperV', 'Trackbars', 255, 255, nothing)

while True:
    # Ambil posisi trackbar
    lower_h = cv2.getTrackbarPos('LowerH', 'Trackbars')
    lower_s = cv2.getTrackbarPos('LowerS', 'Trackbars')
    lower_v = cv2.getTrackbarPos('LowerV', 'Trackbars')
    upper_h = cv2.getTrackbarPos('UpperH', 'Trackbars')
    upper_s = cv2.getTrackbarPos('UpperS', 'Trackbars')
    upper_v = cv2.getTrackbarPos('UpperV', 'Trackbars')

    lower_color = np.array([lower_h, lower_s, lower_v])
    upper_color = np.array([upper_h, upper_s, upper_v])

    # Buat mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Temukan kontur pada mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tampilkan mask dan hasil deteksi
    cv2.imshow('Mask', mask)
    cv2.imshow('Detected Objects', output)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

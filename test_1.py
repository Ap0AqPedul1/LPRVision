import cv2
import numpy as np
import time

# Membuat gambar kosong
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Font dan pengaturan
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 2
font_thickness = 3
text_color = (255, 255, 255)
position = (50, 250)

# Inisialisasi teks yang akan diperbarui
counter = 0

while True:
    # Membuat gambar baru di setiap iterasi untuk menghindari artefak
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # Menambahkan teks yang diperbarui
    text = f'Tulisan Rapi {counter}'
    cv2.putText(image, text, position, font, font_scale, text_color, font_thickness)

    # Menampilkan gambar
    cv2.imshow('Image with Updated Text', image)

    # Menunggu selama 1 detik
    time.sleep(1)
    
    # Mengupdate teks
    counter += 1

    # Menunggu event untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup jendela
cv2.destroyAllWindows()

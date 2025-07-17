import cv2

rtsp_url  = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
cap = cv2.VideoCapture(0)

# Set frame rate yang diinginkan, misal 15 fps (jika kamera mendukung)
fps = 15
# cap.set(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Gagal membuka RTSP stream")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari RTSP stream")
        break

    cv2.imshow("RTSP Stream Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Keluar dari loop jika tekan 'q'
        break
    elif key == ord('c'):
        # Capture frame ketika 'c' ditekan dan simpan ke file
        filename = f"capture_frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame captured dan disimpan sebagai {filename}")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

import cv2

rtsp_url  = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Gagal membuka RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari RTSP stream")
        break

    cv2.imshow("RTSP Stream Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

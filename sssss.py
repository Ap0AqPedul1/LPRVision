import cv2

rtsp_url  = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Tidak bisa membuka stream RTSP")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter('rekaman_live.avi', fourcc, fps, (width, height))

print("Tekan 'q' untuk berhenti rekaman dan keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame tidak diterima, keluar...")
        break
    
    cv2.imshow("Live RTSP Camera", frame)
    output.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()

import socket
import cv2
import pickle
import struct
import time
import json

rtsp_url = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"

cap = cv2.VideoCapture(rtsp_url)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)
print("Menunggu koneksi client...")

client_socket, addr = server_socket.accept()
print("Client terhubung:", addr)

# Terima resolusi
res_data = client_socket.recv(1024).decode().strip()
try:
    width, height = map(int, res_data.lower().split('x'))
except:
    width, height = 640, 480
    print("Format tidak valid. Gunakan default 640x480.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dummy data plat nomor
    dummy_plate = "AB1234CD"
    x1, y1, x2, y2 = 100, 100, 250, 150

    # Gambar kotak dan plat di frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, dummy_plate, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === Tampilkan di sisi server ===
    cv2.imshow("Server View (Original)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Resize sesuai permintaan client
    resized_frame = cv2.resize(frame, (width, height))

    # Encode frame
    encoded, buffer = cv2.imencode('.jpg', resized_frame)
    frame_data = pickle.dumps(buffer)

    # Data JSON
    plate_data = json.dumps({"plate": dummy_plate}).encode()

    try:
        # Kirim panjang + frame
        client_socket.sendall(struct.pack("Q", len(frame_data)))
        client_socket.sendall(frame_data)

        # Kirim panjang + JSON
        client_socket.sendall(struct.pack("Q", len(plate_data)))
        client_socket.sendall(plate_data)
    except Exception as e:
        print("Client terputus:", e)
        break

cap.release()
client_socket.close()
cv2.destroyAllWindows()
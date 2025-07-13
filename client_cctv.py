import socket
import cv2
import pickle
import struct
import json

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.150', 9999))  # Ganti IP sesuai server

# ==== Minta resolusi ====
client_socket.sendall(b"320x240")  # Ganti resolusi jika perlu

def recv_all(sock, length):
    data = b""
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            return None
        data += more
    return data

while True:
    # Ambil panjang frame
    packed_frame_size = recv_all(client_socket, 8)
    if not packed_frame_size:
        break
    frame_size = struct.unpack("Q", packed_frame_size)[0]
    frame_data = recv_all(client_socket, frame_size)

    # Ambil panjang JSON
    packed_json_size = recv_all(client_socket, 8)
    if not packed_json_size:
        break
    json_size = struct.unpack("Q", packed_json_size)[0]
    json_data = recv_all(client_socket, json_size)

    # Decode
    frame = cv2.imdecode(pickle.loads(frame_data), cv2.IMREAD_COLOR)
    plate = json.loads(json_data.decode())

    # Cetak plat nomor dummy
    print("Nomor Plat:", plate.get("plate", "-"))

    # Tampilkan frame
    cv2.imshow("Client Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cv2.destroyAllWindows()

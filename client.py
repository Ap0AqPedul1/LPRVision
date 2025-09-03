import cv2
import asyncio
import threading
import websockets

# URL RTSP Kamera
RTSP_URL = "rtsp://admin:adybangun12@192.168.0.64:554/Streaming/Channels/101"
WS_URL = "ws://localhost:8765"  # WebSocket Server dari PlateRecognizer

# Variabel global untuk menyimpan hasil OCR terakhir
ocr_text = ""

async def websocket_listener():
    global ocr_text
    async with websockets.connect(WS_URL) as websocket:
        while True:
            message = await websocket.recv()
            ocr_text = message  # Update OCR text setiap terima dari server

def start_ws_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_listener())

def video_player():
    global ocr_text
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca stream RTSP")
            break

        # Tambahkan teks OCR di atas video
        cv2.putText(frame, f"OCR: {ocr_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RTSP Stream + OCR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Jalankan WebSocket listener di thread terpisah
    threading.Thread(target=start_ws_thread, daemon=True).start()

    # Jalankan video player
    video_player()

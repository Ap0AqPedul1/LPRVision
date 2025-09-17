import cv2
import json
import os

# path file
img_path = "image/test_4.jpg"      # ganti dengan gambar awal
json_path = "crop_config.json"       # file untuk simpan koordinat crop
output_crop = "back_car_crop.jpg"    # hasil crop

# baca gambar
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Gambar {img_path} tidak ditemukan!")

clone = img.copy()

# variabel global untuk crop manual
ref_point = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # gambar kotak di layar
        cv2.rectangle(clone, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", clone)


# =========================
# STEP 1: Crop manual + simpan JSON
# =========================
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", clone)
    key = cv2.waitKey(1) & 0xFF

    # tekan 'r' untuk reset gambar
    if key == ord("r"):
        clone = img.copy()
        ref_point = []

    # tekan 'c' untuk crop dan simpan
    elif key == ord("c"):
        if len(ref_point) == 2:
            x1, y1 = ref_point[0]
            x2, y2 = ref_point[1]

            # pastikan koordinat valid
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)

            if w > 0 and h > 0:
                roi = img[y:y+h, x:x+w]

                if roi.size != 0:
                    cv2.imwrite(output_crop, roi)

                    crop_config = {"x": x, "y": y, "w": w, "h": h}
                    with open(json_path, "w") as f:
                        json.dump(crop_config, f, indent=4)

                    print(f"✅ Crop disimpan ke {output_crop}")
                    print(f"✅ Koordinat disimpan ke {json_path}")
                else:
                    print("❌ ROI kosong, coba drag ulang.")
            else:
                print("❌ W/H = 0, coba drag ulang.")
        break

    # tekan 'q' untuk keluar tanpa crop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()


# =========================
# STEP 2: Gunakan JSON untuk crop otomatis (gambar lain)
# =========================
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)

    x, y, w, h = cfg["x"], cfg["y"], cfg["w"], cfg["h"]

    # contoh pakai gambar lain (bisa diubah)
    new_img_path = "deteksi_test_4.jpg"
    new_img = cv2.imread(new_img_path)

    if new_img is not None:
        auto_crop = new_img[y:y+h, x:x+w]
        out_name = "auto_crop.jpg"
        cv2.imwrite(out_name, auto_crop)
        print(f"📌 Crop otomatis dari JSON disimpan ke {out_name}")

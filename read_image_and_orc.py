# plate_detect_segment_deskew_ocr_classes.py
import os
import cv2
import math
import re
import numpy as np
from ultralytics import YOLO
import easyocr

# ====== KONFIG GLOBAL ======
MODEL_PATH     = "yolo/best.pt"         # ganti sesuai model
CONF_THRES     = 0.10
OUTPUT_DIR     = "output"          # hasil tahap awal (crop, mask, deskew awal)
OUT_DIR_REFINE = "output_deskew"   # hasil refine (quad/rotate) & OCR
MODE           = "image"           # "image" atau "rtsp"
SOURCE         = "image/test_2.jpg"
SHOW_WINDOWS   = True              # False untuk headless
SAVE_OCR_VIS   = False             # True jika ingin simpan anotasi OCR
FILTER_PLATE   = False             # True untuk filter text A-Z0-9- spasi
# ===========================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_REFINE, exist_ok=True)

# =========================================================
# ================  KELAS PROSES GAMBAR  ==================
# =========================================================
class PlateProcessor:
    """
    Deteksi pelat (YOLO) -> crop -> segmentasi (GrabCut) -> deskew awal (minAreaRect)
    -> refine (coba quad warp / Hough rotate) -> simpan file final (out_path).
    """

    def __init__(self, model_path=MODEL_PATH, conf_thres=CONF_THRES,
                 output_dir=OUTPUT_DIR, out_dir_refine=OUT_DIR_REFINE,
                 show_windows=SHOW_WINDOWS):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.output_dir = output_dir
        self.out_dir_refine = out_dir_refine
        self.show_windows = show_windows
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.out_dir_refine, exist_ok=True)

    # ---------- Util ----------
    @staticmethod
    def _expand_bbox(xyxy, img_shape, pad_ratio=0.08):
        h, w = img_shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad_ratio), int(bh * pad_ratio)
        x1 = max(0, x1 - px); y1 = max(0, y1 - py)
        x2 = min(w - 1, x2 + px); y2 = min(h - 1, y2 + py)
        return x1, y1, x2, y2

    @staticmethod
    def _draw_box(img, xyxy, label, conf=None):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        txt = label if conf is None else f"{label} {conf:.2f}"
        cv2.putText(img, txt, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return img

    @staticmethod
    def _order_pts(pts):
        s = pts.sum(axis=1); d = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype="float32")

    # ---------- Segmentasi + Deskew awal ----------
    def _segment_from_crop(self, crop_bgr, iters=5):
        h, w = crop_bgr.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        rect = (2, 2, w-4, h-4)
        bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
        cv2.grabCut(crop_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        mask_crop = np.where((mask==2)|(mask==0), 0, 1).astype("uint8") * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel, 1)
        mask_crop = cv2.medianBlur(mask_crop, 3)

        b,g,r = cv2.split(crop_bgr)
        cutout_crop = cv2.merge((b,g,r,mask_crop))

        cnts, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask_crop, cutout_crop, crop_bgr, mask_crop, cutout_crop

        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box  = cv2.boxPoints(rect).astype("float32")
        W, H = rect[1]
        if W < H: W, H = H, W
        box = self._order_pts(box)

        dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
        M   = cv2.getPerspectiveTransform(box, dst)
        warped      = cv2.warpPerspective(crop_bgr, M, (int(W), int(H)))
        warped_mask = cv2.warpPerspective(mask_crop, M, (int(W), int(H)))

        wb, wg, wr = cv2.split(warped)
        warped_cutout = cv2.merge((wb, wg, wr, warped_mask))
        return mask_crop, cutout_crop, warped, warped_mask, warped_cutout

    # ---------- Refine (quad/rotate) ----------
    @staticmethod
    def _order_points2(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @classmethod
    def _four_point_warp(cls, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = cls._order_points2(pts.astype("float32"))
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        maxWidth = max(maxWidth, 1)
        maxHeight = max(maxHeight, 1)
        dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    @classmethod
    def _find_plate_quad(cls, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 60, 150)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        img_area = h*w
        best, best_score = None, -1.0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.05*img_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rect = approx.reshape(-1,2).astype("float32")
                (tl, tr, br, bl) = cls._order_points2(rect)
                wA = np.linalg.norm(br - bl)
                wB = np.linalg.norm(tr - tl)
                hA = np.linalg.norm(tr - br)
                hB = np.linalg.norm(tl - bl)
                w_est = max(wA, wB); h_est = max(hA, hB)
                ratio = w_est / (h_est+1e-6)
                ratio_score = -abs(ratio - 3.0)   # ideal ~3:1
                area_score = area / img_area
                score = ratio_score + 2.0*area_score
                if score > best_score:
                    best_score = score
                    best = rect

        return best

    @staticmethod
    def _hough_deskew(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 12)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
        edges = cv2.Canny(thr, 60, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=max(img.shape[:2])//5, maxLineGap=10)
        if lines is None:
            return img, 0.0

        angles = []
        for x1,y1,x2,y2 in lines[:,0]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            ang = math.degrees(math.atan2(dy, dx))
            if -60 < ang < 60:
                angles.append(ang)
        if not angles:
            return img, 0.0

        angle = float(np.median(angles))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    @staticmethod
    def _post_enhance(im):
        sharp = cv2.GaussianBlur(im, (0,0), 1.2)
        sharp = cv2.addWeighted(im, 1.5, sharp, -0.5, 0)
        return sharp

    def _refine_deskew(self, image_bgr: np.ndarray, tag: str):
        if image_bgr is None:
            raise ValueError("_refine_deskew: image_bgr None")

        cv2.imwrite(os.path.join(self.out_dir_refine, f"{tag}_00_input.png"), image_bgr)

        quad = self._find_plate_quad(image_bgr)
        if quad is not None:
            warped = self._four_point_warp(image_bgr, quad)
            method = "warp-quad"
            vis = image_bgr.copy()
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0,255,0), 2)
            cv2.imwrite(os.path.join(self.out_dir_refine, f"{tag}_quad_vis.png"), vis)
        else:
            warped, ang = self._hough_deskew(image_bgr)
            method = f"rotate-hough({ang:.2f}deg)"

        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        warped = self._post_enhance(warped)
        out_path = os.path.join(self.out_dir_refine, f"{tag}_01_deskew.png")
        cv2.imwrite(out_path, warped)
        print(f"[REFINE] Metode: {method}")
        print(f"[REFINE] Simpan: {out_path}")
        return warped, method, out_path

    # ---------- API Publik ----------
    def process_image(self, image_path: str, tag: str = "image"):
        """
        Proses satu gambar. Return dict:
        {
          'out_path': path hasil refine,
          'method': 'warp-quad' atau 'rotate-hough(x)',
          'bbox': (x1,y1,x2,y2), 'conf': float
        }
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        res = self.model.predict(img, conf=self.conf_thres, verbose=False)[0]
        if len(res.boxes) == 0:
            print("[INFO] Tidak ada plat.")
            return {'out_path': None, 'method': None, 'bbox': None, 'conf': None}

        best = int(np.argmax(res.boxes.conf.cpu().numpy()))
        box  = res.boxes[best]
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf.cpu().numpy())

        x1, y1, x2, y2 = self._expand_bbox(xyxy, img.shape)
        crop = img[y1:y2, x1:x2]

        mask_crop, cutout_crop, warped, warped_mask, warped_cutout = self._segment_from_crop(crop)

        annotated = img.copy()
        self._draw_box(annotated, (x1,y1,x2,y2), "License_Plate", conf)
        cv2.imwrite(os.path.join(self.output_dir, "annotated.jpg"), annotated)
        cv2.imwrite(os.path.join(self.output_dir, "crop.jpg"), crop)
        cv2.imwrite(os.path.join(self.output_dir, "crop_mask.png"), mask_crop)
        cv2.imwrite(os.path.join(self.output_dir, "crop_cutout.png"), cutout_crop)
        cv2.imwrite(os.path.join(self.output_dir, "deskew.jpg"), warped)
        cv2.imwrite(os.path.join(self.output_dir, "deskew_mask.png"), warped_mask)
        cv2.imwrite(os.path.join(self.output_dir, "deskew_cutout.png"), warped_cutout)
        print("[DONE] Tahap awal disimpan ke folder 'output'.")

        refined, method, out_path = self._refine_deskew(warped, tag=tag)

        if self.show_windows:
            cv2.imshow("Deskew (awal)", warped)
            cv2.imshow(f"Deskew Refined ({method})", refined)
            cv2.waitKey(1)  # kecil saja biar tidak nge-freeze

        return {'out_path': out_path, 'method': method, 'bbox': (x1,y1,x2,y2), 'conf': conf}

    def run_rtsp(self, source, run_ocr_callable=None):
        """
        Loop RTSP. Tekan 'c' untuk capture dan proses.
        Jika run_ocr_callable diberikan, akan dipanggil dengan arg (out_path, tag).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[ERR] Gagal buka sumber video.")
            return

        print("[INFO] Tekan 'c' untuk capture + proses, 'q' untuk keluar.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if self.show_windows:
                cv2.imshow("Stream", frame)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('c'):
                res = self.model.predict(frame, conf=self.conf_thres, verbose=False)[0]
                if len(res.boxes) == 0:
                    print("[INFO] Tidak ada plat.")
                    continue
                best = int(np.argmax(res.boxes.conf.cpu().numpy()))
                box  = res.boxes[best]
                xyxy = box.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = self._expand_bbox(xyxy, frame.shape)
                crop = frame[y1:y2, x1:x2]

                mask_crop, cutout_crop, warped, warped_mask, warped_cutout = self._segment_from_crop(crop)
                cv2.imwrite(os.path.join(self.output_dir, "deskew.jpg"), warped)
                cv2.imwrite(os.path.join(self.output_dir, "deskew_mask.png"), warped_mask)
                cv2.imwrite(os.path.join(self.output_dir, "deskew_cutout.png"), warped_cutout)

                refined, method, out_path = self._refine_deskew(warped, tag="rtsp")
                print("[SNAPSHOT] Hasil awal di 'output/', refine di 'output_deskew/'")

                if run_ocr_callable is not None and out_path:
                    run_ocr_callable(out_path, "rtsp")

                if self.show_windows:
                    cv2.imshow(f"Deskew Refined ({method})", refined)

            elif k == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# ==================  KELAS OCR SAJA  =====================
# =========================================================
class PlateOCR:
    """
    Kelas khusus OCR:
    - Robust membaca path (handle PNG transparan)
    - read_path -> return results EasyOCR
    - print_results -> langsung print ke terminal
    - annotate_and_save -> opsional simpan visualisasi
    """
    def __init__(self, langs=('en',), gpu=False, filter_plate=FILTER_PLATE,
                 out_dir=OUT_DIR_REFINE, save_vis=SAVE_OCR_VIS):
        self.reader = easyocr.Reader(list(langs), gpu=gpu)
        self.filter_plate = filter_plate
        self.out_dir = out_dir
        self.save_vis = save_vis
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _clean_text(text: str, enable=False) -> str:
        if not enable:
            return text
        return re.sub(r'[^A-Z0-9\- ]', '', text.upper())

    @staticmethod
    def _to_bgr_for_ocr(path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        if im.ndim == 3 and im.shape[2] == 4:
            b, g, r, a = cv2.split(im)
            alpha = a.astype(np.float32) / 255.0
            alpha = np.repeat(alpha[:, :, None], 3, axis=2)
            bgr = cv2.merge((b, g, r)).astype(np.float32)
            white = np.full_like(bgr, 255.0, dtype=np.float32)
            comp = (bgr * alpha) + (white * (1.0 - alpha))
            return comp.astype(np.uint8)
        elif im.ndim == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        else:
            return im  # sudah BGR

    def read_path(self, image_path: str):
        if not os.path.exists(image_path):
            print(f"[OCR] File tidak ditemukan: {image_path}")
            return []
        img = self._to_bgr_for_ocr(image_path)
        if img is None:
            print(f"[OCR] Gagal baca gambar: {image_path}")
            return []
        try:
            results = self.reader.readtext(img)
        except Exception as e:
            print(f"[OCR] Error readtext: {e}")
            return []
        return results

    def print_results(self, results):
        if not results:
            print("[OCR] Tidak ada teks terdeteksi.")
            return
        for i, (_, text, conf) in enumerate(results, 1):
            clean = self._clean_text(text, self.filter_plate)
            if self.filter_plate:
                print(f"{i}. '{clean}' (raw='{text}', conf={conf:.2f})")
            else:
                print(f"{i}. '{text}' (conf={conf:.2f})")

    def annotate_and_save(self, image_path: str, results, tag="ocr"):
        if not self.save_vis or not results:
            return None
        im = cv2.imread(image_path)
        if im is None:
            return None
        for (box, text, conf) in results:
            pts = np.array(box).astype(int)
            cv2.polylines(im, [pts], True, (0,255,0), 2)
            label = self._clean_text(text, self.filter_plate)
            y = max(0, min(pts[0][1]-6, im.shape[0]-1))
            cv2.putText(im, f"{label} {conf:.2f}", (pts[0][0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        outp = os.path.join(self.out_dir, f"{tag}_ocr_vis.png")
        cv2.imwrite(outp, im)
        print(f"[OCR] Visualisasi disimpan: {outp}")
        return outp


# =========================================================
# =======================  MAIN  ==========================
# =========================================================
def main():
    proc = PlateProcessor(model_path=MODEL_PATH,
                          conf_thres=CONF_THRES,
                          output_dir=OUTPUT_DIR,
                          out_dir_refine=OUT_DIR_REFINE,
                          show_windows=SHOW_WINDOWS)

    ocr = PlateOCR(langs=('en',), gpu=False,
                   filter_plate=FILTER_PLATE,
                   out_dir=OUT_DIR_REFINE,
                   save_vis=SAVE_OCR_VIS)

    if MODE == "image":
        info = proc.process_image(SOURCE, tag="image")
        out_path = info.get('out_path')
        if out_path:
            results = ocr.read_path(out_path)
            ocr.print_results(results)
            ocr.annotate_and_save(out_path, results, tag="image")

        if SHOW_WINDOWS:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif MODE == "rtsp":
        src = int(SOURCE) if SOURCE.isdigit() else SOURCE
        def _ocr_callback(out_path, tag):
            print("[OCR] Membaca dari:", out_path)
            res = ocr.read_path(out_path)
            ocr.print_results(res)
            ocr.annotate_and_save(out_path, res, tag=tag)
        proc.run_rtsp(src, run_ocr_callable=_ocr_callback)

if __name__ == "__main__":
    main()

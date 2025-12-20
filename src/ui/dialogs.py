import os
import cv2
import numpy as np
import time
import math
import json
import psutil
from collections import deque
from datetime import datetime, date
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon
from PyQt5.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QDateEdit, QComboBox, QMessageBox,
    QFrame, QGridLayout, QSpinBox, QFileDialog, QApplication
)


# Import helper & config bawaan project Anda
from ..utils import months_between, calc_age_months
from ..config import APP_COPYRIGHT, MODEL_PATH


# --- Import Library AI ---
try:
    from ultralytics import YOLO
    HAS_YOLO = True

except ImportError:
    HAS_YOLO = False
    YOLO = None

try:
    import mediapipe as mp

except ImportError:
    mp = None

# ==========================================
# KONFIGURASI AI & HELPER
# ==========================================
YOLO_IMGSZ = 640
YOLO_MAX_DET = 1
YOLO_CLASSES = [0]
YOLO_MIN_INTERVAL_MS = 250.0
UI_RENDER_FPS = 20

# --- HELPER SKELETON INDICES ---
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
LEFT_ANKLE = 27
LEFT_HEEL = 29

# --- HELPER MATH FUNCTIONS ---
def _to_pixel(lm, w, h):
    return (int(lm.x * w), int(lm.y * h), getattr(lm, "visibility", 1.0))

def _dist(p1, p2):
    (x1, y1), (x2, y2) = p1[:2], p2[:2]
    return math.hypot(x1 - x2, y1 - y2)

def _hip_center(px):
    lh, rh = px.get(LEFT_HIP), px.get(RIGHT_HIP)
    if lh and rh:
        return (int((lh[0] + rh[0]) / 2), int((lh[1] + rh[1]) / 2), min(lh[2], rh[2]))
    return lh or rh

def _extract_px(landmarks, w, h):
    return {i: _to_pixel(lm, w, h) for i, lm in enumerate(landmarks)}

def _draw_point(img, p, label, color=(0, 255, 0)):
    if p is None: return
    cv2.circle(img, p[:2], 5, color, -1)
    cv2.putText(img, label, (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def _draw_link(img, p1, p2, color=(255, 255, 255)):
    if p1 is None or p2 is None: return
    cv2.line(img, p1[:2], p2[:2], color, 2)

def _face_center_x(px):
    cand = []
    for idx in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR):
        p = px.get(idx)
        if p is not None:
            vis = p[2] if len(p) > 2 else 1.0
            cand.append((p[0], max(0.0, float(vis))))

    if not cand:
        return None

    wsum = sum(w for _, w in cand)

    if wsum <= 1e-6:
        return int(sum(x for x, _ in cand) / len(cand))

    return int(sum(x*w for x, w in cand) / wsum)


def _ray_intersect_bbox(origin, direction, bbox):
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None

    x1, y1, x2, y2 = map(float, bbox)
    ts = []

    if abs(dx) > 1e-9:
        t = (x1 - ox)/dx; y = oy + t*dy
        if t >= 0 and y1-1e-6 <= y <= y2+1e-6: ts.append((t, x1, y))
        t = (x2 - ox)/dx; y = oy + t*dy
        if t >= 0 and y1-1e-6 <= y <= y2+1e-6: ts.append((t, x2, y))

    if abs(dy) > 1e-9:
        t = (y1 - oy)/dy; x = ox + t*dx
        if t >= 0 and x1-1e-6 <= x <= x2+1e-6: ts.append((t, x, y1))
        t = (y2 - oy)/dy; x = ox + t*dx
        if t >= 0 and x1-1e-6 <= x <= x2+1e-6: ts.append((t, x, y2))

    if not ts: return None
    t_min, xi, yi = min(ts, key=lambda a: a[0])
    return (int(round(xi)), int(round(yi)))


def _compute_parts(px, head):
    nose = px.get(NOSE)
    hip  = _hip_center(px)
    l_hip   = px.get(LEFT_HIP)
    l_knee  = px.get(LEFT_KNEE)
    l_ankle = px.get(LEFT_ANKLE)
    l_heel  = px.get(LEFT_HEEL)

    if any(v is None for v in [head, nose, hip, l_hip, l_knee, l_ankle, l_heel]):
        return None, None

    h1 = _dist(head, nose)
    h2 = _dist(nose, hip)
    h3 = _dist(l_hip, l_knee)
    h4 = _dist(l_knee, l_ankle)
    h5 = _dist(l_ankle, l_heel)
    H  = h1 + h2 + h3 + h4 + h5

    return (
        {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5, "H": H},
        {"head": head, "nose": nose, "hip": hip,
         "l_hip": l_hip, "l_knee": l_knee, "l_ankle": l_ankle, "l_heel": l_heel}
    )


# --- FUNGSI BANTUAN ICON ---
def set_app_icon(window_obj):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        os.path.join(base_dir, "..", "..", "src", "assets", "sentrisevil_logo.png"),
        os.path.join(base_dir, "..", "assets", "sentrisevil_logo.png"),
        os.path.join(cwd, "src", "assets", "sentrisevil_logo.png"),
        os.path.join(cwd, "assets", "sentrisevil_logo.png"),
        r"src\assets\sentrisevil_logo.png",
        r"SentriSevil\src\assets\sentrisevil_logo.png"
    ]

    for path in candidates:

        if os.path.exists(path):

            window_obj.setWindowIcon(QIcon(path))

            break


# ==========================================

# UI HELPER STYLES

# ==========================================


def style_button(btn, bg="#2962ff", fg="#fff", hover="#1e40ff"):

    btn.setCursor(Qt.PointingHandCursor)

    btn.setFixedHeight(34)

    btn.setStyleSheet(f"""

        QPushButton {{

            background-color: {bg}; color: {fg}; border: none; border-radius: 8px;

            padding: 8px 12px; font-weight: bold; font-family: 'Segoe UI';

        }}

        QPushButton:hover {{ background-color: {hover}; }}

    """)


def style_outline_button(btn):

    btn.setCursor(Qt.PointingHandCursor)

    btn.setFixedHeight(34)

    btn.setStyleSheet("""

        QPushButton {

            background-color: transparent; color: #1e293b; border: 1px solid #cfd8dc;

            border-radius: 8px; padding: 8px 12px; font-weight: bold; font-family: 'Segoe UI';

        }

        QPushButton:hover { background-color: #eef6ff; }

    """)


# ==========================================

# THREAD CLASSES

# ==========================================


class ResourceThread(QThread):

    update_signal = pyqtSignal(float, float, float)

    def __init__(self, parent=None):

        super().__init__(parent)

        self._run = True

    def run(self):

        while self._run:

            try:

                cpu = psutil.cpu_percent(interval=None)

                ram = psutil.virtual_memory().percent

                temp = 0.0

                try:

                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:

                        temp = float(f.read()) / 1000.0

                except:

                    try:

                        temps = psutil.sensors_temperatures()

                        if 'coretemp' in temps: temp = temps['coretemp'][0].current

                        elif 'cpu_thermal' in temps: temp = temps['cpu_thermal'][0].current

                    except: temp = 0.0

                self.update_signal.emit(cpu, ram, temp)

            except: pass

            self.msleep(1500)

    def stop(self): self._run = False; self.wait(500)


class VideoThreadFallback(QThread):
    # Definisi Sinyal
    frame_signal = pyqtSignal(QImage)
    raw_frame_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)
    
    def __init__(self, cam_index=0, target_width=640):
        super().__init__()
        self.idx = cam_index
        self.tw = target_width
        self._run = True
        self._last = None
        self._ema = 0.0

    def run(self):
        if cv2 is None: return
        
        # Gunakan CAP_ANY agar kompatibel (Windows/Linux) dan stabil
        backend = cv2.CAP_ANY 
        cap = cv2.VideoCapture(self.idx, backend)
        
        # [SETTING WIDE] Paksa resolusi HD 1280x720 agar tampilan luas (16:9)
        # Ini mengatasi masalah gambar terlihat "di-zoom" atau terpotong
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Batasi FPS hardware jika memungkinkan untuk mengurangi beban
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.msleep(500) # Tunggu kamera inisialisasi
        
        while self._run:
            if not cap.isOpened():
                self.msleep(1000)
                cap.open(self.idx, backend)
                # Set ulang resolusi saat reconnect
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                continue

            try:
                ret, frame = cap.read()
                
                if not ret: 
                    self.msleep(100)
                    continue
                
                # Hitung FPS
                now = time.time()
                if self._last is not None:
                    inst = 1.0 / max(1e-6, now - self._last)
                    self._ema = inst if self._ema == 0 else 0.1 * inst + 0.9 * self._ema
                    self.fps_signal.emit(float(self._ema))
                self._last = now
                
                # [RESIZE PROPOSIONAL]
                # Frame asli 1280x720 akan di-resize ke lebar target (misal 640)
                # Hasilnya menjadi 640x360 (tetap Wide 16:9), tidak gepeng/terpotong
                h, w = frame.shape[:2]
                if w > self.tw: 
                    s = self.tw / w
                    # Gunakan INTER_AREA untuk hasil pengecilan yang lebih halus
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                
                # Emit data mentah ke AI
                self.raw_frame_signal.emit(frame)
                
                # Convert ke RGB untuk UI
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()
                self.frame_signal.emit(qimg)
                
                # Jeda kecil agar CPU tidak 100%
                self.msleep(15)
                
            except Exception:
                self.msleep(100)

        # Release kamera
        try:
            if cap and cap.isOpened():
                cap.release()
        except: 
            pass

    def stop(self): 
        self._run = False
        self.wait(1000)

# =========================================================================

# SERIAL READER (Logika Persis StuntingMain.py dengan Buffering Kuat)

# =========================================================================

class SerialReaderFallback(QThread):

    sensor_signal = pyqtSignal(float, float) # weight, height

    status_signal = pyqtSignal(str)

   

    # Fungsi statis dari StuntingMain.py untuk parsing berat

    @staticmethod

    def _parse_weight_line(line: str):

        # Format bisa: "Weight[g]: 5200" atau "WT: 5200" atau "Weight: 5.2"

        s = (line or "").strip()

        val = None

       

        if s.startswith("Weight[g]:"):

            parts = s.split(":", 1)

            if len(parts) > 1: val = parts[1].strip()

        elif s.startswith("WT:"):

            parts = s.split(":", 1)

            if len(parts) > 1: val = parts[1].strip()

        elif s.startswith("Weight:"):

            parts = s.split(":", 1)

            if len(parts) > 1: val = parts[1].strip()

           

        if val:

            try:

                return float(val)

            except ValueError:

                return None

        return None


    def __init__(self, port=None, baud=9600):

        super().__init__()

        self.port_name = port

        self.baud = baud

        self._run_flag = True

        self._connected = False

        self._ser = None


    def set_port(self, port):

        self.port_name = port


    def _open(self):

        if not self.port_name:

            self.status_signal.emit("Pilih port dahulu.")

            return False

        try:

            import serial

            self._ser = serial.Serial(self.port_name, self.baud, timeout=1)

            self._connected = True

            self.status_signal.emit(f"Terhubung ke {self.port_name}")

            return True

        except Exception as e:

            self.status_signal.emit(f"Gagal konek serial: {e}")

            return False


    def run(self):

        if self.port_name:

            self._open()


        # Buffer byte untuk menyimpan potongan data yang datang

        buf = b""

       

        while self._run_flag:

            if not self._connected:

                self.msleep(300)

                # Auto reconnect logic jika port sudah diset

                if self.port_name: self._open()

                continue

           

            try:

                if self._ser and self._ser.is_open:

                    if self._ser.in_waiting > 0:

                        data = self._ser.read(self._ser.in_waiting)

                        buf += data

                       

                        # Proses buffer jika ada newline

                        while b"\n" in buf:

                            line_bytes, buf = buf.split(b"\n", 1)

                            line_str = line_bytes.decode("utf-8", errors="ignore").strip()

                           

                            if not line_str: continue

                           

                            # Debug: Print ke console agar user bisa lihat data mentah

                            print(f"[SERIAL RAW] {line_str}")


                            # 1. Coba JSON (Format: {"weight": 5.2, "height": 0})

                            tried_json = False

                            try:

                                obj = json.loads(line_str)

                                weight = float(obj.get("weight", 0.0))

                                height = float(obj.get("height", 0.0))

                                self.sensor_signal.emit(weight, height)

                                tried_json = True

                            except Exception:

                                pass


                            # 2. Coba Raw Text (Format: Weight[g]: 5200 atau WT: 5200)

                            if not tried_json:

                                val_raw = self._parse_weight_line(line_str)

                                if val_raw is not None:

                                    # Logika: Jika string mengandung "g" (gram), bagi 1000.

                                    # Jika tidak yakin, asumsikan input load cell biasanya gram.

                                    # Tapi StuntingMain logicnya: w_kg = float(val) / 1000.0 if "g" in s else float(val)

                                   

                                    # Disini kita cek ulang string aslinya

                                    w_kg = 0.0

                                    if "g" in line_str.lower() and "kg" not in line_str.lower():

                                        w_kg = max(0.0, val_raw / 1000.0)

                                    else:

                                        # Asumsi sudah KG atau format tanpa satuan (biasanya gram dari load cell raw)

                                        # Ubah baris ini jika alat Anda mengirim KG langsung

                                        w_kg = max(0.0, val_raw / 1000.0)

                                       

                                    self.sensor_signal.emit(w_kg, 0.0)

                    else:

                        self.msleep(10)

                               

            except Exception as e:

                self.status_signal.emit(f"Serial error: {e}")

                self._connected = False

                try:

                    if self._ser: self._ser.close()

                except Exception: pass

                self._ser = None

                self.msleep(500)


        try:

            if self._ser: self._ser.close()

        except Exception: pass


    def stop(self):

        self._run_flag = False

        self.wait(1000)


class YoloThread(QThread):

    result_signal = pyqtSignal(QImage); parts_signal = pyqtSignal(dict); status_signal = pyqtSignal(str)

    def __init__(self, model_path, conf=0.5):

        super().__init__(); self.model_path = model_path; self.conf = conf; self._run_flag = True; self._frames = deque(maxlen=1); self._last_infer_time_ms = 0.0; self.min_infer_interval_ms = YOLO_MIN_INTERVAL_MS; self.model = None; self.pose = None; self._last_bbox = None; self._last_pts = None; self._last_dets = []

    def push_frame(self, frame):

        if frame is not None: self._frames.clear(); self._frames.append(frame)

    def _init_models(self):

        if not HAS_YOLO: self.status_signal.emit("Library 'ultralytics' belum diinstall."); return False

        if not os.path.exists(self.model_path): self.status_signal.emit(f"Model tidak ditemukan: {self.model_path}"); return False

        try: self.status_signal.emit("Memuat model YOLO..."); self.model = YOLO(self.model_path); self.status_signal.emit("YOLO Siap + MediaPipe Pose.")

        except Exception as e: self.status_signal.emit(f"Error load YOLO: {e}"); return False

        if mp is not None:

            try: self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

            except Exception as e: self.pose = None; self.status_signal.emit(f"MediaPipe Error: {e}")

        else: self.status_signal.emit("MediaPipe library tidak ditemukan.")

        return True

    def _should_infer(self, now_ms: float) -> bool: return (now_ms - self._last_infer_time_ms) >= self.min_infer_interval_ms

    def _get_head_from_yolo(self, frame_bgr):

        try: results = self.model.predict(frame_bgr, conf=self.conf, device='cpu', verbose=False, imgsz=YOLO_IMGSZ, max_det=YOLO_MAX_DET, classes=YOLO_CLASSES)

        except: return None, None

        if not results: return None, None

        r = results[0]; boxes = getattr(r, "boxes", None)

        if boxes is None or len(boxes) == 0: return None, None

        xyxy = boxes.xyxy.cpu().numpy(); confs = boxes.conf.cpu().numpy(); idx = int(confs.argmax()); x1, y1, x2, y2 = map(int, xyxy[idx]); Hh, Ww = frame_bgr.shape[:2]; x1 = max(0, min(x1, Ww-1)); x2 = max(1, min(x2, Ww)); y1 = max(0, min(y1, Hh-1)); y2 = max(1, min(y2, Hh)); head = (int((x1 + x2)//2), int(y1), 1.0); bbox = (x1, y1, x2, y2); return head, bbox

    def _estimate_head_from_landmarks(self, px, nose, bbox, w, h):

        if bbox is not None and nose is not None:

            hip = _hip_center(px)

            if hip is not None:

                vx = float(nose[0] - hip[0]); vy = float(nose[1] - hip[1]); norm = math.hypot(vx, vy)

                if norm > 1e-6:

                    vx /= norm; vy /= norm; hit = _ray_intersect_bbox((nose[0], nose[1]), (vx, vy), bbox)

                    if hit is not None: x, y = hit; x = max(0, min(int(x), w-1)); y = max(0, min(int(y), h-1)); return (x, y, 1.0)

        if bbox is not None: x_face = _face_center_x(px); x1, y1, x2, _ = bbox; x_face = int((x1 + x2)//2) if x_face is None else x_face; x = max(0, min(int(x_face), w-1)); y = max(0, min(int(y1), h-1)); return (x, y, 1.0)

        return None

    def _infer_once(self, frame_bgr):

        h, w = frame_bgr.shape[:2]; self._last_dets = []; self._last_bbox = None; self._last_pts = None; head_yolo, bbox = self._get_head_from_yolo(frame_bgr)

        if bbox: self._last_bbox = bbox

        pose_landmarks = None

        if self.pose is not None: pr = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)); pose_landmarks = pr.pose_landmarks if pr else None

        if pose_landmarks is not None:

            px = _extract_px(pose_landmarks.landmark, w, h); nose = px.get(NOSE); head = self._estimate_head_from_landmarks(px, nose, bbox, w, h)

            if head is not None and nose is not None:

                parts, pts = _compute_parts(px, head)

                if parts is not None: self._last_pts = pts; self.parts_signal.emit(parts);

                if bbox: self._last_dets = [(bbox[0], bbox[1], bbox[2], bbox[3], True, pose_landmarks)]; return

        if bbox: self._last_dets = [(bbox[0], bbox[1], bbox[2], bbox[3], False, None)]

    def _draw_on_frame(self, frame_bgr):

        if getattr(self, "_last_bbox", None): x1, y1, x2, y2 = self._last_bbox; cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if self._last_dets and self._last_dets[0][4] and self._last_dets[0][5] is not None and mp is not None:

            try: mp.solutions.drawing_utils.draw_landmarks(frame_bgr, self._last_dets[0][5], mp.solutions.pose.POSE_CONNECTIONS, landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

            except: pass

        if getattr(self, "_last_pts", None):

            pts = self._last_pts; _draw_point(frame_bgr, pts.get("head"), "head"); _draw_point(frame_bgr, pts.get("nose"), "nose"); _draw_point(frame_bgr, pts.get("hip"), "hip"); _draw_point(frame_bgr, pts.get("l_hip"), "L-hip"); _draw_point(frame_bgr, pts.get("l_knee"), "L-knee"); _draw_point(frame_bgr, pts.get("l_ankle"), "L-ankle"); _draw_point(frame_bgr, pts.get("l_heel"), "L-heel")

            _draw_link(frame_bgr, pts.get("head"), pts.get("nose"), (0, 255, 255)); _draw_link(frame_bgr, pts.get("nose"), pts.get("hip"), (0, 200, 255)); _draw_link(frame_bgr, pts.get("l_hip"), pts.get("l_knee"), (0, 255, 0)); _draw_link(frame_bgr, pts.get("l_knee"), pts.get("l_ankle"), (255, 0, 0)); _draw_link(frame_bgr, pts.get("l_ankle"), pts.get("l_heel"), (255, 0, 255))

        return frame_bgr

    def run(self):

        if not self._init_models(): return

        while self._run_flag:

            if not self._frames: self.msleep(10); continue

            frame = self._frames[-1];

            if frame is None: self.msleep(1); continue

            now_ms = time.time()*1000.0

            if self._should_infer(now_ms):

                try: self._infer_once(frame); self._last_infer_time_ms = now_ms

                except Exception as e: self.status_signal.emit(f"AI Error: {e}")

            try: out = frame.copy(); out = self._draw_on_frame(out); rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB); qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888).copy(); self.result_signal.emit(qimg)

            except: pass

            self.msleep(30)

        try:

            if self.pose: self.pose.close()

        except: pass

    def stop(self): self._run = False; self.wait(500)


# ==========================================

# DIALOG KALIBRASI BARU

# ==========================================

class ManualRegDialog(QDialog):

    def __init__(self, d, a, b, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Kalibrasi Regresi")

        self.resize(320, 200)

        self.setStyleSheet("font-family: 'Segoe UI'; background: #f8f9fa;")

        set_app_icon(self)


        layout = QVBoxLayout(self)

       

        self.ed_dist = QLineEdit(str(d))

        self.ed_a = QLineEdit(str(a))

        self.ed_b = QLineEdit(str(b))

       

        form = QFormLayout()

        form.addRow("Jarak Kamera (cm):", self.ed_dist)

        form.addRow("Nilai a (Intercept):", self.ed_a)

        form.addRow("Nilai b (Slope):", self.ed_b)

        layout.addLayout(form)

       

        btn_save = QPushButton("Simpan")

        style_button(btn_save, "#27ae60", "#fff", "#2ecc71")

        btn_save.clicked.connect(self.accept)

        layout.addWidget(btn_save)


    def get_values(self):

        try: d = float(self.ed_dist.text())

        except: d = 0.0

        try: a = float(self.ed_a.text())

        except: a = 0.0

        try: b = float(self.ed_b.text())

        except: b = 0.0

        return d, a, b


# ==========================================

# DIALOGS (Baby & Manual)

# ==========================================


class BabyDialog(QtWidgets.QDialog):

    def __init__(self, row=None, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Data Bayi"); self.resize(420, 240)

        self.setStyleSheet("background-color: #f8f9fa; font-family: 'Segoe UI';")

        set_app_icon(self)

       

        layout = QVBoxLayout(self); layout.setContentsMargins(20, 20, 20, 20); layout.setSpacing(15)

        form_widget = QWidget(); form = QFormLayout(form_widget); form.setSpacing(12); form.setLabelAlignment(Qt.AlignLeft)

        self.ed_parent = QLineEdit(); self.ed_baby = QLineEdit()

        self.dob = QDateEdit(calendarPopup=True); self.dob.setDisplayFormat("yyyy-MM-dd"); self.dob.setDate(QDate.currentDate())

        self.sex = QComboBox(); self.sex.addItems(["L","P"])

        input_style = "border: 1px solid #ccc; border-radius: 4px; padding: 5px;"

        for w in [self.ed_parent, self.ed_baby, self.dob, self.sex]: w.setStyleSheet(input_style)

        lbl_style = "color:#444; font-weight:500;"

        form.addRow(QLabel("Nama Orang Tua", styleSheet=lbl_style), self.ed_parent)

        form.addRow(QLabel("Nama Bayi", styleSheet=lbl_style), self.ed_baby)

        form.addRow(QLabel("Tanggal Lahir", styleSheet=lbl_style), self.dob)

        form.addRow(QLabel("Kelamin", styleSheet=lbl_style), self.sex)

        layout.addWidget(form_widget)

        h = QHBoxLayout()

        self.btn_save = QPushButton("Simpan"); style_button(self.btn_save)

        self.btn_cancel = QPushButton("Batal"); style_button(self.btn_cancel, "#e0e0e0", "#333", "#d6d6d6")

        h.addStretch(1); h.addWidget(self.btn_save); h.addWidget(self.btn_cancel); layout.addLayout(h)

        self.btn_save.clicked.connect(self.accept); self.btn_cancel.clicked.connect(self.reject)

        if isinstance(row, dict):

            self.ed_parent.setText(row.get("parent_name",""))

            self.ed_baby.setText(row.get("baby_name",""))

            qd = QtCore.QDate.fromString(row.get("dob",""), "yyyy-MM-dd")

            if qd.isValid(): self.dob.setDate(qd)

            self.sex.setCurrentText(row.get("sex","L"))

    def values(self):

        return {"parent_name": self.ed_parent.text().strip(), "baby_name": self.ed_baby.text().strip(), "dob": self.dob.date().toString("yyyy-MM-dd"), "sex": self.sex.currentText()}


class ManualMeasureDialog(QtWidgets.QDialog):

    def __init__(self, con, baby_row, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Input Manual PB & BB"); self.resize(380, 300)

        self.setStyleSheet("background-color: #f8f9fa; font-family: 'Segoe UI';")

        set_app_icon(self)


        self.con = con; self.baby = baby_row

        layout = QVBoxLayout(self); layout.setContentsMargins(20,20,20,20); layout.setSpacing(15)

        form_widget = QWidget(); form = QFormLayout(form_widget); form.setSpacing(12)

        input_style = "border: 1px solid #ccc; border-radius: 4px; padding: 5px;"

        self.ed_length = QLineEdit(); self.ed_length.setPlaceholderText("cm"); self.ed_length.setStyleSheet(input_style)

        self.ed_weight = QLineEdit(); self.ed_weight.setPlaceholderText("kg"); self.ed_weight.setStyleSheet(input_style)

        self.dt_measure = QDateEdit(calendarPopup=True); self.dt_measure.setDisplayFormat("yyyy-MM-dd"); self.dt_measure.setDate(QDate.currentDate()); self.dt_measure.setStyleSheet(input_style)

        self.sp_age = QSpinBox(); self.sp_age.setRange(0, 60); self.sp_age.setSuffix(" bln"); self.sp_age.setStyleSheet(input_style)

        lbl_style = "color:#444; font-weight:500;"

        form.addRow(QLabel("Panjang Badan (cm)", styleSheet=lbl_style), self.ed_length)

        form.addRow(QLabel("Berat Badan (kg)", styleSheet=lbl_style), self.ed_weight)

        form.addRow(QLabel("Tanggal Ukur", styleSheet=lbl_style), self.dt_measure)

        form.addRow(QLabel("Umur (bulan)", styleSheet=lbl_style), self.sp_age)

        layout.addWidget(form_widget)

        h = QHBoxLayout()

        btn_save = QPushButton("Simpan"); style_button(btn_save)

        btn_cancel = QPushButton("Batal"); style_button(btn_cancel, "#e0e0e0", "#333", "#d6d6d6")

        h.addStretch(1); h.addWidget(btn_save); h.addWidget(btn_cancel); layout.addLayout(h)

        self.dt_measure.dateChanged.connect(self._autofill_age)

        btn_save.clicked.connect(self._save); btn_cancel.clicked.connect(self.reject)

        self._autofill_age()

    def _autofill_age(self):

        try: dob = datetime.strptime(self.baby[3], "%Y-%m-%d").date(); d = self.dt_measure.date().toPyDate(); self.sp_age.setValue(months_between(dob, d))

        except: pass

    def _save(self):

        try: L = float(self.ed_length.text()); W = float(self.ed_weight.text())

        except: QMessageBox.warning(self, "Validasi", "Input angka tidak valid."); return

        if L <= 0 or W <= 0: QMessageBox.warning(self, "Validasi", "Nilai harus > 0."); return

        ts = self.dt_measure.date().toString("yyyy-MM-dd"); cur = self.con.cursor()

        cur.execute("INSERT INTO measure(baby_id,ts,length_cm,weight_kg,age_months) VALUES(?,?,?,?,?)", (self.baby[0], ts, L, W, self.sp_age.value()))

        self.con.commit(); QMessageBox.information(self, "Sukses", "Data tersimpan."); self.accept()


# --- DIALOG UKUR UTAMA ---

class MeasureDialog(QtWidgets.QDialog):

    def __init__(self, con, baby_row, parent=None):

        super().__init__(parent)

        self.setWindowTitle("Ukur Bayi"); self.resize(1200, 740)

        self.setStyleSheet("QDialog{background:#f6fbff; font-family: 'Segoe UI';}")

        set_app_icon(self)


        self.con = con; self.baby = baby_row

        self.video_thread = None; self.serial_thread = None; self.yolo_thread = None

        self.resource_thread = None  # Thread System Monitor

        self.latest_qimg = None; self.height_cm = 0.0; self.H_px = 0.0; self.weight_kg = 0.0

       

        # --- VARIABLE KALIBRASI ---

        self.cal_dist = 100.0 # Default 100 cm

        self.cal_a = 0.0

        self.cal_b = 0.0

       

        # 1. HEADER

        header = QFrame(); header.setFixedHeight(52); header.setStyleSheet("background:#1e88e5;")

        h = QHBoxLayout(header); h.setContentsMargins(12,0,16,0)

        t = QLabel("Ukur Bayi — Kamera & Load Cell"); t.setStyleSheet("color:white; font-weight:700; font-size:16px")

        h.addStretch(1); h.addWidget(t); h.addStretch(1)


        # 2. PANEL VIDEO

        panels = QFrame(); panels.setStyleSheet("QFrame{background:white; border:1px solid #e3f2fd; border-radius:12px}")

        pl = QHBoxLayout(panels); pl.setContentsMargins(12,12,12,12); pl.setSpacing(12)

        self.lbl_video = QLabel("Pratinjau Kamera"); self.lbl_video.setMinimumSize(560,360)

        self.lbl_video.setAlignment(Qt.AlignCenter); self.lbl_video.setStyleSheet("background:#111; color:#aaa; border-radius:12px")

        self.lbl_skel  = QLabel("Output Skeleton"); self.lbl_skel.setMinimumSize(560,360)

        self.lbl_skel.setAlignment(Qt.AlignCenter); self.lbl_skel.setStyleSheet("background:#0b0b0b; color:#aaa; border-radius:12px")

        pl.addWidget(self.lbl_video, 1); pl.addWidget(self.lbl_skel, 1)


        # 3. PANEL CONTROLS

        bottom = QFrame(); bottom.setStyleSheet("QFrame{background:white; border:1px solid #e1f5fe; border-radius:12px}")

        g = QGridLayout(bottom); g.setContentsMargins(12,12,12,12); g.setHorizontalSpacing(12); g.setVerticalSpacing(10)


        # A. Kamera

        card_cam = QFrame(); card_cam.setStyleSheet("QFrame{background:#fff; border:1px solid #e1f5fe; border-radius:12px}")

        cl = QVBoxLayout(card_cam); cl.setContentsMargins(12,12,12,8)

        cl.addWidget(QLabel("Kamera", styleSheet="font-weight:bold; color:#555"), 0)

        row_cam = QHBoxLayout()

        self.cmb_cam = QComboBox(); self.cmb_cam.setStyleSheet("border:1px solid #ccc; padding:4px; border-radius:4px;"); self._scan_cams()

        self.btn_open = QPushButton("Buka"); style_outline_button(self.btn_open)

        self.btn_close = QPushButton("Tutup"); style_outline_button(self.btn_close)

        self.btn_start = QPushButton("Mulai AI"); style_button(self.btn_start, "#42a5f5", "#fff", "#1e88e5")

        self.btn_stop  = QPushButton("Stop AI"); style_outline_button(self.btn_stop)

        row_cam.addWidget(self.cmb_cam); row_cam.addWidget(self.btn_open); row_cam.addWidget(self.btn_close)

        row_cam.addWidget(self.btn_start); row_cam.addWidget(self.btn_stop); cl.addLayout(row_cam)


        # B. Serial

        card_ser = QFrame(); card_ser.setStyleSheet("QFrame{background:#fff; border:1px solid #e1f5fe; border-radius:12px}")

        sl = QVBoxLayout(card_ser); sl.setContentsMargins(12,12,12,8)

        sl.addWidget(QLabel("Load Cell (Serial)", styleSheet="font-weight:bold; color:#555"), 0)

        row_ser = QHBoxLayout()

        self.cmb_port = QComboBox(); self.cmb_port.setStyleSheet("border:1px solid #ccc; padding:4px; border-radius:4px;")

        self.sp_baud = QSpinBox(); self.sp_baud.setRange(1200,115200); self.sp_baud.setValue(9600); self.sp_baud.setStyleSheet("border:1px solid #ccc; padding:4px; border-radius:4px;")

        self.btn_scan = QPushButton("Scan"); style_outline_button(self.btn_scan)

        self.btn_conn = QPushButton("Hubungkan"); style_button(self.btn_conn, "#2ecc71", "#fff", "#27ae60")

        self.btn_disc = QPushButton("Putuskan"); style_outline_button(self.btn_disc)

        row_ser.addWidget(self.cmb_port); row_ser.addWidget(self.sp_baud); row_ser.addWidget(self.btn_scan)

        row_ser.addWidget(self.btn_conn); row_ser.addWidget(self.btn_disc); sl.addLayout(row_ser)

        self.lbl_status = QLabel("Status: -", styleSheet="color:#666; font-size:11px"); sl.addWidget(self.lbl_status)


        # C. Metrik (DIPERBARUI UKURAN)

        card_met = QFrame(); card_met.setStyleSheet("QFrame{background:#fff; border:1px solid #e1f5fe; border-radius:12px}")

        ml = QGridLayout(card_met); ml.setContentsMargins(12,12,12,8)

        def _metric(t, c):

            b = QFrame(); b.setStyleSheet("background:#fafafa; border:1px solid #e0f2f1; border-radius:8px")

            v = QVBoxLayout(b); v.setContentsMargins(8,4,8,4)

            l1 = QLabel(t); l1.setStyleSheet("color:#546e7a; font-weight:700; font-size:14px")

            l2 = QLabel("--"); l2.setStyleSheet(f"color:{c}; font-weight:800; font-size:24px"); v.addWidget(l1); v.addWidget(l2)

            return b, l2

        bx1, self.out_pb = _metric("PB (cm)", "#6d4c41"); bx2, self.out_Hx = _metric("H total (px)", "#37474f"); bx3, self.out_bb = _metric("BB (kg)", "#1b5e20")

        ml.addWidget(bx1,0,0); ml.addWidget(bx2,0,1); ml.addWidget(bx3,0,2)


        # D. Detail Metrik (DIPERBARUI UKURAN)

        card_details = QFrame(); card_details.setStyleSheet("QFrame{background:#fff; border:1px solid #e1f5fe; border-radius:12px}")

        dl = QVBoxLayout(card_details); dl.setContentsMargins(8,8,8,8); dl.setSpacing(4)

        dl.addWidget(QLabel("Rincian Pixel (px)", styleSheet="font-weight:bold; color:#555; font-size:14px"))

       

        detail_grid = QGridLayout(); detail_grid.setSpacing(6)

        def _small_lbl(txt): l = QLabel(txt); l.setStyleSheet("color:#78909c; font-size:10px; font-weight:600"); return l

        def _small_val(): l = QLabel("0.0"); l.setStyleSheet("color:#37474f; font-size:16px; font-weight:800"); return l

       

        self.val_h1 = _small_val(); self.val_h2 = _small_val(); self.val_h3 = _small_val()

        self.val_h4 = _small_val(); self.val_h5 = _small_val(); self.val_H_detail = _small_val()

       

        detail_grid.addWidget(_small_lbl("h1"), 0, 0); detail_grid.addWidget(self.val_h1, 0, 1)

        detail_grid.addWidget(_small_lbl("h2"), 0, 2); detail_grid.addWidget(self.val_h2, 0, 3)

        detail_grid.addWidget(_small_lbl("h3"), 1, 0); detail_grid.addWidget(self.val_h3, 1, 1)

        detail_grid.addWidget(_small_lbl("h4"), 1, 2); detail_grid.addWidget(self.val_h4, 1, 3)

        detail_grid.addWidget(_small_lbl("h5"), 2, 0); detail_grid.addWidget(self.val_h5, 2, 1)

        detail_grid.addWidget(_small_lbl("H Total"), 2, 2); detail_grid.addWidget(self.val_H_detail, 2, 3)

        dl.addLayout(detail_grid)


        # E. Aksi (TOMBOL & MONITORING SYSTEM)

        card_act = QFrame(); card_act.setStyleSheet("QFrame{background:#fff; border:1px solid #e1f5fe; border-radius:12px}")

        al = QVBoxLayout(card_act); al.setContentsMargins(12,12,12,8)

       

        sys_layout = QHBoxLayout(); sys_layout.setSpacing(10)

        def _sys_label(txt, col):

            l = QLabel(txt); l.setStyleSheet(f"color:{col}; font-weight:bold; font-size:11px; border:1px solid #eee; border-radius:4px; padding:2px 4px; background:#fafafa;")

            l.setAlignment(Qt.AlignCenter); return l


        self.lbl_cpu = _sys_label("CPU: 0%", "#e67e22")

        self.lbl_ram = _sys_label("RAM: 0%", "#9b59b6")

        self.lbl_temp = _sys_label("Temp: 0°C", "#e74c3c")

       

        sys_layout.addWidget(self.lbl_cpu); sys_layout.addWidget(self.lbl_ram); sys_layout.addWidget(self.lbl_temp)

        al.addLayout(sys_layout)

       

        self.btn_calib = QPushButton("⚙️ Kalibrasi"); style_outline_button(self.btn_calib)

        self.btn_calib.clicked.connect(self._open_calibration)

       

        self.btn_save = QPushButton("Simpan ke Riwayat"); style_button(self.btn_save, "#2962ff", "#fff", "#1e40ff")

        self.btn_shot = QPushButton("Screenshot"); style_outline_button(self.btn_shot)

        self.btn_exit = QPushButton("Tutup"); style_outline_button(self.btn_exit)

       

        al.addWidget(self.btn_calib)

        al.addWidget(self.btn_save); al.addWidget(self.btn_shot); al.addWidget(self.btn_exit); al.addStretch(1)


        g.addWidget(card_cam, 0, 0, 1, 2)

        g.addWidget(card_ser, 0, 2, 1, 2)

        g.addWidget(card_met, 1, 0, 1, 2)

        g.addWidget(card_details, 1, 2, 1, 1)

        g.addWidget(card_act, 1, 3, 1, 1)


        footer = QLabel(APP_COPYRIGHT); footer.setAlignment(Qt.AlignCenter); footer.setStyleSheet("color:#607d8b; font-size:11px; padding:6px;")

        root = QVBoxLayout(self); root.addWidget(header); root.addWidget(panels); root.addWidget(bottom); root.addWidget(footer)


        self.btn_open.clicked.connect(self._open_cam); self.btn_close.clicked.connect(self._close_cam)

        self.btn_scan.clicked.connect(self._scan_ports); self.btn_conn.clicked.connect(self._connect_serial); self.btn_disc.clicked.connect(self._disconnect_serial)

        self.btn_save.clicked.connect(self._save_to_db); self.btn_shot.clicked.connect(self._screenshot); self.btn_exit.clicked.connect(self.reject)

        self.btn_start.clicked.connect(self._start_yolo); self.btn_stop.clicked.connect(self._stop_yolo)


        # CHANGE: Jangan start serial dulu, biarkan user konek manual

        # self._start_serial_bg();

        self.serial_thread = None # Init kosong

        self._scan_ports(); self._open_cam(); self._start_yolo()

        self._start_resource_monitor()

       

        self.render_timer = QTimer(self); self.render_timer.setInterval(33); self.render_timer.timeout.connect(self._render); self.render_timer.start()


    def _start_resource_monitor(self):

        self.resource_thread = ResourceThread()

        self.resource_thread.update_signal.connect(self._on_resource_update)

        self.resource_thread.start()


    @pyqtSlot(float, float, float)

    def _on_resource_update(self, cpu, ram, temp):

        self.lbl_cpu.setText(f"CPU: {cpu:.1f}%")

        self.lbl_ram.setText(f"RAM: {ram:.1f}%")

        self.lbl_temp.setText(f"Temp: {temp:.1f}°C")

       

        if cpu > 80: self.lbl_cpu.setStyleSheet("color:white; background:#e74c3c; border-radius:4px; padding:2px;")

        else: self.lbl_cpu.setStyleSheet("color:#e67e22; border:1px solid #eee; border-radius:4px; padding:2px; background:#fafafa;")

       

        if temp > 75: self.lbl_temp.setStyleSheet("color:white; background:#c0392b; border-radius:4px; padding:2px;")

        else: self.lbl_temp.setStyleSheet("color:#e74c3c; border:1px solid #eee; border-radius:4px; padding:2px; background:#fafafa;")


    def _open_calibration(self):

        dlg = ManualRegDialog(self.cal_dist, self.cal_a, self.cal_b, self)

        if dlg.exec_() == QDialog.Accepted:

            self.cal_dist, self.cal_a, self.cal_b = dlg.get_values()

            QMessageBox.information(self, "Info", f"Kalibrasi Tersimpan:\nJarak: {self.cal_dist}\na: {self.cal_a}\nb: {self.cal_b}")


    def _scan_cams(self):

        self.cmb_cam.clear()

        if cv2 is None: self.cmb_cam.addItem("No CV2", 0); return

        self.cmb_cam.addItem("Cam 0", 0); self.cmb_cam.addItem("Cam 1", 1)


    def _scan_ports(self):

        self.cmb_port.clear()

        try: import serial.tools.list_ports as lp; ports = [p.device for p in lp.comports()]

        except: ports = []

        if not ports: self.cmb_port.addItem("")

        else:

            for p in ports: self.cmb_port.addItem(p)


    def _open_cam(self):

        idx = self.cmb_cam.currentData() or 0; self._stop_video()

        self.video_thread = VideoThreadFallback(cam_index=int(idx), target_width=640)

        self.video_thread.frame_signal.connect(self._on_qimg)

        if hasattr(self.video_thread, "raw_frame_signal"): self.video_thread.raw_frame_signal.connect(self._on_raw_bgr)

        if hasattr(self.video_thread, "fps_signal"): self.video_thread.fps_signal.connect(lambda f: self._status(f"FPS: {f:.1f}"))

        self.video_thread.start(); self._status(f"Kamera {idx} terbuka")


    def _close_cam(self):

        self._stop_video(); self.lbl_video.setText("Pratinjau Kamera"); self._status("Kamera ditutup")


    def _stop_video(self):

        if self.video_thread:

            try: self.video_thread.stop()

            except: pass

            self.video_thread = None


    def _connect_serial(self):

        port = self.cmb_port.currentText().strip()

        if not port: self._status("Pilih port dulu"); return

       

        # Stop thread lama jika ada

        if self.serial_thread:

            if self.serial_thread.isRunning():

                self.serial_thread.stop()

            self.serial_thread = None


        # Init thread baru

        self.serial_thread = SerialReaderFallback()

        self.serial_thread.set_port(port)

        self.serial_thread.sensor_signal.connect(self._on_sensor)

        self.serial_thread.status_signal.connect(self._status)

        self.serial_thread.start()

       

        self._status(f"Konek ke {port}...")


    def _disconnect_serial(self):

        if self.serial_thread:

            self.serial_thread.stop()

            self.serial_thread.set_port(None)

        self._status("Serial putus")

   

    def _start_yolo(self):
        """Memulai thread AI baru."""
        # 1. Pastikan thread lama sudah dibersihkan
        self._stop_yolo()

        if not HAS_YOLO:
            self._status("Library YOLO tidak ditemukan.")
            return

        self._status("Memuat Model AI...")
        
        # 2. Buat thread baru yang segar
        try:
            self.yolo_thread = YoloThread(MODEL_PATH)
            
            # 3. Sambungkan sinyal kembali
            self.yolo_thread.result_signal.connect(self._on_skel_qimg)
            self.yolo_thread.parts_signal.connect(self._on_parts)
            self.yolo_thread.status_signal.connect(self._status)
            
            # 4. Jalankan
            self.yolo_thread.start()
            self._status("AI Berjalan")
        except Exception as e:
            self.yolo_thread = None
            self._status(f"Gagal Start AI: {e}")


    def _stop_yolo(self):
        """Menghentikan AI dengan aman tanpa membekukan aplikasi."""
        if self.yolo_thread is not None:
            # 1. PUTUSKAN KONEKSI SINYAL DULU
            # Ini mencegah thread yang mau mati mengirim gambar ke UI (penyebab skeleton error)
            try:
                self.yolo_thread.result_signal.disconnect()
                self.yolo_thread.parts_signal.disconnect()
            except: 
                pass
            
            # 2. Perintahkan berhenti (menggunakan timeout bawaan 500ms di class thread)
            # Jangan gunakan self.yolo_thread.wait() manual disini agar tidak hang
            self.yolo_thread.stop()
            
            # 3. Hapus referensi thread
            self.yolo_thread = None
            
            # 4. Bersihkan tampilan layar
            self.lbl_skel.clear()
            self.lbl_skel.setText("AI Berhenti")
            self._status("AI Dihentikan")


    @pyqtSlot(QImage)

    def _on_qimg(self, qimg): self.latest_qimg = qimg

   

    @pyqtSlot(np.ndarray)
    def _on_raw_bgr(self, bgr):
        # Cek sederhana: Hanya kirim frame jika thread AI ada
        # Jangan pakai .isRunning() yang kompleks di sini untuk performa
        if self.yolo_thread:
            self.yolo_thread.push_frame(bgr)


    @pyqtSlot(QImage)

    def _on_skel_qimg(self, qimg):

        if qimg and not qimg.isNull():

            pm = QPixmap.fromImage(qimg).scaled(self.lbl_skel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.lbl_skel.setPixmap(pm)


    @pyqtSlot(dict)

    def _on_parts(self, parts: dict):

        self.H_px = float(parts.get("H", 0.0))

        self.val_h1.setText(f"{parts.get('h1', 0):.1f}"); self.val_h2.setText(f"{parts.get('h2', 0):.1f}")

        self.val_h3.setText(f"{parts.get('h3', 0):.1f}"); self.val_h4.setText(f"{parts.get('h4', 0):.1f}")

        self.val_h5.setText(f"{parts.get('h5', 0):.1f}"); self.val_H_detail.setText(f"{self.H_px:.1f}")


        if self.cal_dist > 0 and self.cal_b != 0:

            ratio = self.H_px / self.cal_dist

            self.height_cm = self.cal_a + (self.cal_b * ratio)

        else:

            self.height_cm = 0.0

       

        self._update_metrics()


    @pyqtSlot(float, float)

    def _on_sensor(self, w, h):

        # Update nilai BB realtime dari Serial

        self.weight_kg = max(0.0, float(w))

        self.out_bb.setText(f"{self.weight_kg:.3f}")


    def _update_metrics(self):

        self.out_Hx.setText(f"{self.H_px:.1f}")

        self.out_pb.setText(f"{self.height_cm:.2f}")

        # Jangan overwrite out_bb disini jika ingin responsif, biarkan _on_sensor mengupdate

        # self.out_bb.setText(f"{self.weight_kg:.3f}")


    def _render(self):

        if getattr(self, "latest_qimg", None) is not None:

            pm = QPixmap.fromImage(self.latest_qimg).scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.lbl_video.setPixmap(pm)


    def _status(self, s): self.lbl_status.setText(f"Status: {s}")


    def _screenshot(self):

        if getattr(self, "latest_qimg", None) is not None:

            path, _ = QFileDialog.getSaveFileName(self, "Simpan", "ukur.jpg", "Images (*.jpg)")

            if path: self.latest_qimg.save(path)


    def _save_to_db(self):

        if not self.baby: return

        try: L = float(self.height_cm); W = float(self.weight_kg)

        except: return

        if L <= 0 or W <= 0: QMessageBox.warning(self, "Validasi", "Data PB/BB harus > 0"); return

        try: dob = datetime.strptime(self.baby[3], "%Y-%m-%d").date(); age = calc_age_months(dob)

        except: age = 0

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); cur = self.con.cursor()

        cur.execute("INSERT INTO measure(baby_id,ts,length_cm,weight_kg,age_months) VALUES(?,?,?,?,?)", (self.baby[0], ts, L, W, age))

        self.con.commit(); QMessageBox.information(self, "Sukses", "Data tersimpan."); self.accept()


    def closeEvent(self, e):

        self._stop_video(); self._stop_yolo()

        if self.serial_thread: self.serial_thread.stop()

        if self.resource_thread: self.resource_thread.stop()

        e.accept()
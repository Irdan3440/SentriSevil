import os
import cv2
import numpy as np
import time
import math
from datetime import datetime, date

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage, QColor, QIcon
from PyQt5.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QDateEdit, QComboBox, QMessageBox, 
    QFrame, QGridLayout, QSpinBox, QFileDialog, QApplication, 
    QGroupBox, QDoubleSpinBox, QGraphicsDropShadowEffect
)

# Import helper & config
from ..utils import months_between, calc_age_months, BabyClassifier
from ..config import APP_COPYRIGHT, ICON_PATH

# --- SAFE IMPORT MEDIAPIPE ---
HAS_MP = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        HAS_MP = True
except ImportError: pass
except Exception: pass

# ==========================================
# 1. HELPER LOGIC
# ==========================================
NOSE = 0
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
LEFT_ANKLE = 27
LEFT_HEEL = 29 

def _dist(p1, p2):
    if p1 is None or p2 is None: return 0.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _midpoint(p1, p2):
    if p1 is None or p2 is None: return None
    return (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

# ==========================================
# 2. HELPER STYLES
# ==========================================
def add_shadow(widget, color="#000000", alpha=30, blur=10):
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setColor(QColor(color))
    eff.setColor(QColor(0, 0, 0, alpha))
    eff.setOffset(0, 2)
    widget.setGraphicsEffect(eff)

def style_card(frame):
    frame.setStyleSheet("""
        QFrame { background-color: white; border: 1px solid #e0e0e0; border-radius: 12px; }
    """)
    add_shadow(frame)

def style_button(btn, bg="#2962ff", fg="white"):
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFixedHeight(36)
    btn.setStyleSheet(f"""
        QPushButton {{ background-color: {bg}; color: {fg}; border: none; border-radius: 8px; font-weight: bold; font-family: 'Segoe UI'; font-size: 13px; }}
        QPushButton:hover {{ filter: brightness(110%); }}
    """)

def style_outline_button(btn):
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFixedHeight(36)
    btn.setStyleSheet("""
        QPushButton { background-color: transparent; color: #37474f; border: 1px solid #cfd8dc; border-radius: 8px; font-weight: 600; font-family: 'Segoe UI'; font-size: 13px; }
        QPushButton:hover { background-color: #f5f5f5; }
    """)

# ==========================================
# 3. THREADS
# ==========================================
class VideoThreadFallback(QThread):
    frame_signal = pyqtSignal(QImage)
    raw_frame_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, cam_index=0, target_width=640):
        super().__init__()
        self.idx = cam_index; self.tw = target_width; self._run = True
        
    def run(self):
        try:
            cap = cv2.VideoCapture(self.idx, cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
            while self._run and cap.isOpened():
                ret, frame = cap.read()
                if not ret: self.msleep(10); continue
                h, w = frame.shape[:2]
                if w > self.tw:
                    s = self.tw/w
                    frame = cv2.resize(frame, (int(w*s), int(h*s)))
                self.raw_frame_signal.emit(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_signal.emit(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888).copy())
            cap.release()
        except: pass
    def stop(self): self._run = False; self.wait(500)

class PoseThread(QThread):
    skeleton_signal = pyqtSignal(QImage)
    parts_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__(); self._run = True; self.latest_frame = None
    def push_frame(self, frame): self.latest_frame = frame
    def run(self):
        if not HAS_MP: return
        try:
            mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
            mp_draw = mp.solutions.drawing_utils
            while self._run:
                if self.latest_frame is None: self.msleep(20); continue
                frame = self.latest_frame.copy()
                h_img, w_img, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_pose.process(rgb)
                data_parts = {"valid": False}
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    lm = results.pose_landmarks.landmark
                    def get_pt(idx):
                        if idx >= len(lm) or lm[idx].visibility < 0.5: return None
                        return (int(lm[idx].x * w_img), int(lm[idx].y * h_img))
                    p_nose = get_pt(NOSE)
                    p_l_hip = get_pt(LEFT_HIP); p_r_hip = get_pt(RIGHT_HIP)
                    p_hip = _midpoint(p_l_hip, p_r_hip) if (p_l_hip and p_r_hip) else (p_l_hip or p_r_hip)
                    p_knee = get_pt(LEFT_KNEE); p_ankle = get_pt(LEFT_ANKLE); p_heel = get_pt(LEFT_HEEL)
                    p_head = None
                    if p_nose and p_hip:
                        dy = p_nose[1] - p_hip[1]
                        p_head = (p_nose[0], int(p_nose[1] + dy * 0.5)) 
                        cv2.circle(frame, p_head, 5, (0,255,255), -1)
                    if all([p_head, p_nose, p_hip, p_knee, p_ankle, p_heel]):
                        h1 = _dist(p_head, p_nose); h2 = _dist(p_nose, p_hip)
                        h3 = _dist(p_hip, p_knee); h4 = _dist(p_knee, p_ankle); h5 = _dist(p_ankle, p_heel)
                        H_total = h1 + h2 + h3 + h4 + h5
                        data_parts = {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5, "H": H_total, "valid": True}
                        pts = [p_head, p_nose, p_hip, p_knee, p_ankle, p_heel]
                        for i in range(len(pts)-1): cv2.line(frame, pts[i], pts[i+1], (0, 255, 255), 2)
                self.parts_signal.emit(data_parts)
                rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.skeleton_signal.emit(QImage(rgb_out.data, w_img, h_img, w_img*3, QImage.Format_RGB888).copy())
                self.latest_frame = None
        except: pass
    def stop(self): self._run = False; self.wait(500)

class SerialReaderFallback(QThread):
    sensor_signal = pyqtSignal(float)
    def __init__(self, port=None): super().__init__(); self.port = port; self._run = True
    def set_port(self, p): self.port = p
    def run(self):
        if not self.port: return
        try:
            import serial
            ser = serial.Serial(self.port, 9600, timeout=1)
            while self._run:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("W:") or line.startswith("Weight"):
                    try: self.sensor_signal.emit(float(line.split(":")[1]))
                    except: pass
                self.msleep(100)
            ser.close()
        except: pass
    def stop(self): self._run = False; self.wait()

# ==========================================
# 4. DIALOGS
# ==========================================

class BabyDialog(QtWidgets.QDialog):
    def __init__(self, row=None, parent=None):
        super().__init__(parent); self.setWindowTitle("Data Bayi"); self.resize(400, 260)
        self.setStyleSheet("font-family: 'Segoe UI';")
        self.setWindowIcon(QIcon(ICON_PATH))
        l = QVBoxLayout(self); f = QFormLayout()
        self.ed_parent = QLineEdit(); self.ed_baby = QLineEdit()
        self.dob = QDateEdit(calendarPopup=True); self.dob.setDate(QDate.currentDate())
        self.sex = QComboBox(); self.sex.addItems(["L","P"])
        f.addRow("Nama Orang Tua", self.ed_parent); f.addRow("Nama Bayi", self.ed_baby)
        f.addRow("Tanggal Lahir", self.dob); f.addRow("Kelamin", self.sex)
        l.addLayout(f)
        h = QHBoxLayout(); b_save = QPushButton("Simpan"); b_save.clicked.connect(self.accept)
        b_canc = QPushButton("Batal"); b_canc.clicked.connect(self.reject)
        h.addStretch(); h.addWidget(b_save); h.addWidget(b_canc); l.addLayout(h)
        if row:
            self.ed_parent.setText(row.get("parent_name",""))
            self.ed_baby.setText(row.get("baby_name",""))
            self.dob.setDate(QDate.fromString(row.get("dob",""), "yyyy-MM-dd"))
            self.sex.setCurrentText(row.get("sex","L"))
    def values(self):
        return {"parent_name":self.ed_parent.text(),"baby_name":self.ed_baby.text(),
                "dob":self.dob.date().toString("yyyy-MM-dd"),"sex":self.sex.currentText()}

class ManualMeasureDialog(QtWidgets.QDialog):
    def __init__(self, con, baby_row, parent=None):
        super().__init__(parent); self.con=con; self.baby=baby_row
        self.setWindowTitle("Input Manual"); self.resize(300, 200)
        self.setWindowIcon(QIcon(ICON_PATH))
        l = QVBoxLayout(self); f = QFormLayout()
        self.ed_l = QLineEdit(); self.ed_w = QLineEdit()
        f.addRow("Panjang (cm)", self.ed_l); f.addRow("Berat (kg)", self.ed_w)
        l.addLayout(f)
        b=QPushButton("Simpan"); b.clicked.connect(self.save); l.addWidget(b)
    def save(self):
        try:
            L, W = float(self.ed_l.text()), float(self.ed_w.text())
            cur=self.con.cursor()
            cur.execute("INSERT INTO measure(baby_id,ts,length_cm,weight_kg,age_months) VALUES(?,?,?,?,?)",
                        (self.baby[0], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), L, W, 0))
            self.con.commit(); self.accept()
        except: pass

# --- MAIN MEASURE DIALOG (INTEGRATED CLASSIFIER) ---
class MeasureDialog(QtWidgets.QDialog):
    def __init__(self, con, baby_row, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ukur Bayi (AI Pose & Regresi)")
        self.resize(1300, 768)
        self.setStyleSheet("font-family:'Segoe UI'; font-size:13px;")
        self.setWindowIcon(QIcon(ICON_PATH))
        
        self.con = con; self.baby = baby_row
        self.H_px = 0.0; self.weight_kg = 0.0; self.calculated_cm = 0.0
        
        # Init Classifier
        self.classifier = BabyClassifier()
        self.baby_sex = baby_row[4] # sex
        self.baby_dob = datetime.strptime(baby_row[3], "%Y-%m-%d").date()
        self.baby_age = calc_age_months(self.baby_dob)
        
        # Main Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10,10,10,10)
        main_layout.setSpacing(10)
        
        # === AREA KIRI: VIDEO ===
        left_panel = QFrame(); left_panel.setFrameShape(QFrame.StyledPanel)
        lp_layout = QVBoxLayout(left_panel); lp_layout.setContentsMargins(0,0,0,0)
        
        video_area = QHBoxLayout()
        self.lbl_cam = QLabel("Kamera Mati"); self.lbl_cam.setAlignment(Qt.AlignCenter)
        self.lbl_cam.setStyleSheet("background:black; color:white; border:1px solid #555;")
        self.lbl_cam.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        self.lbl_ai = QLabel("Menunggu Pose..."); self.lbl_ai.setAlignment(Qt.AlignCenter)
        self.lbl_ai.setStyleSheet("background:#111; color:cyan; border:1px solid #555;")
        self.lbl_ai.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        video_area.addWidget(self.lbl_cam); video_area.addWidget(self.lbl_ai)
        lp_layout.addWidget(QLabel("<b>Real-time Video & Skeleton Analysis</b>")); lp_layout.addLayout(video_area)
        
        # === AREA KANAN: SIDEBAR CONTROL ===
        sidebar = QFrame(); sidebar.setFixedWidth(340)
        sidebar.setStyleSheet("background:#f9f9f9; border-left:1px solid #ccc;")
        sb_layout = QVBoxLayout(sidebar); sb_layout.setSpacing(15); sb_layout.setAlignment(Qt.AlignTop)
        
        # 1. Group Koneksi
        gb_conn = QGroupBox("Koneksi Perangkat")
        gb_conn.setStyleSheet("QGroupBox { font-weight:bold; border:1px solid gray; border-radius:5px; margin-top:10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l_conn = QVBoxLayout(gb_conn)
        self.cmb_cam = QComboBox(); self.cmb_cam.addItems([f"Kamera {i}" for i in range(2)])
        btn_cam = QPushButton("Buka Kamera"); btn_cam.clicked.connect(self.start_camera)
        self.cmb_port = QComboBox(); self.cmb_port.addItems(["COM3", "COM4", "COM5", "/dev/ttyUSB0"])
        btn_ser = QPushButton("Konek Serial"); btn_ser.clicked.connect(self.start_serial)
        
        h_cam = QHBoxLayout(); h_cam.addWidget(self.cmb_cam); h_cam.addWidget(btn_cam)
        h_ser = QHBoxLayout(); h_ser.addWidget(self.cmb_port); h_ser.addWidget(btn_ser)
        l_conn.addLayout(h_cam); l_conn.addLayout(h_ser); sb_layout.addWidget(gb_conn)
        
        # 2. Group Metrik Pixel
        gb_met = QGroupBox("Detail Pose (Pixel)")
        gb_met.setStyleSheet("QGroupBox { font-weight:bold; border:1px solid gray; border-radius:5px; margin-top:10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        grid_met = QGridLayout(gb_met)
        self.v_h1 = QLabel("0"); self.v_h2 = QLabel("0"); self.v_h3 = QLabel("0")
        self.v_h4 = QLabel("0"); self.v_h5 = QLabel("0")
        self.v_H  = QLabel("0"); self.v_H.setStyleSheet("color:blue; font-weight:bold; font-size:14px;")
        def add_row(r, label, val_lbl): grid_met.addWidget(QLabel(label), r, 0); grid_met.addWidget(val_lbl, r, 1)
        add_row(0, "h1 (Head-Nose):", self.v_h1); add_row(1, "h2 (Nose-Hip):", self.v_h2)
        add_row(2, "h3 (Hip-Knee):", self.v_h3); add_row(3, "h4 (Knee-Ankle):", self.v_h4)
        add_row(4, "h5 (Ankle-Heel):", self.v_h5); grid_met.addWidget(QFrame(frameShape=QFrame.HLine), 5, 0, 1, 2)
        add_row(6, "H Total (px):", self.v_H); sb_layout.addWidget(gb_met)
        
        # 3. Group Kalibrasi & Hasil
        gb_res = QGroupBox("Kalibrasi & Hasil")
        gb_res.setStyleSheet("QGroupBox { font-weight:bold; border:1px solid gray; border-radius:5px; margin-top:10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l_res = QVBoxLayout(gb_res)
        f_cal = QFormLayout()
        self.spin_a = QDoubleSpinBox(); self.spin_a.setRange(-9999, 9999); self.spin_a.setValue(-52.33)
        self.spin_b = QDoubleSpinBox(); self.spin_b.setRange(-100, 100); self.spin_b.setValue(0.26)
        f_cal.addRow("Intersep (a):", self.spin_a); f_cal.addRow("Slope (b):", self.spin_b)
        l_res.addLayout(f_cal)
        
        # Hasil Ukur
        l_res.addWidget(QLabel("Panjang Badan (cm):", styleSheet="color:#555"))
        self.lbl_pb = QLabel("0.0 cm"); self.lbl_pb.setStyleSheet("font-size:24px; font-weight:bold; color:red; border:1px solid #ddd; background:white; qproperty-alignment:AlignCenter;")
        l_res.addWidget(self.lbl_pb)
        
        l_res.addWidget(QLabel("Berat Badan (kg):", styleSheet="color:#555"))
        self.lbl_bb = QLabel("0.0 kg"); self.lbl_bb.setStyleSheet("font-size:24px; font-weight:bold; color:green; border:1px solid #ddd; background:white; qproperty-alignment:AlignCenter;")
        l_res.addWidget(self.lbl_bb)
        
        # STATUS KLASIFIKASI (NEW)
        l_res.addWidget(QLabel("Klasifikasi Gizi:", styleSheet="color:#555; margin-top:5px;"))
        self.lbl_status_stunting = QLabel("-")
        self.lbl_status_stunting.setStyleSheet("background:#e3f2fd; color:#0d47a1; font-weight:bold; padding:4px; border-radius:4px;")
        self.lbl_status_wasting = QLabel("-")
        self.lbl_status_wasting.setStyleSheet("background:#fff3e0; color:#e65100; font-weight:bold; padding:4px; border-radius:4px;")
        l_res.addWidget(self.lbl_status_stunting)
        l_res.addWidget(self.lbl_status_wasting)
        
        sb_layout.addWidget(gb_res)
        
        # 4. Buttons
        btn_save = QPushButton("SIMPAN DATA"); btn_save.setFixedHeight(40); btn_save.setStyleSheet("background:#2962ff; color:white; font-weight:bold; border-radius:5px;")
        btn_save.clicked.connect(self.save_data)
        btn_close = QPushButton("TUTUP"); btn_close.setFixedHeight(30); btn_close.clicked.connect(self.reject)
        sb_layout.addStretch(); sb_layout.addWidget(btn_save); sb_layout.addWidget(btn_close)
        
        main_layout.addWidget(left_panel, 70); main_layout.addWidget(sidebar, 30)
        self.th_cam = None; self.th_pose = None; self.th_serial = None

    # --- LOGIC ---
    def start_camera(self):
        idx = self.cmb_cam.currentIndex()
        self.th_cam = VideoThreadFallback(idx)
        self.th_cam.frame_signal.connect(lambda q: self.lbl_cam.setPixmap(QPixmap.fromImage(q).scaled(self.lbl_cam.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        self.th_cam.raw_frame_signal.connect(self.process_ai)
        self.th_cam.start()
        
        if HAS_MP:
            self.th_pose = PoseThread()
            self.th_pose.skeleton_signal.connect(lambda q: self.lbl_ai.setPixmap(QPixmap.fromImage(q).scaled(self.lbl_ai.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            self.th_pose.parts_signal.connect(self.update_metrics)
            self.th_pose.start()
        else: self.lbl_ai.setText("MediaPipe Error")

    def start_serial(self):
        self.th_serial = SerialReaderFallback(self.cmb_port.currentText())
        self.th_serial.sensor_signal.connect(self.update_weight)
        self.th_serial.start()

    @pyqtSlot(np.ndarray)
    def process_ai(self, frame):
        if self.th_pose: self.th_pose.push_frame(frame)

    @pyqtSlot(dict)
    def update_metrics(self, data):
        if not data.get("valid", False):
            self.v_H.setText("-")
            return
        
        self.v_h1.setText(f"{data['h1']:.1f}"); self.v_h2.setText(f"{data['h2']:.1f}")
        self.v_h3.setText(f"{data['h3']:.1f}"); self.v_h4.setText(f"{data['h4']:.1f}")
        self.v_h5.setText(f"{data['h5']:.1f}")
        
        H_px = data['H']; self.H_px = H_px; self.v_H.setText(f"{H_px:.1f}")
        
        # Regresi & Klasifikasi
        a = self.spin_a.value(); b = self.spin_b.value()
        cm = a + (b * H_px)
        self.calculated_cm = max(0.0, cm)
        self.lbl_pb.setText(f"{self.calculated_cm:.1f} cm")
        self.run_classification()

    @pyqtSlot(float)
    def update_weight(self, val):
        self.weight_kg = val
        self.lbl_bb.setText(f"{self.weight_kg:.2f} kg")
        self.run_classification()

    def run_classification(self):
        # Klasifikasi Stunting (PB/U)
        st_stunting = self.classifier.classify_stunting(self.baby_sex, self.baby_age, self.calculated_cm)
        self.lbl_status_stunting.setText(f"PB/U: {st_stunting}")
        
        # Klasifikasi Wasting (BB/PB)
        st_wasting = self.classifier.classify_wasting(self.baby_sex, self.calculated_cm, self.weight_kg)
        self.lbl_status_wasting.setText(f"BB/PB: {st_wasting}")

    def save_data(self):
        if self.calculated_cm <= 0:
            QMessageBox.warning(self, "Validasi", "Panjang badan belum terdeteksi (0 cm).")
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur = self.con.cursor()
            cur.execute("INSERT INTO measure(baby_id, ts, length_cm, weight_kg, age_months) VALUES(?,?,?,?,?)",
                        (self.baby[0], ts, self.calculated_cm, self.weight_kg, self.baby_age))
            self.con.commit()
            QMessageBox.information(self, "Tersimpan", f"Data tersimpan!\nStatus: {self.lbl_status_stunting.text()}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def closeEvent(self, e):
        if self.th_cam: self.th_cam.stop()
        if self.th_pose: self.th_pose.stop()
        if self.th_serial: self.th_serial.stop()
        e.accept()
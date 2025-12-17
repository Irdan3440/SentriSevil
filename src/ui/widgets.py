import os
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QFrame, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from datetime import datetime, date

# Import logika perhitungan dari utils
from ..utils import load_wfl_curves, load_hfa_curves, months_between

# --- Canvas Grafik BB/PB (WFL) ---
class GrowthCanvasWFL(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(5.0, 3.0), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

    def plot(self, sex: str, points):
        self.ax.clear()
        self.ax.set_title("Grafik BB/PB (WFL)")
        self.ax.set_xlabel("Panjang (cm)")
        self.ax.set_ylabel("Berat (kg)")
        self.ax.grid(True, alpha=.25)

        curves = load_wfl_curves(sex)
        all_x, all_y = [], []

        # Plot Kurva WHO
        if curves:
            xs = curves["x"]
            self.ax.plot(xs, curves["med"], lw=2, label="Median")
            self.ax.plot(xs, curves["-2sd"], "--", label="-2 SD")
            self.ax.plot(xs, curves["+2sd"], "--", label="+2 SD")
            self.ax.legend(loc="lower right")
            all_x += xs
            all_y += curves["med"] + curves["-2sd"] + curves["+2sd"]

        # Plot Data Bayi
        if points: 
            # points = [(len, weight, ts), ...]
            # Sort by length agar garis nyambung urut
            pts = sorted(points, key=lambda p: float(p[0]))
            X = [float(p[0]) for p in pts]
            Y = [float(p[1]) for p in pts]
            self.ax.plot(X, Y, "o-", lw=2, label="Bayi")
            all_x += X
            all_y += Y
        
        # Auto-scale sumbu agar tidak terlalu zoom in/out
        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            pad_x = (xmax - xmin) * 0.05 if xmax > xmin else 2.0
            pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 0.5
            self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
            self.ax.set_ylim(max(0, ymin - pad_y), ymax + pad_y)

        self.draw()

# --- Canvas Grafik PB/U (HFA) ---
class GrowthCanvasHFA(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(5.0, 3.0), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

    def plot(self, sex: str, dob_str: str, points):
        self.ax.clear()
        self.ax.set_title("Grafik PB/U (HFA)")
        self.ax.set_xlabel("Umur (bulan)")
        self.ax.set_ylabel("Panjang (cm)")
        self.ax.grid(True, alpha=.25)

        curves = load_hfa_curves(sex)
        all_x, all_y = [], []

        if curves:
            xs = curves["m"]
            self.ax.plot(xs, curves["med"], lw=2, label="Median")
            self.ax.plot(xs, curves["-2sd"], "--", label="-2 SD")
            self.ax.plot(xs, curves["+2sd"], "--", label="+2 SD")
            all_x += xs
            all_y += curves["med"] + curves["-2sd"] + curves["+2sd"]

        if points:
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
            except:
                dob = date.today()
            
            # Hitung umur (bulan) untuk setiap titik ukur
            pts_sorted = []
            for l_val, w_val, ts in points:
                try: d1 = datetime.fromisoformat(ts).date()
                except: d1 = date.today()
                pts_sorted.append((months_between(dob, d1), float(l_val)))
            
            # Urutkan berdasarkan umur
            pts_sorted.sort(key=lambda x: x[0])
            
            X = [p[0] for p in pts_sorted]
            Y = [p[1] for p in pts_sorted] # length
            self.ax.plot(X, Y, "o-", lw=2, label="Bayi")
            all_x += X
            all_y += Y

        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            pad_x = max(1.0, (xmax-xmin)*0.05)
            pad_y = max(0.5, (ymax-ymin)*0.05)
            self.ax.set_xlim(max(0, xmin - pad_x), xmax + pad_x)
            self.ax.set_ylim(max(0, ymin - pad_y), ymax + pad_y)

        self.draw()

# --- Kartu Statistik (StatCard) ---
class StatCard(QFrame):
    clicked = pyqtSignal()

    # Perhatikan: parameter __init__ sudah diperlengkap di sini
    def __init__(self, title: str, value: int = 0, color: str = "#1E90FF",
                 icon_path: str = None, fallback_emoji: str = None, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"background-color:{self._color}; border-radius:14px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        # Header: Icon + Title
        hdr = QHBoxLayout(); hdr.setSpacing(8)
        ico = QLabel()
        
        # Logika: Jika ada file icon pakai gambar, jika tidak pakai emoji
        if icon_path and os.path.exists(icon_path):
            pm = QPixmap(icon_path).scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            ico.setPixmap(pm); ico.setFixedSize(28, 28)
        else:
            # Fallback emoji
            ico.setText(fallback_emoji or "•")
            ico.setStyleSheet("color:white; font-size:18px; font-weight:700; background:transparent;")
        
        hdr.addWidget(ico)

        self.lblTitle = QLabel(title)
        self.lblTitle.setStyleSheet("font-size:13px; font-weight:600; color:white; background:transparent;")
        hdr.addWidget(self.lblTitle); hdr.addStretch()
        lay.addLayout(hdr)

        # Value (Angka Besar)
        self.lblValue = QLabel(str(value))
        self.lblValue.setStyleSheet("font-size:28px; font-weight:800; color:white; background:transparent;")
        lay.addWidget(self.lblValue)

        lay.addStretch(1)
        
        # Tombol Detail
        self.btn = QPushButton("Lihat Detail")
        self.btn.setStyleSheet(
            "QPushButton{background:white;color:#111;border:0;border-radius:8px;padding:6px 12px}"
            "QPushButton:hover{background:#eee}"
        )
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.clicked.connect(self.clicked.emit)
        lay.addWidget(self.btn, alignment=Qt.AlignRight)

    def setValue(self, v: int):
        self.lblValue.setText(str(v))
    
    def setToolTip(self, text):
        super().setToolTip(text)
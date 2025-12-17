import cv2
import time
import os
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

# --- Video Thread ---
class VideoThreadFallback(QThread):
    frame_signal = pyqtSignal(QImage)
    raw_frame_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)
    
    def __init__(self, cam_index=0, target_width=640):
        super().__init__()
        self.idx = cam_index
        self.tw = target_width
        self._run = True
        self.cap = None

    def run(self):
        if cv2 is None: return
        self.cap = cv2.VideoCapture(self.idx, cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
        while self._run and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok: 
                self.msleep(10); continue
            
            # Resize logic here if needed
            h, w = frame.shape[:2]
            if w > self.tw:
                s = self.tw/w
                frame = cv2.resize(frame, (int(w*s), int(h*s)))

            try: self.raw_frame_signal.emit(frame)
            except: pass
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888).copy()
            self.frame_signal.emit(q)
            
        if self.cap: self.cap.release()

    def stop(self):
        self._run = False
        self.wait(500)

# --- Serial Thread ---
class SerialReaderFallback(QThread):
    sensor_signal = pyqtSignal(float, float) # weight, height
    status_signal = pyqtSignal(str)
    
    def __init__(self, port=None, baud=9600):
        super().__init__()
        self.port = port
        self.baud = baud
        self._run = True
        self._ser = None

    def set_port(self, p): self.port = p

    def run(self):
        try:
            import serial
        except ImportError:
            self.status_signal.emit("Modul serial tidak ditemukan")
            return

        if self.port:
            try:
                self._ser = serial.Serial(self.port, self.baud, timeout=1)
                self.status_signal.emit(f"Terhubung {self.port}")
            except Exception as e:
                self.status_signal.emit(f"Error: {e}")
                return

        while self._run:
            if self._ser and self._ser.is_open:
                try:
                    line = self._ser.readline().decode('utf-8').strip()
                    if line:
                        # Parsing dummy logic (sesuaikan dengan format data alat)
                        # Contoh format: "W:5.2"
                        if line.startswith("W:"):
                            w = float(line.split(":")[1])
                            self.sensor_signal.emit(w, 0.0)
                except: pass
            self.msleep(100)
            
        if self._ser: self._ser.close()

    def stop(self):
        self._run = False
        self.wait()
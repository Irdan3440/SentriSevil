import os
import sys
import platform
import datetime
from datetime import date

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, 
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QToolButton, QMessageBox, QFrame,
                             QAbstractItemView, QInputDialog, QFileDialog,
                             QGroupBox, QFormLayout, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QSize
from PyQt5.QtGui import QFont, QIcon, QTextDocument, QPixmap, QDesktopServices, QPainter, QBrush, QColor, QPen
from PyQt5.QtPrintSupport import QPrinter

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

# Import widget dan dialog
from .widgets import StatCard, GrowthCanvasWFL, GrowthCanvasHFA
from .dialogs import BabyDialog, MeasureDialog, ManualMeasureDialog
from ..utils import BabyClassifier, is_wasting_PB_BB, is_stunting_HFA, calc_age_months, months_between
from ..config import APP_COPYRIGHT, APP_NAME, APP_VERSION, DEV_PROFILE, SUPV_PROFILE, LOGO_PATH, DB_FILE

# --- Helper Style ---
def style_action_button(btn, text, color="#333"):
    btn.setText(text)
    btn.setStyleSheet(f"""
        QToolButton {{ background: transparent; border: 1px solid #ddd; border-radius: 4px; color: {color}; padding: 2px; }}
        QToolButton:hover {{ background: #eee; border-color: #bbb; }}
    """)
    btn.setCursor(Qt.PointingHandCursor)

# ================= DASHBOARD PAGE =================
class DashboardPage(QWidget):
    def __init__(self, con):
        super().__init__()
        self.con = con
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20); layout.setSpacing(20)
        
        cards = QHBoxLayout(); cards.setSpacing(15)
        self.c_bayi = StatCard("Total Bayi", 0, "#2979ff", fallback_emoji="👶")
        self.c_stunt = StatCard("Total Stunting", 0, "#1b5e20", fallback_emoji="📉")
        self.c_wasting = StatCard("Total Wasting (PB/BB)", 0, "#0097a7", fallback_emoji="⚖️")
        cards.addWidget(self.c_bayi); cards.addWidget(self.c_stunt); cards.addWidget(self.c_wasting)
        layout.addLayout(cards)
        
        graph_frame = QFrame()
        graph_frame.setStyleSheet("background: white; border-radius: 10px; border: 1px solid #e0e0e0;")
        v_graph = QVBoxLayout(graph_frame)
        lbl_graph = QLabel("Data Bayi per Bulan"); lbl_graph.setAlignment(Qt.AlignCenter)
        lbl_graph.setStyleSheet("font-weight: bold; font-size: 14px; color: #333; margin-top: 10px;")
        v_graph.addWidget(lbl_graph)
        self.canvas = FigureCanvas(Figure(figsize=(5,3), tight_layout=True))
        self.ax = self.canvas.figure.add_subplot(111)
        v_graph.addWidget(self.canvas)
        layout.addWidget(graph_frame); layout.addStretch() 
        self.refresh()
        
    def refresh(self):
        cur = self.con.cursor()
        try:
            cur.execute("SELECT id, dob, sex FROM baby")
            babies = cur.fetchall()
            total_bayi = len(babies)
        except: total_bayi = 0; babies = []
        self.c_bayi.setValue(total_bayi)

        total_stunting = 0; total_wasting = 0
        for (bid, dob_str, sex) in babies:
            cur.execute("SELECT length_cm, weight_kg FROM measure WHERE baby_id=? ORDER BY ts DESC LIMIT 1", (bid,))
            last = cur.fetchone()
            if last:
                length_cm, weight_kg = last
                try:
                    d_dob = datetime.datetime.strptime(dob_str, "%Y-%m-%d").date()
                    age_m = calc_age_months(d_dob)
                except: age_m = 0
                if is_stunting_HFA(age_m, length_cm, sex): total_stunting += 1
                if is_wasting_PB_BB(length_cm, weight_kg, sex): total_wasting += 1

        self.c_stunt.setValue(total_stunting); self.c_wasting.setValue(total_wasting)
        counts = [0]*12; curr_year = datetime.datetime.now().year
        for (_, dob_str, _) in babies:
            try:
                d = datetime.datetime.strptime(dob_str, "%Y-%m-%d")
                if d.year == curr_year: counts[d.month-1] += 1
            except: pass
        self.ax.clear(); self.ax.set_title(f"Kelahiran Bayi Tahun {curr_year}")
        self.ax.bar(["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"], counts)
        self.canvas.draw()

# ================= DATA PAGE =================
class DataPage(QWidget):
    open_profile = pyqtSignal(int)
    
    def __init__(self, con):
        super().__init__()
        self.con = con
        layout = QVBoxLayout(self); layout.setContentsMargins(20, 20, 20, 20); layout.setSpacing(15)
        
        top_bar = QHBoxLayout()
        lbl_title = QLabel("Data Bayi"); lbl_title.setFont(QFont("Segoe UI", 16, QFont.Bold)); lbl_title.setStyleSheet("color: #0f2743;")
        top_bar.addWidget(lbl_title); top_bar.addStretch()
        self.btn_print = QPushButton("Cetak"); self.btn_search = QPushButton("Cari Bayi"); self.btn_add = QPushButton("Tambah Data")
        
        btn_style = "QPushButton { background-color: #2979ff; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #2962ff; }"
        for btn in (self.btn_print, self.btn_search, self.btn_add):
            btn.setCursor(Qt.PointingHandCursor); btn.setStyleSheet(btn_style); top_bar.addWidget(btn)
        layout.addLayout(top_bar)
        
        self.tbl = QTableWidget(0, 6); self.tbl.setHorizontalHeaderLabels(["No", "Nama Bayi", "Orangtua", "Lahir", "JK", "Aksi"])
        self.tbl.setStyleSheet("""
            QTableWidget { background-color: white; border: 1px solid #e0e0e0; gridline-color: #f0f0f0; font-size: 13px; }
            QHeaderView::section { background-color: #f5f5f5; color: #333; padding: 8px; border: none; font-weight: bold; border-bottom: 2px solid #ddd; }
            QTableWidget::item { padding: 5px; }
        """)
        self.tbl.verticalHeader().setVisible(False); self.tbl.setAlternatingRowColors(True); self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows); self.tbl.setShowGrid(False)
        header = self.tbl.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents); header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch); header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents); header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        layout.addWidget(self.tbl)
        
        self.btn_add.clicked.connect(self.add_data); self.btn_search.clicked.connect(self.search_data); self.btn_print.clicked.connect(self.print_pdf)
        self.reload()

    def reload(self, kw=None):
        self.tbl.setRowCount(0); cur = self.con.cursor()
        base = "SELECT id, baby_name, parent_name, dob, sex FROM baby"
        if kw: base += f" WHERE baby_name LIKE '%{kw}%' OR parent_name LIKE '%{kw}%'"
        cur.execute(base + " ORDER BY id DESC")
        for i, (bid, name, parent, dob, sex) in enumerate(cur.fetchall(), start=1):
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            item_no = QTableWidgetItem(str(i)); item_no.setTextAlignment(Qt.AlignCenter); self.tbl.setItem(r, 0, item_no)
            self.tbl.setItem(r, 1, QTableWidgetItem(name)); self.tbl.setItem(r, 2, QTableWidgetItem(parent))
            self.tbl.setItem(r, 3, QTableWidgetItem(dob)); self.tbl.setItem(r, 4, QTableWidgetItem("Laki-Laki" if sex == "L" else "Perempuan"))
            w_action = QWidget(); l_action = QHBoxLayout(w_action); l_action.setContentsMargins(4, 2, 4, 2); l_action.setSpacing(4)
            btn_view = QToolButton(); style_action_button(btn_view, "👁", "#2979ff")
            btn_edit = QToolButton(); style_action_button(btn_edit, "✏️", "#fbc02d")
            btn_del  = QToolButton(); style_action_button(btn_del,  "🗑", "#d32f2f")
            btn_view.clicked.connect(lambda _, x=bid: self.open_profile.emit(x))
            btn_edit.clicked.connect(lambda _, x=bid: self.edit_data(x))
            btn_del.clicked.connect(lambda _, x=bid: self.delete_data(x))
            l_action.addWidget(btn_view); l_action.addWidget(btn_edit); l_action.addWidget(btn_del); self.tbl.setCellWidget(r, 5, w_action)

    def add_data(self):
        dlg = BabyDialog(parent=self)
        if dlg.exec_():
            v = dlg.values(); cur = self.con.cursor()
            cur.execute("INSERT INTO baby(baby_name, parent_name, dob, sex) VALUES(?,?,?,?)", (v['baby_name'], v['parent_name'], v['dob'], v['sex']))
            self.con.commit(); self.reload()
            
    def edit_data(self, bid):
        cur = self.con.cursor(); cur.execute("SELECT baby_name,parent_name,dob,sex FROM baby WHERE id=?", (bid,))
        r = cur.fetchone()
        if not r: return
        dlg = BabyDialog({"baby_name": r[0], "parent_name": r[1], "dob": r[2], "sex": r[3]}, self)
        if dlg.exec_():
            v = dlg.values()
            cur.execute("UPDATE baby SET baby_name=?, parent_name=?, dob=?, sex=? WHERE id=?", (v['baby_name'], v['parent_name'], v['dob'], v['sex'], bid))
            self.con.commit(); self.reload()

    def delete_data(self, bid):
        if QMessageBox.question(self, "Hapus", "Hapus data bayi ini beserta riwayatnya?") == QMessageBox.Yes:
            cur = self.con.cursor(); cur.execute("DELETE FROM measure WHERE baby_id=?", (bid,)); cur.execute("DELETE FROM baby WHERE id=?", (bid,)); self.con.commit(); self.reload()

    def search_data(self):
        text, ok = QInputDialog.getText(self, "Cari Bayi", "Masukkan Nama:")
        if ok: self.reload(text)

    def print_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Simpan PDF", "data_bayi.pdf", "PDF (*.pdf)")
        if not path: return
        printer = QPrinter(QPrinter.HighResolution); printer.setOutputFormat(QPrinter.PdfFormat); printer.setOutputFileName(path)
        doc = QTextDocument(); html = f"<h2 align='center'>Data Bayi Puskesmas</h2><table border='1' cellspacing='0' cellpadding='4' width='100%'><tr bgcolor='#f0f0f0'><th>No</th><th>Nama</th><th>Ortu</th><th>Lahir</th><th>JK</th></tr>"
        cur = self.con.cursor(); cur.execute("SELECT baby_name, parent_name, dob, sex FROM baby ORDER BY id DESC")
        for i, (nm, ortu, dob, sex) in enumerate(cur.fetchall(), 1):
            jk = "L" if sex=="L" else "P"; html += f"<tr><td>{i}</td><td>{nm}</td><td>{ortu}</td><td>{dob}</td><td>{jk}</td></tr>"
        html += "</table>"; doc.setHtml(html); doc.print_(printer); QMessageBox.information(self, "Sukses", "Tersimpan")

# ================= PROFILE PAGE =================
class ProfilePage(QWidget):
    def __init__(self, con):
        super().__init__()
        self.con = con
        self.current_baby = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        title = QLabel("Profil Bayi")
        title.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        root.addWidget(title)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        self.gbIdent = QGroupBox("Identitas")
        self.gbIdent.setStyleSheet("QGroupBox{font-weight:600; background:white; border:1px solid #ddd; border-radius:8px; margin-top:8px;} QGroupBox::title{subcontrol-origin: margin; left:10px; padding: 0 3px;}")
        
        form = QFormLayout(self.gbIdent)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setContentsMargins(12, 12, 12, 12)
        
        self.f_name   = QLabel("-"); self.f_sex    = QLabel("-")
        self.f_age    = QLabel("-"); self.f_parent = QLabel("-")
        self.f_dob    = QLabel("-")
        
        for w in (self.f_name, self.f_sex, self.f_age, self.f_parent, self.f_dob):
            w.setStyleSheet("color:#334155; font-weight:500;")
        
        form.addRow("Nama",           self.f_name)
        form.addRow("Jenis Kelamin",  self.f_sex)
        form.addRow("Umur",           self.f_age)
        form.addRow("Orangtua",       self.f_parent)
        form.addRow("Tanggal Lahir",  self.f_dob)
        
        self.gbHist = QGroupBox("Riwayat")
        self.gbHist.setStyleSheet("QGroupBox{font-weight:600; background:white; border:1px solid #ddd; border-radius:8px; margin-top:8px;} QGroupBox::title{subcontrol-origin: margin; left:10px; padding: 0 3px;}")
        vHist = QVBoxLayout(self.gbHist)
        vHist.setContentsMargins(0, 10, 0, 0)
        
        self.tbl_hist = QTableWidget(0, 5)
        self.tbl_hist.setHorizontalHeaderLabels(["No", "Berat (kg)", "Panjang (cm)", "Status", "Tanggal"])
        self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_hist.verticalHeader().setVisible(False)
        self.tbl_hist.setFrameShape(QFrame.NoFrame)
        self.tbl_hist.setStyleSheet("""
            QTableWidget { background: transparent; selection-background-color: #e3f2fd; selection-color: black; }
            QHeaderView::section { background: #f8fafc; border: none; border-bottom: 1px solid #e2e8f0; font-weight: bold; padding: 4px; }
        """)
        vHist.addWidget(self.tbl_hist)

        top_layout.addWidget(self.gbIdent, 1)
        top_layout.addWidget(self.gbHist, 2)
        root.addLayout(top_layout)

        graphs = QHBoxLayout()
        graphs.setSpacing(8)

        style_graph = "QGroupBox{font-weight:600; background:white; border:1px solid #ddd; border-radius:8px; margin-top:8px;} QGroupBox::title{subcontrol-origin: margin; left:10px; padding: 0 3px;}"

        gbWFL = QGroupBox("Grafik Perkembangan — BB/PB (WFL)")
        gbWFL.setStyleSheet(style_graph)
        vW = QVBoxLayout(gbWFL)
        self.canvas_wfl = GrowthCanvasWFL()
        vW.addWidget(self.canvas_wfl)

        gbHFA = QGroupBox("Grafik Perkembangan — PB/U (HFA)")
        gbHFA.setStyleSheet(style_graph)
        vH = QVBoxLayout(gbHFA)
        self.canvas_hfa = GrowthCanvasHFA()
        vH.addWidget(self.canvas_hfa)

        graphs.addWidget(gbWFL, 1)
        graphs.addWidget(gbHFA, 1)
        root.addLayout(graphs)

        bar = QHBoxLayout()
        self.btn_measure = QPushButton("Ukur Bayi (Sensor)")
        self.btn_manual  = QPushButton("Input Manual")
        self.btn_print   = QPushButton("Cetak")
        
        for b in (self.btn_measure, self.btn_manual, self.btn_print):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("QPushButton { background-color: #2979ff; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #2962ff; }")

        bar.addStretch(1)
        bar.addWidget(self.btn_measure)
        bar.addWidget(self.btn_manual)
        bar.addWidget(self.btn_print)
        root.addLayout(bar)

        self.btn_measure.clicked.connect(self.do_measure)
        self.btn_manual.clicked.connect(self.do_manual)
        self.btn_print.clicked.connect(self.print_card)

    def set_baby(self, baby_row):
        self.current_baby = baby_row 
        self._load_data()

    def _load_data(self):
        if not self.current_baby: return
        bid, name, parent, dob, sex = self.current_baby

        self.f_name.setText(name)
        self.f_parent.setText(parent)
        self.f_dob.setText(dob)
        self.f_sex.setText("Laki-Laki" if sex == "L" else "Perempuan")
        
        try:
            d0 = datetime.datetime.strptime(dob, "%Y-%m-%d").date()
            self.f_age.setText(f"{calc_age_months(d0)} bulan")
        except: self.f_age.setText("-")

        cur = self.con.cursor()
        cur.execute("SELECT weight_kg, length_cm, ts, COALESCE(age_months,-1) FROM measure WHERE baby_id=? ORDER BY ts DESC", (bid,))
        rows = cur.fetchall()
        
        # Inisialisasi BabyClassifier untuk mengambil status spesifik
        clf = BabyClassifier()
        
        self.tbl_hist.setRowCount(0)
        for i, (w, l, ts, age_m) in enumerate(rows, start=1):
            if age_m < 0:
                try: 
                   d1 = datetime.datetime.fromisoformat(ts).date()
                   d0 = datetime.datetime.strptime(dob, "%Y-%m-%d").date()
                   age_m = months_between(d0, d1)
                except: age_m = 0
            
            # Memanggil status secara spesifik dari utils.py
            stat_hfa = clf.classify_stunting(sex, age_m, l)
            stat_wfl = clf.classify_wasting(sex, l, w)
            
            # Logika penggabungan teks status
            if stat_hfa == "Normal" and stat_wfl == "Gizi Baik (Normal)":
                status_str = "Normal"
            else:
                status_str = f"PB/U: {stat_hfa} | BB/PB: {stat_wfl}"
            
            r = self.tbl_hist.rowCount()
            self.tbl_hist.insertRow(r)
            self.tbl_hist.setItem(r, 0, QTableWidgetItem(str(i)))
            self.tbl_hist.setItem(r, 1, QTableWidgetItem(f"{w:.2f}"))
            self.tbl_hist.setItem(r, 2, QTableWidgetItem(f"{l:.1f}"))
            self.tbl_hist.setItem(r, 3, QTableWidgetItem(status_str))
            self.tbl_hist.setItem(r, 4, QTableWidgetItem(str(ts)))

        cur.execute("SELECT length_cm, weight_kg, ts FROM measure WHERE baby_id=? ORDER BY ts", (bid,))
        pts = cur.fetchall()
        self.canvas_wfl.plot(sex, pts)
        self.canvas_hfa.plot(sex, dob, pts)

    def do_measure(self):
        if not self.current_baby: return
        dlg = MeasureDialog(self.con, self.current_baby, self)
        if dlg.exec_(): self._load_data()

    def do_manual(self):
        if not self.current_baby: return
        dlg = ManualMeasureDialog(self.con, self.current_baby, self)
        if dlg.exec_(): self._load_data()

    def print_card(self):
        QMessageBox.information(self, "Info", "Fitur cetak kartu profil belum diimplementasikan di versi modular ini.")

# ================= ABOUT PAGE (CLEANED) =================
class AboutPage(QWidget):
    def __init__(self, con):
        super().__init__()
        self.con = con
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        
        # 1. Judul
        title = QLabel("Tentang Aplikasi")
        title.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        root.addWidget(title)
        
        # 2. Logo (Jika ada)
        if os.path.exists(LOGO_PATH):
            logo_holder = QWidget()
            vlogo = QVBoxLayout(logo_holder)
            vlogo.setContentsMargins(0,4,0,10)
            l = QLabel(); l.setAlignment(Qt.AlignCenter)
            pm = QPixmap(LOGO_PATH).scaled(140,140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            l.setPixmap(pm)
            vlogo.addWidget(l)
            root.addWidget(logo_holder)
        
        # 3. App Info Card
        app_card = QGroupBox()
        app_card.setStyleSheet("QGroupBox{background:#ffffff;border:1px solid #e9edf5;border-radius:12px;margin-top:0}")
        app_l = QVBoxLayout(app_card)
        app_l.setContentsMargins(16,12,16,12)
        app_l.addWidget(QLabel(f"<b>{APP_NAME}</b> v{APP_VERSION} — Aplikasi pencatatan dan pemantauan tumbuh kembang bayi (PB/BB)."))
        root.addWidget(app_card)
        
        # 4. Profil Pengembang & Dosen
        root.addWidget(self._person_card(DEV_PROFILE))
        root.addWidget(self._person_card(SUPV_PROFILE))
        
        # 5. Tools & Runtime Info telah DIHAPUS sesuai permintaan
        
        root.addStretch(1) # Agar konten terdorong ke atas

    def _person_card(self, data: dict) -> QGroupBox:
        card = QGroupBox()
        card.setStyleSheet("QGroupBox{background:#ffffff;border:1px solid #e6ebf4;border-radius:12px;margin-top:10px}")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12,10,12,10)
        lay.setSpacing(12)

        pic = QLabel()
        pic.setFixedSize(72, 72)
        pic.setStyleSheet("background:#f1f5f9;border:1px solid #e2e8f0;border-radius:12px")
        
        pm = self._load_photo(data.get("photo"))
        if pm: 
            pic.setPixmap(pm.scaled(pic.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        lay.addWidget(pic)

        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0,0,0,0)
        
        title = QLabel(data.get("title","-"))
        title.setStyleSheet("color:#1f2937;font-weight:700")
        
        name  = QLabel(f"Nama: {data.get('name','-')}")
        extra1 = data.get("nim") or data.get("nidn") or ""
        label1 = "NIM" if "nim" in data else ("NIDN/NIP" if "nidn" in data else "No.")
        line1 = QLabel(f"{label1}: {extra1}") if extra1 else QLabel("")
        inst  = QLabel(f"Instansi: {data.get('instansi','-')}")
        
        for w in (title, name, line1, inst):
            w.setStyleSheet("color:#334155;")
            v.addWidget(w)
        lay.addWidget(box, 1)
        return card

    def _load_photo(self, path: str) -> QPixmap:
        try:
            if path and os.path.exists(path):
                pm = QPixmap(path)
                if not pm.isNull(): return pm
        except Exception: pass
        
        # Placeholder jika foto tidak ada
        pm = QPixmap(72,72)
        pm.fill(Qt.white)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#e2e8f0")))
        p.setBrush(QBrush(QColor("#f8fafc")))
        p.drawRoundedRect(0,0,72,72,8,8)
        p.end()
        return pm
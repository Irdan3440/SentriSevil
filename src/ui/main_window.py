from PyQt5.QtWidgets import (QMainWindow, QStackedWidget, QListWidget, 
                             QListWidgetItem, QHBoxLayout, QVBoxLayout, 
                             QWidget, QLabel, QMessageBox, QApplication, QFrame)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont

from .pages import DashboardPage, DataPage, ProfilePage, AboutPage
from ..config import APP_NAME, APP_COPYRIGHT

class MainWindow(QMainWindow):
    def __init__(self, con):
        super().__init__()
        self.con = con
        self.setWindowTitle(APP_NAME)
        self.resize(1260, 820)
        
        # --- KONFIGURASI FONT GLOBAL ---
        # Mengatur font default aplikasi agar terlihat modern
        self.setFont(QFont("Segoe UI", 10))

        # --- WIDGET UTAMA & LAYOUT ---
        # Kita menggunakan QWidget sebagai container utama
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout utama adalah Vertikal: [Konten (Sidebar+Halaman)] + [Footer]
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- BAGIAN KONTEN (Sidebar + Halaman) ---
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 1. SIDEBAR (KIRI)
        # Container sidebar agar background warnanya penuh dari atas ke bawah
        self.sidebar_container = QWidget()
        self.sidebar_container.setFixedWidth(220) 
        self.sidebar_container.setStyleSheet("background-color: #0f2743;") # Warna Biru Gelap Asli
        
        sidebar_vbox = QVBoxLayout(self.sidebar_container)
        sidebar_vbox.setContentsMargins(0, 20, 0, 20)
        sidebar_vbox.setSpacing(10)

        # List Menu
        self.sidebar = QListWidget()
        self.sidebar.setFrameShape(QListWidget.NoFrame)
        self.sidebar.setFocusPolicy(Qt.NoFocus) # Hilangkan garis putus-putus saat klik
        
        # Styling Sidebar mirip CSS
        self.sidebar.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                color: white;
                font-size: 14px;
                outline: 0;
            }
            QListWidget::item {
                padding: 12px 20px;
                margin-bottom: 4px;
                border-left: 4px solid transparent;
            }
            QListWidget::item:selected {
                background-color: #153a62; /* Warna saat dipilih */
                border-left: 4px solid #1e88e5; /* Garis aksen di kiri */
                color: white;
                font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #112e4d;
            }
        """)

        # Menambahkan Item Menu
        menus = ["Dashboard", "Data Bayi", "Profil", "Tentang", "Keluar"]
        for m in menus:
            item = QListWidgetItem(m)
            self.sidebar.addItem(item)
            
        sidebar_vbox.addWidget(self.sidebar)
        content_layout.addWidget(self.sidebar_container)

        # 2. HALAMAN KANAN (STACKED WIDGET)
        self.pages_container = QWidget()
        # Warna background abu-abu muda seperti di gambar asli
        self.pages_container.setStyleSheet("background-color: #f5f7fb;") 
        
        pages_layout = QVBoxLayout(self.pages_container)
        pages_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stack = QStackedWidget()
        self.pg_dash = DashboardPage(con)
        self.pg_data = DataPage(con)
        self.pg_prof = ProfilePage(con)
        self.pg_about = AboutPage(con)
        
        self.stack.addWidget(self.pg_dash) # Index 0
        self.stack.addWidget(self.pg_data) # Index 1
        self.stack.addWidget(self.pg_prof) # Index 2
        self.stack.addWidget(self.pg_about)# Index 3
        
        pages_layout.addWidget(self.stack)
        content_layout.addWidget(self.pages_container)

        # Tambahkan Layout Konten ke Layout Utama
        main_layout.addLayout(content_layout)

        # --- FOOTER ---
        self.footer = QLabel(APP_COPYRIGHT)
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setFixedHeight(40)
        self.footer.setStyleSheet("""
            background-color: #f5f7fb;
            color: #607d8b;
            font-size: 11px;
            border-top: 1px solid #e0e0e0;
        """)
        main_layout.addWidget(self.footer)

        # --- SIGNALS & LOGIC ---
        self.sidebar.currentRowChanged.connect(self.switch_page)
        self.pg_data.open_profile.connect(self.go_to_profile)
        
        # Set halaman awal
        self.sidebar.setCurrentRow(0)

        # Styling tambahan untuk komponen global (tombol, tabel, dll)
        self.setStyleSheet("""
            QMainWindow { background: #f5f7fb; }
            QPushButton {
                font-size: 12px;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QTableWidget {
                background: white;
                border: 1px solid #ddd;
                gridline-color: #eee;
            }
            QHeaderView::section {
                background-color: #f1f2f6;
                padding: 4px;
                border: 0px;
                font-weight: bold;
                color: #555;
            }
        """)

    def switch_page(self, idx):
        # Logika Keluar (Index 4 adalah "Keluar")
        if idx == 4:
            if QMessageBox.question(self, "Keluar", "Tutup aplikasi?") == QMessageBox.Yes:
                QApplication.quit()
            else:
                # Kembalikan seleksi ke halaman sebelumnya jika batal
                current = self.stack.currentIndex()
                self.sidebar.blockSignals(True)
                self.sidebar.setCurrentRow(current if current < 2 else current + 1)
                self.sidebar.blockSignals(False)
            return

        # Mapping index sidebar ke index stack
        # Sidebar: 0=Dash, 1=Data, 2=Profil, 3=About
        # Stack:   0=Dash, 1=Data, 2=Profil, 3=About
        target_idx = idx 
        self.stack.setCurrentIndex(target_idx)
        
        # Refresh halaman jika diperlukan
        if target_idx == 0: self.pg_dash.refresh()
        if target_idx == 1: self.pg_data.reload()

    def go_to_profile(self, bid):
        cur = self.con.cursor()
        cur.execute("SELECT id, baby_name, parent_name, dob, sex FROM baby WHERE id=?", (bid,))
        row = cur.fetchone()
        if row:
            self.pg_prof.set_baby(row)
            self.stack.setCurrentWidget(self.pg_prof)
            
            # Update sidebar selection tanpa memicu signal switch_page berulang
            self.sidebar.blockSignals(True)
            self.sidebar.setCurrentRow(2) # Arahkan sidebar ke menu Profil
            self.sidebar.blockSignals(False)
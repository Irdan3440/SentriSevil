import os
import sys

# Matikan log oneDNN tensorflow yang mengganggu (opsional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QIcon

# Import modul internal dengan Absolute Import (Wajib untuk main.py)
from src.database import ensure_db, get_connection
from src.ui.main_window import MainWindow
from src.config import APP_NAME, ICON_PATH

def main():
    # 1. Pastikan Database & Tabel siap
    ensure_db()
    
    # 2. Buka Koneksi Databases
    con = get_connection()
    
    # 3. Setup Aplikasi Qt
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # --- PERBAIKAN TAMPILAN UNTUK RASPBERRY PI ---
    # Memaksa menggunakan style 'Fusion' yang lintas platform dan stabil
    app.setStyle("Fusion")

    # Memaksa warna elemen input agar teks selalu hitam di atas background putih
    # Ini mengatasi masalah teks putih/transparan pada tema gelap Raspberry Pi
    app.setStyleSheet("""
        QWidget {
            color: black;             /* Teks umum hitam */
        }
        QLineEdit, QComboBox, QSpinBox, QDateEdit, QDoubleSpinBox {
            background-color: white;  /* Background input putih */
            color: black;             /* Teks input hitam */
            border: 1px solid #a0a0a0;
            padding: 2px;
        }
        QLabel {
            color: black;             /* Label teks hitam */
        }
        /* Memperbaiki tampilan menu dropdown yang kadang gelap */
        QAbstractItemView {
            background-color: white;
            color: black;
            selection-background-color: #0078d7;
            selection-color: white;
        }
    """)
    # ----------------------------------------------
    
    # Set Icon Aplikasi Global
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    
    # Set Font Global agar tampilan lebih modern
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # 4. Buka Jendela Utama
    win = MainWindow(con)
    win.show()
    
    # 5. Jalankan Event Loop
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Terjadi Error: {e}")
    finally:
        # Tutup koneksi database saat aplikasi ditutup
        if con:
            con.close()

if __name__ == "__main__":
    main()
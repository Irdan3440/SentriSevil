import os

# --- APP INFO ---
APP_NAME      = "SentriSevil V2.0"
APP_VERSION   = "1.0.0"
APP_COPYRIGHT = "Copyright ©Irdan Ramadhani 2026"

# --- PATHS ---
# Base dir = folder tempat file ini berada (src/), jadi project root mundur satu level
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_FILE  = os.path.join(DATA_DIR, "posyandu.db")
# Path Model YOLO (Sesuai permintaan)
MODEL_PATH = os.path.join(DATA_DIR, "best.pt")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
# Path ke Logo: SentriSevil/src/assets/sentrisevil_logo.png
ICON_PATH = os.path.join(ASSETS_DIR, "sentrisevil_logo.png")
# Path Foto/Logo (Pastikan file ini ada di folder project Anda)
# Anda bisa menaruh file gambar di root folder SentriSevil
LOGO_PATH = os.path.join(PROJECT_ROOT, "sentrisevil_logo.png")

# --- PROFIL PENGEMBANG ---
DEV_PROFILE = {
    "title": "Pengembang",
    "name": "Irdan Ramadhani",
    "nim": "2240304024",
    "instansi": "Program Studi Teknik Komputer",
    "photo": os.path.join(PROJECT_ROOT, "Irdan.jpg"), # Pastikan file Irdan.jpg ada
}

SUPV_PROFILE = {
    "title": "Dosen Pembimbing",
    "name": "xxx",
    "nidn": "09XXXXXXXX",
    "instansi": "Program Studi Teknik Komputer",
    "photo": os.path.join(PROJECT_ROOT, "supv_photo.jpg"),
}

# perbaikan path agar dinamis
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # folder src
PROJECT_ROOT = os.path.dirname(BASE_DIR)              # folder SentriSevil
DATA_DIR = os.path.join(PROJECT_ROOT, "data")         # folder SentriSevil/data
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")     # folder SentriSevil/assets (Buat folder ini jika perlu)

DB_FILE = os.path.join(DATA_DIR, "posyandu.db")

# Jika logo ada di folder assets, ubah ini:
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
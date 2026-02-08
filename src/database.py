import sqlite3
import os

DB_NAME = "sentrisevil.db"

def init_db():
    """Inisialisasi database dan tabel."""
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    
    # 1. Tabel Bayi (DIPERBAIKI: Jadi 'baby', bukan 'babies')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS baby (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_name TEXT,
            baby_name TEXT,
            dob TEXT,
            sex TEXT,
            created_at TEXT
        )
    """)
    
    # 2. Tabel Pengukuran
    cur.execute("""
        CREATE TABLE IF NOT EXISTS measure (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            baby_id INTEGER,
            ts TEXT,
            length_cm REAL,
            weight_kg REAL,
            age_months INTEGER
        )
    """)

    # 3. Tabel Settings (Untuk Kalibrasi)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    con.commit()
    con.close()
    print(f"Database {DB_NAME} siap (tabel 'baby' & 'settings' tersedia).")

def get_db_connection():
    return sqlite3.connect(DB_NAME)

# --- FUNGSI HELPER SETTINGS (Untuk Kalibrasi) ---
def save_setting(key, value):
    """Menyimpan setting ke database."""
    con = get_db_connection()
    try:
        con.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        con.commit()
    except Exception as e:
        print(f"Gagal simpan setting {key}: {e}")
    finally:
        con.close()

def get_setting(key, default_value):
    """Mengambil setting dari database."""
    con = get_db_connection()
    try:
        cur = con.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        if row:
            return row[0]
        return default_value
    except Exception as e:
        print(f"Gagal baca setting {key}: {e}")
        return default_value
    finally:
        con.close()

# --- ALIAS UNTUK KOMPATIBILITAS (Agar main.py tidak error) ---
ensure_db = init_db
get_connection = get_db_connection
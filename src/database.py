import sqlite3
import os
from .config import DB_FILE, DATA_DIR

def ensure_db():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS baby(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        baby_name   TEXT NOT NULL,
        parent_name TEXT NOT NULL,
        dob TEXT NOT NULL,
        sex TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS measure(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        baby_id INTEGER NOT NULL,
        ts TEXT NOT NULL,
        length_cm REAL NOT NULL,
        weight_kg REAL NOT NULL,
        age_months INTEGER,
        FOREIGN KEY(baby_id) REFERENCES baby(id)
    )""")
    # Migrasi kolom jika perlu
    cur.execute("PRAGMA table_info(measure)")
    cols = {r[1] for r in cur.fetchall()}
    if "age_months" not in cols:
        cur.execute("ALTER TABLE measure ADD COLUMN age_months INTEGER")
    con.commit()
    con.close()

def get_connection():
    return sqlite3.connect(DB_FILE)
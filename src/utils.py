import os
import pandas as pd
import bisect
from datetime import date, datetime
from .config import DATA_DIR

# ==========================================
# 1. HELPER TANGGAL
# ==========================================
def months_between(d0: date, d1: date) -> int:
    """Menghitung selisih bulan penuh antara dua tanggal."""
    m = (d1.year - d0.year) * 12 + (d1.month - d0.month)
    if d1.day < d0.day:
        m -= 1
    return max(0, m)

def calc_age_months(dob: date) -> int:
    """Menghitung umur dalam bulan dari tanggal lahir sampai hari ini."""
    return months_between(dob, date.today())

# ==========================================
# 2. LOGIKA KLASIFIKASI BARU (BabyClassifier)
# ==========================================
class BabyClassifier:
    def __init__(self):
        self.ready = False
        try:
            # Pastikan nama file sesuai dengan yang ada di folder 'data'
            self.pbu_male = pd.read_csv(os.path.join(DATA_DIR, 'PBU_Laki_Laki_0_24_Bulan_Permenkes_2_2020.csv'))
            self.pbu_female = pd.read_csv(os.path.join(DATA_DIR, 'PBU_Perempuan_0_24_Bulan_Permenkes_2_2020.csv'))
            self.bbpb_male = pd.read_csv(os.path.join(DATA_DIR, 'BBPB_Laki_Laki_0_24_Bulan_Permenkes_2_2020.csv'))
            self.bbpb_female = pd.read_csv(os.path.join(DATA_DIR, 'BBPB_Perempuan_0_24_Bulan_Permenkes_2_2020.csv'))
            self.ready = True
        except Exception as e:
            print(f"[ERROR] Gagal memuat CSV Klasifikasi: {e}")

    def _get_table(self, gender, indicator):
        if not self.ready: return None
        g = str(gender).upper().strip()
        is_male = g == 'L' or g == 'LAKI-LAKI'
        
        if indicator == 'PBU': return self.pbu_male if is_male else self.pbu_female
        elif indicator == 'BBPB': return self.bbpb_male if is_male else self.bbpb_female
        return None

    def classify_stunting(self, gender, age_months, length_cm):
        """Klasifikasi PB/U (Stunting)"""
        df = self._get_table(gender, 'PBU')
        if df is None: return "Data Error"
        
        row = df[df['umur_bulan'] == age_months]
        if row.empty: return "Umur > 24 bln"
        
        row = row.iloc[0]
        val = float(length_cm)
        
        if val < row['minus_3_sd']: return "Sangat Pendek (Severely Stunted)"
        elif row['minus_3_sd'] <= val < row['minus_2_sd']: return "Pendek (Stunted)"
        elif row['minus_2_sd'] <= val <= row['plus_3_sd']: return "Normal"
        elif val > row['plus_3_sd']: return "Tinggi"
        return "Unknown"

    def classify_wasting(self, gender, length_cm, weight_kg):
        """Klasifikasi BB/PB (Wasting/Gizi)"""
        df = self._get_table(gender, 'BBPB')
        if df is None: return "Data Error"
        
        rounded_length = round(float(length_cm) * 2) / 2
        row = df[df['panjang_badan_cm'] == rounded_length]
        if row.empty: return "Panjang diluar range"
        
        row = row.iloc[0]
        val = float(weight_kg)
        
        if val < row['minus_3_sd']: return "Gizi Buruk (Severely Wasted)"
        elif row['minus_3_sd'] <= val < row['minus_2_sd']: return "Gizi Kurang (Wasted)"
        elif row['minus_2_sd'] <= val <= row['plus_1_sd']: return "Gizi Baik (Normal)"
        elif row['plus_1_sd'] < val <= row['plus_2_sd']: return "Berisiko Gizi Lebih"
        elif row['plus_2_sd'] < val <= row['plus_3_sd']: return "Gizi Lebih (Overweight)"
        elif val > row['plus_3_sd']: return "Obesitas"
        return "Unknown"

# ==========================================
# 3. FUNGSI LEGACY (UNTUK KOMPATIBILITAS PAGES.PY & WIDGETS.PY)
# ==========================================

# Fungsi wrapper untuk is_wasting_PB_BB (biar pages.py tidak error)
def is_wasting_PB_BB(length_cm, weight_kg, sex):
    try:
        classifier = BabyClassifier()
        if not classifier.ready: return None
        res = classifier.classify_wasting(sex, length_cm, weight_kg)
        # Logika sederhana: True jika masalah gizi (kurang/buruk), False jika normal/lebih
        # Sesuaikan dengan kebutuhan dashboard Anda
        if "Kurang" in res or "Buruk" in res: return True
        return False
    except: return None

# Fungsi wrapper untuk is_stunting_HFA (biar pages.py tidak error)
def is_stunting_HFA(age_months, length_cm, sex):
    try:
        classifier = BabyClassifier()
        if not classifier.ready: return None
        res = classifier.classify_stunting(sex, age_months, length_cm)
        # Logika sederhana: True jika pendek/sangat pendek
        if "Pendek" in res: return True
        return False
    except: return None

# Fungsi load kurva untuk grafik di widgets.py
def load_wfl_curves(sex: str):
    fname = 'BBPB_Laki_Laki_0_24_Bulan_Permenkes_2_2020.csv' if sex == 'L' else 'BBPB_Perempuan_0_24_Bulan_Permenkes_2_2020.csv'
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        return {
            "x": df['panjang_badan_cm'].tolist(),
            "-2sd": df['minus_2_sd'].tolist(),
            "med": df['median'].tolist(),
            "+2sd": df['plus_2_sd'].tolist()
        }
    except: return None

def load_hfa_curves(sex: str):
    fname = 'PBU_Laki_Laki_0_24_Bulan_Permenkes_2_2020.csv' if sex == 'L' else 'PBU_Perempuan_0_24_Bulan_Permenkes_2_2020.csv'
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        return {
            "m": df['umur_bulan'].tolist(),
            "-2sd": df['minus_2_sd'].tolist(),
            "med": df['median'].tolist(),
            "+2sd": df['plus_2_sd'].tolist()
        }
    except: return None
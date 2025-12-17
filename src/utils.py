import os
import pandas as pd
from datetime import date, datetime
from .config import DATA_DIR

# --- Helper Tanggal ---
def months_between(d0: date, d1: date) -> int:
    """Menghitung selisih bulan penuh antara dua tanggal."""
    m = (d1.year - d0.year) * 12 + (d1.month - d0.month)
    if d1.day < d0.day:
        m -= 1
    return max(0, m)

def calc_age_months(dob: date) -> int:
    """Menghitung umur dalam bulan dari tanggal lahir sampai hari ini."""
    return months_between(dob, date.today())

# --- Logic Klasifikasi Gizi (Z-Score) ---
class BabyClassifier:
    def __init__(self):
        # Load dataset CSV saat inisialisasi agar tidak berat saat runtime
        try:
            self.pbu_male = pd.read_csv(os.path.join(DATA_DIR, 'PBU_Laki_Laki_0_24_Bulan_Permenkes_2_2020.csv'))
            self.pbu_female = pd.read_csv(os.path.join(DATA_DIR, 'PBU_Perempuan_0_24_Bulan_Permenkes_2_2020.csv'))
            self.bbpb_male = pd.read_csv(os.path.join(DATA_DIR, 'BBPB_Laki_Laki_0_24_Bulan_Permenkes_2_2020.csv'))
            self.bbpb_female = pd.read_csv(os.path.join(DATA_DIR, 'BBPB_Perempuan_0_24_Bulan_Permenkes_2_2020.csv'))
            self.ready = True
        except Exception as e:
            print(f"[ERROR] Gagal memuat data CSV Antropometri: {e}")
            self.ready = False

    def _get_table(self, gender, indicator):
        if not self.ready: return None
        # Normalisasi input gender (L/P atau Laki-laki/Perempuan)
        g = str(gender).lower().strip()
        is_male = g in ['l', 'laki-laki', 'laki']
        
        if indicator == 'PBU':
            return self.pbu_male if is_male else self.pbu_female
        elif indicator == 'BBPB':
            return self.bbpb_male if is_male else self.bbpb_female
        return None

    def classify_stunting(self, gender, age_months, length_cm):
        """Klasifikasi PB/U (Stunting)"""
        df = self._get_table(gender, 'PBU')
        if df is None: return "Data Error"
        
        # Cari baris berdasarkan umur
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
        
        # Pembulatan panjang badan ke kelipatan 0.5 terdekat (sesuai standar CSV)
        rounded_length = round(float(length_cm) * 2) / 2
        
        row = df[df['panjang_badan_cm'] == rounded_length]
        if row.empty: return "Panjang di luar range"
        
        row = row.iloc[0]
        val = float(weight_kg)
        
        if val < row['minus_3_sd']: return "Gizi Buruk (Severely Wasted)"
        elif row['minus_3_sd'] <= val < row['minus_2_sd']: return "Gizi Kurang (Wasted)"
        elif row['minus_2_sd'] <= val <= row['plus_1_sd']: return "Gizi Baik (Normal)"
        elif row['plus_1_sd'] < val <= row['plus_2_sd']: return "Berisiko Gizi Lebih"
        elif row['plus_2_sd'] < val <= row['plus_3_sd']: return "Gizi Lebih (Overweight)"
        elif val > row['plus_3_sd']: return "Obesitas"
        return "Unknown"

# --- Fungsi Legacy (Opsional, simpan jika ada modul lain yg pakai) ---
# Jika widgets.py masih pakai load_wfl_curves, biarkan fungsi lama di bawah ini
# Tapi lebih baik refactor widgets.py nanti untuk pakai BabyClassifier juga.
def load_wfl_curves(sex): return None 
def load_hfa_curves(sex): return None
def is_wasting_PB_BB(l, w, s): return None
def is_stunting_HFA(a, l, s): return None
import mediapipe as mp
try:
    print(f"Path Mediapipe: {mp.__file__}")
    pose = mp.solutions.pose
    print("SUKSES: mp.solutions.pose ditemukan!")
except AttributeError:
    print("GAGAL: mp.solutions.pose TIDAK ditemukan. Masih ada konflik file/instalasi.")
except Exception as e:
    print(f"ERROR Lain: {e}")
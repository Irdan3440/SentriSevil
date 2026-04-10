import cv2

# Buka koneksi ke webcam (biasanya indeks 0 untuk kamera default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak dapat diakses!")
    exit()

print("Tekan tombol 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Gagal mengambil frame.")
        break
        
    # Tampilkan frame dari webcam
    cv2.imshow('Tes Webcam', frame)
    
    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()q
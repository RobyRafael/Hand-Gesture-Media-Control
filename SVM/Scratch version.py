import cv2  # Library untuk pemrosesan gambar, pip install opencv-python
import numpy as np  # Library untuk operasi numerik dan array
import pyautogui  # Library untuk mengontrol keyboard dan mouse
import time  # Library untuk fungsi waktu dan delay
import os  # Library untuk operasi sistem dan file
import pickle  # Library untuk menyimpan dan memuat objek Python
import math  # Library untuk fungsi matematika
import csv  # Library untuk membaca dan menulis file CSV
from datetime import datetime  # Library untuk manajemen waktu dan tanggal

# Pengaturan direktori
script_dir = os.path.dirname(os.path.abspath(__file__))  # Dapatkan lokasi file saat ini
screenshots_dir = os.path.join(script_dir, 'custom_gesture_screenshots')  # Buat path untuk folder screenshot
if not os.path.exists(screenshots_dir):  # Jika folder belum ada
    os.makedirs(screenshots_dir)  # Buat folder tersebut
    print(f"Created directory: {screenshots_dir}")  # Tampilkan pesan berhasil

# Path untuk menyimpan model dan data
model_path = os.path.join(script_dir, 'custom_svm_model.pkl')  # Path untuk file model SVM
training_data_path = os.path.join(script_dir, 'custom_svm_training_data.pkl')  # Path untuk file data pelatihan
csv_data_path = os.path.join(script_dir, 'custom_svm_training_data.csv')  # Path untuk file CSV data pelatihan

# Pemetaan kelas gerakan tangan
gesture_classes = {
    0: "No gesture detected",  # Tidak ada gerakan terdeteksi
    1: "Play/Pause",  # Untuk memulai/menghentikan sementara media
    2: "Stop",  # Untuk menghentikan media
    3: "Next Track",  # Untuk beralih ke trek berikutnya
    4: "Previous Track",  # Untuk beralih ke trek sebelumnya
    5: "Volume Up",  # Untuk menaikkan volume
    6: "Volume Down"  # Untuk menurunkan volume
}

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # Mulai webcam dengan indeks 1 (ganti ke 0 jika tidak berfungsi)
running = True  # Flag untuk menjalankan program

# Atur resolusi kamera (opsional, sesuaikan dengan kemampuan kamera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variabel untuk panel informasi
info_panel_width = 400  # Lebar panel informasi di sisi kanan (pixel)
calculation_panel_color = (30, 30, 30)  # Warna latar panel informasi (abu-abu gelap)

# Warna untuk visualisasi pada frame
colors = {
    "contour": (0, 255, 0),            # Hijau untuk kontur tangan
    "hull": (0, 0, 255),               # Merah untuk hull
    "defect_point": (0, 0, 255),       # Merah untuk titik defect
    "defect_line": (0, 255, 0),        # Hijau untuk garis defect
    "bounding_box": (255, 255, 0),     # Kuning untuk bounding box
    "ellipse": (0, 255, 255),          # Cyan untuk ellipse
    "centroid": (255, 0, 255),         # Magenta untuk pusat massa
    "finger_tip": (0, 255, 127),       # Hijau-biru untuk ujung jari
    "text_bg": (0, 0, 0),              # Hitam untuk latar teks
    "text": (255, 255, 255),           # Putih untuk teks
    "convexity_good": (0, 255, 0),     # Hijau untuk convexity baik
    "convexity_medium": (0, 255, 255), # Kuning untuk convexity sedang    "convexity_bad": (0, 100, 255),    # Oranye untuk convexity rendah
    "metric_bg": (50, 50, 50),         # Latar untuk metrik
    "metric_fill": (0, 180, 180),      # Warna isi bar metrik
    "hull_area": (150, 150, 255),      # Warna area hull
    "contour_area": (150, 255, 150),    # Warna area kontur
    "feature_bar_bg": (40, 40, 40),    # Latar belakang bar fitur
    "feature_importance": (0, 200, 255), # Warna bar kepentingan fitur
    "movement_trail": (255, 100, 0),   # Warna jejak gerakan
    "feature_highlight": (255, 255, 0) # Warna highlight fitur penting
}

# Font untuk teks pada kamera
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale_small = 0.4
font_scale_medium = 0.6
font_scale_large = 0.8
font_thickness_small = 1
font_thickness_medium = 1
font_thickness_large = 2

# Variabel untuk pengendalian aksi
last_action_time = 0  # Waktu terakhir aksi dilakukan
cooldown_time = 0.5  # Waktu jeda antar aksi (dalam detik)
previous_gesture = "No gesture detected"  # Menyimpan gerakan sebelumnya

# Variabel untuk pengumpulan data pelatihan
collecting_data = False  # Status pengumpulan data (awalnya tidak aktif)
training_data = []  # Daftar untuk menyimpan data fitur
training_labels = []  # Daftar untuk menyimpan label kelas
current_collection_class = 0  # Kelas gerakan yang sedang dikumpulkan
sample_cooldown = 0  # Jeda antar pengambilan sampel

# ============= DETEKSI TANGAN DAN EKSTRAKSI FITUR CUSTOM =============
def put_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness=1, padding=5):
    """Menaruh teks dengan latar belakang untuk meningkatkan keterbacaan"""
    # Dapatkan dimensi teks
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Hitung posisi dan dimensi latar belakang
    x, y = position
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + padding
    
    # Gambar latar belakang
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Gambar teks
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    
    return img

def draw_metric_bar(img, position, width, height, value, max_value=1.0, colors=None, label=None, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Menggambar bar metrik untuk visualisasi nilai"""
    x, y = position
    
    if colors is None:
        colors = {
            "bg": (50, 50, 50),
            "fill": (0, 255, 0),
            "text": (255, 255, 255)
        }
    
    # Gambar latar bar
    cv2.rectangle(img, (x, y), (x + width, y + height), colors["bg"], -1)
    
    # Gambar nilai pada bar
    fill_width = int((value / max_value) * width)
    fill_width = max(0, min(fill_width, width))  # Pastikan dalam rentang yang valid
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), colors["fill"], -1)
    
    # Gambar border
    cv2.rectangle(img, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
    # Jika label diberikan, tampilkan
    if label:
        cv2.putText(img, f"{label}: {value:.2f}", (x, y - 5), font, 0.4, colors["text"], 1)
    
    return img

def visualize_convexity(img, contour, hull_points, convexity, position):
    """Menggambar visualisasi perbandingan convexity (area kontur vs hull)"""
    x, y = position
    width, height = 120, 80
    
    # Buat mini box untuk visualisasi
    viz_img = np.zeros((height, width, 3), dtype=np.uint8)
    viz_img[:, :] = (50, 50, 50)  # Latar belakang abu-abu
    
    # Gambar border
    cv2.rectangle(viz_img, (0, 0), (width-1, height-1), (200, 200, 200), 1)
    
    # Ukur convexity untuk memilih warna
    if convexity > 0.9:
        fill_color = colors["convexity_good"]
        label_color = (255, 255, 255)
    elif convexity > 0.7:
        fill_color = colors["convexity_medium"]
        label_color = (0, 0, 0)
    else:
        fill_color = colors["convexity_bad"]
        label_color = (255, 255, 255)
    
    # Skalakan dan offset kontur untuk visualisasi mini
    scale_factor = min(width / max(max(c[0][0] for c in contour), 1),
                      height / max(max(c[0][1] for c in contour), 1)) * 0.8
    
    # Temukan centroid untuk offset
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    offset_x = width // 2 - int(cx * scale_factor)
    offset_y = height // 2 - int(cy * scale_factor)
    
    # Skalakan kontur untuk visualisasi
    mini_contour = np.array([[[int(p[0][0] * scale_factor) + offset_x, 
                               int(p[0][1] * scale_factor) + offset_y]] for p in contour])
    
    # Skalakan hull untuk visualisasi
    mini_hull = np.array([[[int(p[0] * scale_factor) + offset_x, 
                            int(p[1] * scale_factor) + offset_y]] for p in hull_points])
    
    # Gambar hull pada visualisasi mini
    cv2.drawContours(viz_img, [mini_hull], 0, colors["hull_area"], -1)
    
    # Gambar kontur pada visualisasi mini
    cv2.drawContours(viz_img, [mini_contour], 0, colors["contour_area"], -1)
    
    # Tambahkan teks rasio
    cv2.putText(viz_img, f"Ratio: {convexity:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, label_color, 1)
    
    # Tempatkan visualisasi pada image utama
    img[y:y+height, x:x+width] = viz_img
    
    return img

def detect_skin(frame):
    """Mendeteksi kulit menggunakan segmentasi warna dalam ruang YCrCb"""
    # Konversi ke YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Tentukan rentang warna kulit untuk segmentasi
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)  # Batas bawah warna kulit
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)  # Batas atas warna kulit
    
    # Buat mask untuk warna kulit
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Terapkan operasi morfologi untuk membersihkan mask
    kernel = np.ones((5, 5), np.uint8)  # Kernel untuk operasi morfologi
    mask = cv2.dilate(mask, kernel, iterations=2)  # Pelebaran untuk mengisi celah kecil
    mask = cv2.erode(mask, kernel, iterations=2)  # Erosi untuk menghapus noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Penghalusan tepi
    
    return mask

def find_contours(mask):
    """Mencari kontur dalam mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Temukan kontur terbesar (diasumsikan sebagai tangan)
    if contours:
        return max(contours, key=cv2.contourArea)  # Kembalikan kontur dengan area terbesar
    return None

def find_convex_hull_and_defects(contour):
    """Mencari convex hull dan convexity defects dari kontur"""
    if contour is None or len(contour) < 5:  # Pastikan kontur valid
        return None, None, None
    
    # Temukan convex hull
    hull = cv2.convexHull(contour, returnPoints=False)  # Dapatkan indeks hull
    
    # Temukan convexity defects
    try:
        defects = cv2.convexityDefects(contour, hull)  # Dapatkan defects (celah antara jari)
    except:
        return contour, hull, None
    
    return contour, hull, defects

def extract_features(contour, defects, frame_shape):
    """Mengekstrak fitur tangan dari kontur dan defects"""
    if contour is None or defects is None:
        return None
    
    # Inisialisasi fitur
    features = []
    height, width, _ = frame_shape
    
    # 1. Ekstrak area dan perimeter kontur
    area = cv2.contourArea(contour)  # Luas area tangan
    perimeter = cv2.arcLength(contour, True)  # Panjang keliling tangan
    
    # Normalisasi berdasarkan dimensi frame
    normalized_area = area / (height * width)  # Area dinormalisasi
    features.append(normalized_area)
    
    # 2. Hitung convexity (rasio area kontur terhadap area convex hull)
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        convexity = area / hull_area  # Rasio convexity
        features.append(convexity)
    else:
        features.append(0)
    
    # 3. Temukan convexity defects (ruang antara jari)
    finger_count = 0  # Hitungan jari
    defect_distances = []  # Jarak defects
    
    # Temukan centroid (pusat massa) kontur
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])  # Koordinat x pusat
        cy = int(M["m01"] / M["m00"])  # Koordinat y pusat
    else:
        cx, cy = 0, 0
    
    # Proses defects untuk menemukan struktur seperti jari
    if defects is not None and len(defects) > 0:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]  # Start, end, far point, dan distance
            start = tuple(contour[s][0])  # Titik awal
            end = tuple(contour[e][0])  # Titik akhir
            far = tuple(contour[f][0])  # Titik terjauh
            
            # Hitung jarak antar titik
            dist_to_center = math.sqrt((far[0] - cx)**2 + (far[1] - cy)**2)  # Jarak ke pusat
            normalized_dist = dist_to_center / math.sqrt(width**2 + height**2)  # Normalisasi jarak
            defect_distances.append(normalized_dist)
            
            # Hitung jumlah jari potensial menggunakan sudut
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)  # Sisi a segitiga
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)  # Sisi b segitiga
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)  # Sisi c segitiga
            
            # Terapkan hukum kosinus untuk menemukan sudut
            if a*b > 0:
                angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))  # Hitung sudut
                
                # Jika sudut kurang dari 90 derajat, mungkin ini adalah ruang antar jari
                if angle <= math.pi/2:
                    finger_count += 1
    
    # 4. Tambahkan jumlah jari ke fitur
    features.append(finger_count)
    
    # 5. Tambahkan jarak defect (hingga 5)
    defect_distances.sort(reverse=True)  # Urutkan jarak dari yang terbesar
    for i in range(min(5, len(defect_distances))):
        features.append(defect_distances[i])
    
    # Padding jika defect kurang dari 5
    while len(features) < 8:  # 3 fitur dasar + 5 jarak defect
        features.append(0)
    
    # 6. Tambahkan rasio aspek bounding rect
    x, y, w, h = cv2.boundingRect(contour)  # Dapatkan persegi pembatas
    aspect_ratio = float(w) / h if h > 0 else 0  # Hitung rasio aspek
    features.append(aspect_ratio)
    
    # 7. Tambahkan orientasi tangan (sudut dari elips)
    if len(contour) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)  # Dapatkan elips yang cocok
        features.append(angle / 180.0)  # Normalisasi ke 0-1
    else:
        features.append(0)
    
    return features

# ============= IMPLEMENTASI SVM KUSTOM DARI AWAL =============
class CustomSVM:
    """Implementasi SVM kustom tanpa menggunakan library machine learning"""
    def __init__(self, C=1.0, max_iter=1000, tol=1e-3):
        self.C = C  # Parameter regularisasi
        self.max_iter = max_iter  # Iterasi maksimum
        self.tol = tol  # Toleransi konvergensi
        self.models = {}  # Menyimpan model SVM biner untuk one-vs-rest
        self.trained = False  # Status pelatihan
        self.classes = None  # Kelas yang tersedia
        
    def _linear_kernel(self, x1, x2):
        """Fungsi kernel linear sederhana"""
        return np.dot(x1, x2)  # Hasil dot product
    
    def _rbf_kernel(self, x1, x2, gamma=0.1):
        """Fungsi kernel RBF (Gaussian)"""
        return np.exp(-gamma * np.sum((x1 - x2) ** 2))  # Fungsi eksponensial kernel Gaussian
        
    def _train_binary_svm(self, X, y, positive_class):
        """Melatih pengklasifikasi SVM biner menggunakan algoritma SMO yang disederhanakan"""
        n_samples, n_features = X.shape  # Jumlah sampel dan fitur
        
        # Konversi label ke biner {-1, 1}
        binary_y = np.where(y == positive_class, 1, -1)  # 1 untuk kelas positif, -1 untuk lainnya
        
        # Inisialisasi parameter
        alphas = np.zeros(n_samples)  # Koefisien alpha
        b = 0  # Bias
        
        # Hitung matriks kernel (gunakan kernel linear untuk kesederhanaan)
        K = np.zeros((n_samples, n_samples))  # Matriks kernel
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._linear_kernel(X[i], X[j])  # Hitung kernel untuk setiap pasangan sampel
        
        # Algoritma SMO yang disederhanakan
        iter_count = 0
        while iter_count < self.max_iter:
            alpha_changed = 0  # Penghitung alpha yang diubah
            
            for i in range(n_samples):
                # Hitung error
                f_i = np.sum(alphas * binary_y * K[i]) + b  # Prediksi saat ini
                E_i = f_i - binary_y[i]  # Error
                
                # Periksa kondisi KKT
                if (binary_y[i] * E_i < -self.tol and alphas[i] < self.C) or \
                   (binary_y[i] * E_i > self.tol and alphas[i] > 0):
                    
                    # Pilih j acak != i
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Hitung error untuk j
                    f_j = np.sum(alphas * binary_y * K[j]) + b  # Prediksi untuk j
                    E_j = f_j - binary_y[j]  # Error untuk j
                    
                    # Simpan nilai alpha lama
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    
                    # Hitung batas L dan H
                    if binary_y[i] != binary_y[j]:
                        L = max(0, alphas[j] - alphas[i])  # Batas bawah
                        H = min(self.C, self.C + alphas[j] - alphas[i])  # Batas atas
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)  # Batas bawah
                        H = min(self.C, alphas[i] + alphas[j])  # Batas atas
                    
                    if L == H:
                        continue
                    
                    # Hitung eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]  # Turunan kedua
                    if eta >= 0:
                        continue
                    
                    # Perbarui alpha_j
                    alphas[j] = alpha_j_old - binary_y[j] * (E_i - E_j) / eta  # Update alpha j
                    
                    # Clip alpha_j
                    alphas[j] = max(L, min(H, alphas[j]))  # Batasi dalam rentang [L,H]
                    
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Perbarui alpha_i
                    alphas[i] = alpha_i_old + binary_y[i] * binary_y[j] * (alpha_j_old - alphas[j])  # Update alpha i
                    
                    # Perbarui threshold b
                    b1 = b - E_i - binary_y[i] * (alphas[i] - alpha_i_old) * K[i, i] - \
                         binary_y[j] * (alphas[j] - alpha_j_old) * K[i, j]  # b1 untuk i
                    
                    b2 = b - E_j - binary_y[i] * (alphas[i] - alpha_i_old) * K[i, j] - \
                         binary_y[j] * (alphas[j] - alpha_j_old) * K[j, j]  # b2 untuk j
                    
                    if 0 < alphas[i] < self.C:
                        b = b1  # Gunakan b1 jika i adalah support vector
                    elif 0 < alphas[j] < self.C:
                        b = b2  # Gunakan b2 jika j adalah support vector
                    else:
                        b = (b1 + b2) / 2  # Gunakan rata-rata
                    
                    alpha_changed += 1
            
            if alpha_changed == 0:
                iter_count += 1  # Tingkatkan penghitung iterasi jika tidak ada perubahan
            else:
                iter_count = 0  # Reset penghitung jika ada perubahan
        
        # Temukan support vectors
        sv_indices = alphas > 1e-5  # Indeks support vector
        support_vectors = X[sv_indices]  # Support vector
        support_alphas = alphas[sv_indices]  # Alpha untuk support vector
        support_labels = binary_y[sv_indices]  # Label untuk support vector
        
        # Hitung bobot untuk SVM linear
        w = np.sum((support_alphas * support_labels).reshape(-1, 1) * support_vectors, axis=0)  # Bobot
        
        return {
            'w': w,  # Bobot
            'b': b,  # Bias
            'support_vectors': support_vectors,  # Support vector
            'support_alphas': support_alphas,  # Alpha
            'support_labels': support_labels  # Label
        }
    
    def fit(self, X, y):
        """Melatih SVM multi-kelas menggunakan pendekatan one-vs-rest"""
        X = np.array(X)  # Konversi ke array numpy
        y = np.array(y)  # Konversi ke array numpy
        
        # Dapatkan kelas unik
        self.classes = np.unique(y)  # Daftar kelas unik
        
        if len(self.classes) < 2:
            raise ValueError("Butuh setidaknya 2 kelas untuk klasifikasi")  # Error jika kurang dari 2 kelas
        
        # Latih SVM biner untuk setiap kelas (one-vs-rest)
        for cls in self.classes:
            print(f"Melatih SVM untuk kelas {cls}...")
            self.models[cls] = self._train_binary_svm(X, y, cls)  # Latih untuk setiap kelas
        
        self.trained = True  # Set status pelatihan menjadi selesai
        return self
    
    def _predict_binary(self, x, model):
        """Membuat prediksi biner menggunakan model terlatih"""
        return np.dot(x, model['w']) + model['b']  # Fungsi keputusan SVM
    
    def predict(self, X):
        """Memprediksi label kelas untuk sampel dalam X"""
        if not self.trained:
            return np.zeros(len(X))  # Kembalikan nol jika belum dilatih
        
        X = np.array(X)  # Konversi ke array numpy
        n_samples = X.shape[0]  # Jumlah sampel
        predictions = np.zeros(n_samples)  # Inisialisasi prediksi
        
        for i in range(n_samples):
            scores = {}  # Skor untuk setiap kelas
            for cls, model in self.models.items():
                # Hitung skor keputusan
                scores[cls] = self._predict_binary(X[i], model)  # Skor untuk setiap kelas
            
            # Pilih kelas dengan skor tertinggi
            predictions[i] = max(scores.items(), key=lambda x: x[1])[0]  # Kelas dengan skor tertinggi
        
        return predictions
    
    def _sigmoid(self, x, a=1, b=0):
        """Fungsi sigmoid untuk mengubah output SVM menjadi probabilitas"""
        return 1 / (1 + np.exp(a * x + b))  # Fungsi sigmoid
    
    def predict_proba(self, X):
        """Mengembalikan estimasi probabilitas untuk sampel dalam X"""
        if not self.trained:
            return np.zeros((len(X), len(gesture_classes)))  # Kembalikan nol jika belum dilatih
        
        X = np.array(X)  # Konversi ke array numpy
        n_samples = X.shape[0]  # Jumlah sampel
        n_classes = len(gesture_classes)  # Jumlah kelas
        probas = np.zeros((n_samples, n_classes))  # Inisialisasi probabilitas
        
        for i in range(n_samples):
            # Dapatkan skor mentah untuk setiap kelas
            raw_scores = np.zeros(n_classes)  # Skor untuk setiap kelas
            for cls, model in self.models.items():
                if cls < n_classes:  # Pastikan indeks kelas dalam batas
                    raw_scores[cls] = self._predict_binary(X[i], model)  # Skor untuk kelas ini
            
            # Terapkan softmax untuk mendapatkan probabilitas
            exp_scores = np.exp(raw_scores - np.max(raw_scores))  # Kurangi max untuk stabilitas numerik
            probas[i] = exp_scores / np.sum(exp_scores)  # Normalisasi menjadi probabilitas
        
        return probas

# Inisialisasi classifier
classifier = CustomSVM(C=1.0, max_iter=100)  # Menggunakan iterasi lebih sedikit untuk mempercepat pelatihan

# Fungsi untuk menyimpan screenshot dengan informasi gerakan
def save_screenshot(frame, class_id, sample_number):
    # Buat direktori khusus kelas jika belum ada
    class_dir = os.path.join(screenshots_dir, f"class_{class_id}_{gesture_classes[class_id]}")
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    # Buat nama file dengan timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(class_dir, f"sample_{sample_number}_{timestamp}.jpg")
    
    # Tambahkan label kelas ke frame
    labeled_frame = frame.copy()
    cv2.putText(labeled_frame, f"Class: {class_id} - {gesture_classes[class_id]}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Simpan screenshot
    cv2.imwrite(filename, labeled_frame)
    print(f"Saved screenshot: {filename}")

# Fungsi untuk menyimpan data pelatihan ke file CSV
def save_training_data_to_csv():
    global training_data, training_labels
    
    if len(training_data) == 0 or len(training_labels) == 0:
        print("Tidak ada data pelatihan yang tersedia untuk disimpan ke CSV.")
        return False
    
    try:
        # Buat header untuk file CSV
        header = []
        
        # Nama-nama untuk fitur
        feature_names = [
            'normalized_area',          # Area yang dinormalisasi
            'convexity',                # Rasio convexity
            'finger_count'              # Jumlah jari terdeteksi
        ]
        
        # Tambahkan nama untuk jarak defect (hingga 5)
        for i in range(5):
            feature_names.append(f'defect_distance_{i}')
        
        # Tambahkan nama fitur tambahan
        feature_names.extend(['aspect_ratio', 'orientation'])
        
        # Tambahkan nama fitur ke header
        header.extend(feature_names)
        
        # Tambahkan label kelas ke header
        header.append('gesture_class')
        
        # Buat data baris untuk setiap sampel
        rows = []
        for i in range(len(training_data)):
            features = training_data[i]
            label = training_labels[i]
            
            # Gabungkan fitur dan label dalam satu baris
            row = list(features) + [label]
            rows.append(row)
        
        # Tulis ke file CSV
        with open(csv_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        
        print(f"Data pelatihan disimpan ke CSV: {csv_data_path} ({len(rows)} sampel)")
        return True
    except Exception as e:
        print(f"Error menyimpan data ke CSV: {e}")
        return False

# Fungsi untuk memuat data pelatihan dari file CSV
def load_training_data_from_csv():
    global training_data, training_labels
    
    if not os.path.exists(csv_data_path):
        print("File CSV data pelatihan tidak ditemukan.")
        return False
    
    try:
        training_data = []
        training_labels = []
        
        # Baca dari file CSV
        with open(csv_data_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Lewati baris header
            
            for row in reader:
                if len(row) >= 10:  # Minimal 9 fitur + 1 label
                    try:
                        # Ambil fitur (semua kolom kecuali yang terakhir)
                        features = [float(val) for val in row[:-1]]
                        
                        # Ambil label (kolom terakhir)
                        label = int(row[-1])
                        
                        # Tambahkan ke data pelatihan
                        training_data.append(features)
                        training_labels.append(label)
                    except ValueError:
                        print(f"Warning: Melewati baris CSV yang tidak valid")
        
        print(f"Dimuat {len(training_data)} sampel dari CSV: {csv_data_path}")
        return len(training_data) > 0
    except Exception as e:
        print(f"Error memuat data dari CSV: {e}")
        return False

# Ubah fungsi save_training_data untuk juga menyimpan ke CSV
def save_training_data():
    global training_data, training_labels
    
    if len(training_data) == 0 or len(training_labels) == 0:
        print("Tidak ada data pelatihan yang tersedia untuk disimpan.")
        return False
    
    try:
        # Simpan data menggunakan pickle
        with open(training_data_path, 'wb') as f:
            pickle.dump((training_data, training_labels), f)
        
        print(f"Data pelatihan disimpan ke {training_data_path} ({len(training_data)} sampel)")
        
        # Juga simpan ke CSV
        save_training_data_to_csv()
        
        return True
    except Exception as e:
        print(f"Error menyimpan data pelatihan: {e}")
        return False

# Ubah fungsi load_training_data untuk coba memuat dari CSV jika pickle gagal
def load_training_data():
    global training_data, training_labels
    
    # Coba muat dari file pickle terlebih dahulu
    if os.path.exists(training_data_path):
        try:
            # Muat data menggunakan pickle
            with open(training_data_path, 'rb') as f:
                training_data, training_labels = pickle.load(f)
            
            print(f"Dimuat {len(training_data)} sampel dari {training_data_path}")
            
            # Pastikan data CSV juga up-to-date
            save_training_data_to_csv()
            
            return len(training_data) > 0
        except Exception as e:
            print(f"Error memuat data dari pickle: {e}")
            # Jika gagal, coba muat dari CSV
            return load_training_data_from_csv()
    else:
        # Jika file pickle tidak ada, coba muat dari CSV
        return load_training_data_from_csv()

# Modifikasi fungsi train_model untuk menyimpan CSV setelah pelatihan
def train_model():
    global classifier, training_data, training_labels
    
    if len(training_data) == 0 or len(training_labels) == 0:
        print("Tidak ada data pelatihan tersedia.")
        return False
    
    try:
        # Periksa apakah kita memiliki cukup sampel dari setiap kelas
        class_counts = {}
        for label in training_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Cetak distribusi kelas
        print("Distribusi kelas dalam data pelatihan:")
        for cls, count in class_counts.items():
            print(f"  Kelas {cls} ({gesture_classes.get(cls, 'Unknown')}): {count} sampel")
        
        # Pastikan kita memiliki setidaknya 2 kelas
        if len(class_counts) < 2:
            print("Error: Butuh setidaknya 2 kelas berbeda untuk melatih model.")
            return False
        
        # Konversi ke array numpy
        X = np.array(training_data)
        y = np.array(training_labels)
        
        print(f"Melatih classifier SVM kustom dengan {len(training_data)} sampel...")
        # Reset classifier dan latih
        classifier = CustomSVM(C=1.0, max_iter=100)
        classifier.fit(X, y)
        
        # Simpan model ke file pickle
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        # Simpan data pelatihan ke pickle dan CSV
        save_training_data()
        
        print(f"Classifier SVM kustom dilatih dan disimpan ke {model_path}")
        return True
    except Exception as e:
        print(f"Error melatih model: {str(e)}")
        return False

# Fungsi untuk memuat model jika ada
def load_model():
    global classifier, training_data, training_labels
    
    # Coba muat data pelatihan terlebih dahulu
    loaded_data = load_training_data()
    
    # Kemudian coba muat model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            print(f"Model SVM kustom dimuat dari {model_path}")
            return True
        except Exception as e:
            print(f"Error memuat model: {e}")
            
            # Jika kita memiliki data pelatihan, coba melatih ulang
            if loaded_data and len(training_data) > 0:
                print("Mencoba melatih ulang model dengan data yang dimuat...")
                return train_model()
            return False
    else:
        print("Tidak ditemukan model SVM yang sudah dilatih sebelumnya.")
        return False

# Coba memuat model yang ada dan data pelatihan
model_loaded = load_model()

# Cetak instruksi
print("Kontrol Media Gerakan Tangan dengan SVM Kustom:")
print("- Gerakan tangan berbeda akan mengontrol pemutaran media")
print("\nKontrol Mode Pelatihan:")
print("- Tekan 't' untuk toggle pengumpulan data pelatihan")
print("- Tekan '0-6' untuk memilih kelas gerakan yang akan dikumpulkan")
print("- Tekan 'm' untuk melatih model")
print("- Tekan 's' untuk menyimpan data pelatihan tanpa melatih ulang")
print("- Tekan 'c' untuk menghapus semua data pelatihan yang dikumpulkan")
print("- Tombol ESC: Keluar program")

# Loop program utama
try:    
    while running and cap.isOpened():
        ret, frame = cap.read()  # Baca frame dari kamera
        if not ret:
            print("Gagal mendapatkan frame dari kamera")
            break
        
        # Cerminkan frame secara horizontal untuk interaksi yang lebih intuitif
        frame = cv2.flip(frame, 1)  # Flip horizontal agar lebih alami
        
        # Dapatkan dimensi frame
        h, w, c = frame.shape  # Tinggi, lebar, dan kanal warna
        
        # Buat panel informasi di sebelah kanan kamera
        info_panel = np.zeros((h, info_panel_width, 3), dtype=np.uint8)
        info_panel[:, :] = calculation_panel_color  # Warna latar belakang panel
        
        # Gabungkan frame kamera dengan panel informasi
        combined_frame = np.hstack((frame, info_panel))
        combined_w = combined_frame.shape[1]  # Lebar total frame
        
        # Teks status untuk ditampilkan
        status_text = "No gesture detected"  # Status default
        prediction = 0  # Prediksi awal
        
        # Tampilkan judul panel informasi
        cv2.putText(combined_frame, "PANEL INFORMASI & PERHITUNGAN", 
                    (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(combined_frame, (w + 10, 40), (combined_w - 10, 40), (0, 255, 255), 1)
        
        # Variabel untuk menyimpan semua kalkulasi
        all_calculations = {}
        
        # Tampilkan status pelatihan
        if collecting_data:
            cv2.putText(frame, f"Mengumpulkan kelas: {current_collection_class} - {gesture_classes[current_collection_class]}", 
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Hitung sampel untuk kelas saat ini
            current_class_count = sum(1 for label in training_labels if label == current_collection_class)
            cv2.putText(frame, f"Sampel kelas: {current_class_count}", 
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Tampilkan total sampel yang dikumpulkan
        cv2.putText(frame, f"Total sampel: {len(training_labels)}", 
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Deteksi tangan dalam frame
        skin_mask = detect_skin(frame)  # Deteksi kulit
        hand_contour = find_contours(skin_mask)  # Temukan kontur tangan
        
        # Status deteksi di panel informasi
        cv2.putText(combined_frame, "Status Deteksi:", 
                   (w + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        hand_status = "Terdeteksi" if hand_contour is not None else "Tidak Terdeteksi"
        cv2.putText(combined_frame, f"Tangan: {hand_status}", 
                   (w + 20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if hand_contour is not None else (0, 0, 255), 1)
        
        # Tampilkan skin mask di pojok kanan bawah frame (bukan di panel info)
        skin_display = cv2.resize(skin_mask, (w//4, h//4))  # Ubah ukuran mask
        frame[h-h//4:h, w-w//4:w] = cv2.cvtColor(skin_display, cv2.COLOR_GRAY2BGR)  # Tempatkan di kanan bawah
        if hand_contour is not None:
            # Gambar kontur
            cv2.drawContours(frame, [hand_contour], 0, colors["contour"], 2)  # Gambar kontur tangan
            
            # Tambahkan informasi kontur ke panel informasi
            cv2.putText(combined_frame, "Informasi Kontur:", 
                       (w + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Hitung area dan perimeter untuk ditampilkan
            area = cv2.contourArea(hand_contour)
            perimeter = cv2.arcLength(hand_contour, True)
            all_calculations["area"] = area
            all_calculations["perimeter"] = perimeter
            
            # Tampilkan informasi area dan perimeter pada frame utama
            put_text_with_background(frame, f"Area: {int(area)}", (10, 60), font, font_scale_small,
                                    colors["text"], colors["text_bg"], font_thickness_small)    # Tampilkan area dengan latar yang mencolok
            area_text = f"Area: {int(area)}"
            text_size, _ = cv2.getTextSize(area_text, font, font_scale_small, font_thickness_small)
            put_text_with_background(frame, area_text, (10, 60), font, font_scale_small,
                                    colors["text"], colors["text_bg"], font_thickness_small)
            
            # Tambahkan visualisasi normalisasi area
            normalized_area = area / (w * h)
            normalized_area_text = f"Norm Area: {normalized_area:.4f}"
            put_text_with_background(frame, normalized_area_text, (10, 35), font, font_scale_small,
                                    colors["text"], colors["text_bg"], font_thickness_small)
            
            # Tambahkan bar visualisasi untuk normalisasi area
            draw_metric_bar(frame, (10, 45), 100, 8, normalized_area, 0.5, 
                            {"bg": colors["text_bg"], "fill": (0, 200, 200), "text": colors["text"]})

            # Tambahkan visualisasi dan penjelasan proses ekstraksi fitur di panel informasi
            explanation_section_y = h // 2 + 50
            cv2.putText(combined_frame, "PENJELASAN EKSTRAKSI FITUR (BAHASA INDONESIA):", 
                        (w + 10, explanation_section_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)
            cv2.line(combined_frame, (w + 10, explanation_section_y + 10), 
                        (w + 390, explanation_section_y + 10), (255, 255, 100), 1)
            
            # Penjelasan proses ekstraksi fitur dalam Bahasa Indonesia
            explanation_lines = [
                "1. Normalisasi Area: Luas kontur dibagi dengan luas frame",
                "   untuk mendapatkan nilai yang konsisten terlepas dari",
                "   jarak tangan dari kamera.",
                "",
                "2. Convexity Ratio: Perbandingan luas kontur dengan luas",
                "   convex hull. Menunjukkan seberapa 'cekung' bentuk",
                "   tangan. Jari yang terbuka membuat rasio lebih kecil.",
                "",
                "3. Jumlah Jari: Dihitung dari jumlah sudut < 90° pada",
                "   convexity defects yang berada di antara jari.",
                "",
                "4. Jarak Defects: 5 jarak terbesar dari titik defect ke",
                "   pusat massa, ternormalisasi oleh ukuran frame.",
                "",
                "5. Rasio Aspek: Perbandingan lebar dan tinggi kotak",
                "   pembatas, menunjukkan orientasi tangan.",
                "",
                "6. Orientasi: Sudut kemiringan elips yang menyesuaikan",
                "   dengan kontur tangan, dinormalisasi ke rentang 0-1."
            ]
            
            y_offset = explanation_section_y + 30
            for line in explanation_lines:
                cv2.putText(combined_frame, line, 
                            (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset += 20
        
            cv2.putText(combined_frame, f"Area: {int(area)} pixel²", 
                        (w + 20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
            cv2.putText(combined_frame, f"Perimeter: {int(perimeter)} pixel", 
                        (w + 20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
            
            # Temukan convex hull dan defects
            contour, hull, defects = find_convex_hull_and_defects(hand_contour)
            
            if contour is not None:                # Gambar hull
                if hull is not None:
                    hull_points = [contour[h[0]] for h in hull]  # Dapatkan titik-titik hull
                    cv2.drawContours(frame, [np.array(hull_points)], 0, colors["hull"], 3)  # Gambar hull
                    
                    # Hitung dan tampilkan informasi hull
                    hull_contour = np.array(hull_points)
                    hull_area = cv2.contourArea(hull_contour)
                    all_calculations["hull_area"] = hull_area
                    
                    # Gambar visualisasi perbandingan convexity (kontur vs hull) di sudut kiri atas
                    if convexity > 0:
                        visualize_convexity(frame, contour, hull_points, convexity, (10, 160))
                    
                    # Tambahkan overlay untuk menunjukkan area hull vs kontur
                    # Transparan overlay pada hull area
                    hull_mask = np.zeros_like(frame)
                    cv2.drawContours(hull_mask, [hull_contour], 0, (150, 50, 200), -1)
                    # Lapisi overlay pada frame dengan transparansi
                    alpha = 0.3  # Tingkat transparansi (0-1)
                    mask = hull_mask.astype(bool)
                    frame[mask] = cv2.addWeighted(frame, 0.7, hull_mask, 0.3, 0)[mask]
                    
                    # Buat indikator hull-contour pada frame utama
                    # Tampilkan hull area pada frame utama
                    put_text_with_background(frame, f"Hull Area: {int(hull_area)}", (10, 85), font, font_scale_small,
                                           colors["text"], colors["text_bg"], font_thickness_small)
                    
                    cv2.putText(combined_frame, f"Hull Area: {int(hull_area)} pixel²", 
                               (w + 20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
                    
                    # Hitung convexity
                    if hull_area > 0:
                        convexity = area / hull_area
                        all_calculations["convexity"] = convexity
                        
                        # Tampilkan convexity pada frame utama dengan bar indikator
                        convexity_text = f"Convexity: {convexity:.3f}"
                        put_text_with_background(frame, convexity_text, (10, 110), font, font_scale_small,
                                               colors["text"], colors["text_bg"], font_thickness_small)
                        
                        # Tambahkan bar indikator convexity
                        draw_metric_bar(frame, (10, 120), 100, 10, convexity, 1.0, 
                                     {"bg": colors["text_bg"], 
                                      "fill": colors["convexity_good"] if convexity > 0.9 else 
                                             colors["convexity_medium"] if convexity > 0.7 else 
                                             colors["convexity_bad"],
                                      "text": colors["text"]})
                        
                        cv2.putText(combined_frame, f"Convexity: {convexity:.3f}", 
                                   (w + 20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
                
                # Gambar defects dan kumpulkan informasi
                finger_count = 0
                defect_distances = []
                finger_tips = []
                
                # Temukan centroid (pusat massa) kontur
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  # Koordinat x pusat
                    cy = int(M["m01"] / M["m00"])  # Koordinat y pusat
                else:
                    cx, cy = 0, 0
                
                # Tampilkan centroid pada frame
                cv2.circle(frame, (cx, cy), 7, colors["centroid"], -1)
                cv2.circle(frame, (cx, cy), 9, (255, 255, 255), 1)  # Lingkaran putih di sekitar centroid
                
                # Tambahkan label pada centroid
                put_text_with_background(frame, "Centroid", (cx + 10, cy), font, font_scale_small,
                                        colors["text"], colors["text_bg"], font_thickness_small)
                
                all_calculations["centroid"] = (cx, cy)
                
                cv2.putText(combined_frame, f"Pusat Massa: ({cx}, {cy})", 
                           (w + 20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 255), 1)
                
                # Tampilkan informasi deteksi jari
                cv2.putText(combined_frame, "Deteksi Jari:", 
                           (w + 10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Variabel untuk menyimpan sudut antar jari
                angles = []
                
                if defects is not None and len(defects) > 0:
                    cv2.putText(combined_frame, f"Jumlah Defects: {defects.shape[0]}", 
                               (w + 20, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 150), 1)
                          # Tambahkan jumlah defects pada frame utama
                    put_text_with_background(frame, f"Defects: {defects.shape[0]}", (10, 135), font, font_scale_small,
                                           colors["text"], colors["text_bg"], font_thickness_small)
                    
                    # Buat visualisasi konveks di kanan atas
                    convex_viz_height = 130
                    convex_viz_width = 200
                    convex_viz_x = w - convex_viz_width - 10
                    convex_viz_y = 60
                    
                    # Buat area visualisasi dengan latar belakang semi-transparan
                    cv2.rectangle(frame, (convex_viz_x, convex_viz_y), 
                                 (convex_viz_x + convex_viz_width, convex_viz_y + convex_viz_height), 
                                 (0, 0, 0, 128), -1)
                    
                    # Judul visualisasi
                    cv2.putText(frame, "Defect Analysis", (convex_viz_x + 5, convex_viz_y + 20), 
                               font, 0.5, (255, 255, 255), 1)
                    
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]  # Start, end, far point, dan distance
                        start = tuple(contour[s][0])  # Titik awal
                        end = tuple(contour[e][0])  # Titik akhir
                        far = tuple(contour[f][0])  # Titik terjauh
                          # Gambar garis dan titik defect pada frame
                        cv2.line(frame, start, end, colors["defect_line"], 2)  # Gambar garis
                        cv2.circle(frame, far, 5, colors["defect_point"], -1)  # Gambar lingkaran
                        
                        # Tambahkan label pada titik far
                        put_text_with_background(frame, f"D{i+1}", (far[0] + 5, far[1] - 5), font, 
                                               font_scale_small, colors["text"], colors["text_bg"], font_thickness_small)
                        
                        # Gambar dan label start dan end sebagai kemungkinan ujung jari
                        cv2.circle(frame, start, 6, colors["finger_tip"], -1)
                        cv2.circle(frame, end, 6, colors["finger_tip"], -1)
                        
                        # Tambahkan start dan end ke daftar kemungkinan ujung jari
                        finger_tips.append(start)
                        finger_tips.append(end)
                        
                        # Hitung jarak defect ke pusat
                        dist_to_center = math.sqrt((far[0] - cx)**2 + (far[1] - cy)**2)
                        normalized_dist = dist_to_center / math.sqrt(w**2 + h**2)
                        defect_distances.append(normalized_dist)
                        
                        # Tambahkan visualisasi jarak ke panel defect
                        if i < 5:  # Batasi untuk 5 defects saja untuk visualisasi
                            # Gambar bar untuk jarak
                            y_pos = convex_viz_y + 30 + i * 20
                            draw_metric_bar(frame, 
                                          (convex_viz_x + 10, y_pos), 
                                          120, 10, 
                                          normalized_dist, 
                                          0.5,  # Nilai maksimum untuk normalisasi
                                          {"bg": (50, 50, 50), 
                                           "fill": (0, 150 + i*20, 255 - i*20), 
                                           "text": (255, 255, 255)},
                                          f"Dist {i+1}")
                        
                        # Hitung sudut untuk deteksi jari
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                          # Hitung sudut menggunakan hukum kosinus
                        if a*b > 0:
                            angle_rad = math.acos((b**2 + c**2 - a**2) / (2*b*c))
                            angle_deg = angle_rad * 180 / math.pi
                            angles.append(angle_deg)
                            
                            # Tampilkan sudut pada frame utama
                            put_text_with_background(frame, f"{angle_deg:.1f}°", 
                                                   (far[0] - 40, far[1] + 15), font, font_scale_small,
                                                   colors["text"], colors["text_bg"], font_thickness_small)
                            
                            # Visualisasi sudut dengan garis
                            # Gambar garis dari titik far ke midpoint antara start dan end
                            mid_x = (start[0] + end[0]) // 2
                            mid_y = (start[1] + end[1]) // 2
                            
                            # Gambar garis dari far ke tengah busur
                            cv2.line(frame, far, (mid_x, mid_y), (0, 255, 255), 1, cv2.LINE_AA)
                            
                            # Warna berdasarkan sudut (hijau jika <90°, kuning jika <120°, merah jika >120°)
                            angle_color = (0, 255, 0) if angle_deg < 90 else \
                                         (0, 255, 255) if angle_deg < 120 else \
                                         (0, 0, 255)
                            
                            # Menggambar busur untuk visualisasi sudut (simplifikasi)
                            radius = 20
                            # Hitung vektor unit dari far ke start dan end
                            vec_s = ((start[0] - far[0])/b, (start[1] - far[1])/b)
                            vec_e = ((end[0] - far[0])/c, (end[1] - far[1])/c)
                            
                            # Gambar busur menggunakan vektor unit
                            start_angle = math.atan2(vec_s[1], vec_s[0])
                            end_angle = math.atan2(vec_e[1], vec_e[0])
                            
                            # Koreksi sudut jika perlu
                            if start_angle > end_angle:
                                start_angle, end_angle = end_angle, start_angle
                            if end_angle - start_angle > math.pi:
                                start_angle += 2*math.pi
                                start_angle, end_angle = end_angle, start_angle
                            
                            # Gambar busur untuk menunjukkan sudut
                            cv2.ellipse(frame, far, (radius, radius), 0, 
                                       start_angle * 180 / math.pi, 
                                       end_angle * 180 / math.pi, 
                                       angle_color, 2)
                            
                            # Jika sudut kurang dari 90 derajat, mungkin ini adalah ruang antar jari
                            if angle_rad <= math.pi/2:
                                finger_count += 1
                
                # Hilangkan duplikat dari ujung jari dengan memeriksa jarak
                unique_finger_tips = []
                if finger_tips:
                    unique_finger_tips.append(finger_tips[0])
                    for point in finger_tips[1:]:
                        add_point = True
                        for unique_point in unique_finger_tips:
                            distance = math.sqrt((point[0] - unique_point[0])**2 + (point[1] - unique_point[1])**2)
                            if distance < 20:  # Jika terlalu dekat dengan titik yang sudah ada
                                add_point = False
                                break
                        if add_point:
                            unique_finger_tips.append(point)
                
                # Tampilkan ujung jari yang terdeteksi
                for i, tip in enumerate(unique_finger_tips):
                    if i < 5:  # Batasi hingga 5 jari
                        cv2.circle(frame, tip, 8, colors["finger_tip"], -1)
                        cv2.circle(frame, tip, 10, (255, 255, 255), 1)  # Lingkaran putih di sekitar ujung jari
                        put_text_with_background(frame, f"F{i+1}", (tip[0] + 10, tip[1]), font, font_scale_small,
                                               colors["text"], colors["text_bg"], font_thickness_small)
                
                # Tampilkan jumlah jari terdeteksi
                all_calculations["finger_count"] = finger_count
                
                # Tampilkan jumlah jari pada frame utama dengan latar yang mencolok
                finger_text = f"Jari: {finger_count}"
                text_size, _ = cv2.getTextSize(finger_text, font, font_scale_medium, font_thickness_medium)
                put_text_with_background(frame, finger_text, (w - text_size[0] - 10, 60), font, font_scale_medium,
                                       colors["text"], (0, 100, 0), font_thickness_medium)
                
                cv2.putText(combined_frame, f"Jumlah Jari: {finger_count}", 
                           (w + 20, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 150), 1)
                
                # Tampilkan sudut-sudut antar jari (hingga 5)
                if angles:
                    angles.sort()
                    for i in range(min(5, len(angles))):
                        cv2.putText(combined_frame, f"Sudut {i+1}: {angles[i]:.1f}°", 
                                   (w + 20, 335 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 150), 1)
                
                # Tampilkan informasi bounding rectangle
                x, y, width, height = cv2.boundingRect(contour)
                aspect_ratio = float(width) / height if height > 0 else 0
                all_calculations["aspect_ratio"] = aspect_ratio
                
                # Gambar bounding rectangle pada frame
                cv2.rectangle(frame, (x, y), (x+width, y+height), colors["bounding_box"], 2)
                
                # Tampilkan dimensi bounding box pada frame utama
                put_text_with_background(frame, f"W:{width} H:{height}", (x, y-10), font, font_scale_small,
                                       colors["text"], colors["text_bg"], font_thickness_small)
                
                # Tampilkan aspect ratio pada frame utama
                put_text_with_background(frame, f"Ratio: {aspect_ratio:.2f}", (x, y+height+15), font, font_scale_small,
                                       colors["text"], colors["text_bg"], font_thickness_small)
                
                cv2.putText(combined_frame, "Bounding Rectangle:", 
                           (w + 10, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(combined_frame, f"Width: {width}, Height: {height}", 
                           (w + 20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                cv2.putText(combined_frame, f"Aspect Ratio: {aspect_ratio:.3f}", 
                           (w + 20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                
                # Tampilkan informasi orientasi elips jika kontur cukup besar
                if len(contour) >= 5:
                    (x_el, y_el), (MA, ma), angle = cv2.fitEllipse(contour)
                    # Gambar elips pada frame
                    cv2.ellipse(frame, ((int(x_el), int(y_el)), (int(MA), int(ma)), int(angle)), colors["ellipse"], 2)
                    
                    # Tampilkan data elips pada frame utama
                    put_text_with_background(frame, f"Elips: {angle:.1f}°", (int(x_el) - 40, int(y_el) - 15), font, 
                                           font_scale_small, colors["text"], colors["text_bg"], font_thickness_small)
                    
                    all_calculations["ellipse_angle"] = angle
                    all_calculations["ellipse_major_axis"] = MA
                    all_calculations["ellipse_minor_axis"] = ma
                    
                    cv2.putText(combined_frame, "Orientasi Elips:", 
                               (w + 10, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(combined_frame, f"Angle: {angle:.1f}°", 
                               (w + 20, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
                    cv2.putText(combined_frame, f"Major Axis: {MA:.1f}, Minor Axis: {ma:.1f}", 
                               (w + 20, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
                  # Ekstrak fitur dari tangan
                features = extract_features(contour, defects, frame.shape)  # Ekstrak fitur
                if features:
                    # Tampilkan tabel fitur terukur di kanan bawah frame
                    feature_table_height = 160
                    feature_table_width = 200
                    feature_table_x = w - feature_table_width - 10
                    feature_table_y = h - feature_table_height - 10
                    
                    # Latar belakang semi-transparan untuk tabel
                    overlay = frame.copy()
                    cv2.rectangle(overlay, 
                                 (feature_table_x, feature_table_y), 
                                 (feature_table_x + feature_table_width, feature_table_y + feature_table_height), 
                                 (30, 30, 30), -1)
                    # Terapkan transparansi
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Judul tabel
                    cv2.putText(frame, "FEATURE MEASUREMENTS", 
                               (feature_table_x + 10, feature_table_y + 20), 
                               font, 0.5, (255, 255, 255), 1)
                    cv2.line(frame, 
                            (feature_table_x + 10, feature_table_y + 25), 
                            (feature_table_x + feature_table_width - 10, feature_table_y + 25), 
                            (255, 255, 255), 1)
                    
                    # Nama dan nilai fitur
                    feature_names = [
                        "Normalized Area",
                        "Convexity Ratio",
                        "Finger Count",
                        "Max Defect Dist",
                        "Aspect Ratio",
                        "Orientation"
                    ]
                    
                    feature_values = [
                        features[0],
                        features[1],
                        features[2],
                        max(features[3:8]) if len(features) > 3 else 0,
                        features[8] if len(features) > 8 else 0,
                        features[9] if len(features) > 9 else 0
                    ]
                    
                    # Rangkai untuk visualisasi
                    for i, (name, value) in enumerate(zip(feature_names, feature_values)):
                        y_pos = feature_table_y + 45 + i * 20
                        
                        # Nama fitur
                        cv2.putText(frame, name, 
                                   (feature_table_x + 15, y_pos), 
                                   font, 0.4, (200, 200, 200), 1)
                        
                        # Nilai fitur dengan bar visual
                        max_vals = [0.5, 1.0, 5.0, 0.5, 3.0, 1.0]  # Nilai max untuk setiap fitur
                        
                        # Cari warna berdasarkan nilai
                        color_val = min(255, int(value / max_vals[i] * 255)) if max_vals[i] > 0 else 0
                        bar_color = (0, color_val, 255-color_val)
                        
                        # Gambar bar untuk nilai
                        bar_width = min(int(value / max_vals[i] * 70), 70) if max_vals[i] > 0 else 0
                        cv2.rectangle(frame, 
                                     (feature_table_x + 125, y_pos - 10), 
                                     (feature_table_x + 125 + bar_width, y_pos - 4), 
                                     bar_color, -1)
                        
                        # Nilai numerik
                        if isinstance(value, int):
                            value_text = f"{value}"
                        else:
                            value_text = f"{value:.3f}"
                        cv2.putText(frame, value_text, 
                                   (feature_table_x + 125 + 75, y_pos), 
                                   font, 0.4, (255, 255, 255), 1)
                    
                    # Jika mengumpulkan data, tambahkan ke set pelatihan
                    if collecting_data:
                        # Tambah sampel ke data pelatihan
                        training_data.append(features)  # Tambahkan fitur
                        training_labels.append(current_collection_class)  # Tambahkan label
                        
                        # Hitung jumlah kelas saat ini
                        current_class_count = sum(1 for label in training_labels if label == current_collection_class)
                        
                        # Ambil screenshot jika cooldown telah berlalu
                        if sample_cooldown <= 0:
                            save_screenshot(frame, current_collection_class, current_class_count)  # Simpan screenshot
                            sample_cooldown = 10  # Set cooldown untuk 10 frame
                        else:
                            sample_cooldown -= 1  # Kurangi cooldown
                        
                        # Batasi laju pengumpulan
                        time.sleep(0.1)  # Jeda 0.1 detik
                    
                    # Tampilkan semua fitur di panel informasi
                    cv2.putText(combined_frame, "Fitur Terektraksi:", 
                               (w + 10, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Nama-nama fitur untuk ditampilkan
                    feature_names = [
                        "Normalized Area",
                        "Convexity Ratio",
                        "Finger Count",
                        "Defect Distance 1",
                        "Defect Distance 2",
                        "Defect Distance 3",
                        "Defect Distance 4",
                        "Defect Distance 5",
                        "Aspect Ratio",
                        "Orientation"
                    ]
                    
                    # Tampilkan nilai fitur
                    for i in range(min(len(features), len(feature_names))):
                        cv2.putText(combined_frame, f"{feature_names[i]}: {features[i]:.4f}", 
                                   (w + 20, 615 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 255), 1)
                    
                    # Jika model dimuat, prediksi gerakan
                    if model_loaded and classifier.trained:
                        try:
                            # Reshape fitur untuk prediksi
                            features_array = np.array([features])  # Konversi ke array numpy
                            
                            # Prediksi gerakan
                            prediction = classifier.predict(features_array)[0]  # Prediksi kelas
                            
                            # Dapatkan probabilitas untuk setiap kelas
                            probabilities = classifier.predict_proba(features_array)[0]  # Probabilitas
                              # Hanya terima prediksi dengan keyakinan di atas 0.4 (40%)
                            max_prob = np.max(probabilities)  # Probabilitas maksimum
                            if max_prob > 0.4:
                                # Dapatkan kelas dengan probabilitas tertinggi
                                prediction = np.argmax(probabilities)  # Kelas terprediksi
                                # Konversi prediksi ke nama gerakan
                                status_text = gesture_classes.get(prediction, "Unknown gesture")  # Teks status
                                
                                # Tampilkan status dengan latar penuh di pojok kiri atas frame
                                text_size, _ = cv2.getTextSize(status_text, font, font_scale_large, font_thickness_large)
                                
                                # Buat latar belakang dengan warna yang berbeda sesuai gerakan
                                bg_color = (0, 100, 0)  # Default: hijau gelap
                                if prediction == 1:  # Play/Pause
                                    bg_color = (0, 120, 0)  # Hijau
                                elif prediction == 2:  # Stop
                                    bg_color = (0, 0, 120)  # Merah
                                elif prediction == 3:  # Next Track
                                    bg_color = (120, 0, 0)  # Biru
                                elif prediction == 4:  # Previous Track
                                    bg_color = (0, 120, 120)  # Kuning
                                elif prediction == 5:  # Volume Up
                                    bg_color = (120, 0, 120)  # Ungu
                                elif prediction == 6:  # Volume Down
                                    bg_color = (120, 120, 0)  # Cyan
                                
                                # Gambar latar belakang yang lebih lebar
                                cv2.rectangle(frame, (0, 0), (text_size[0] + 40, 40), bg_color, -1)
                                
                                # Gambar indikator keyakinan
                                confidence_width = int(max_prob * (text_size[0] + 40))
                                cv2.rectangle(frame, (0, 40), (confidence_width, 45), (255, 255, 255), -1)
                                
                                # Gambar teks dengan warna putih
                                cv2.putText(frame, status_text, (10, 30), font, font_scale_large, (255, 255, 255), font_thickness_large)
                            else:
                                # Jika tidak ada kelas dengan keyakinan di atas 40%, jangan kenali gerakan
                                prediction = 0  # Reset prediksi
                                status_text = "No gesture detected (low confidence)"  # Teks status
                                
                                # Tampilkan status dengan latar transparan
                                cv2.rectangle(frame, (0, 0), (400, 40), (50, 50, 50), -1)
                                cv2.putText(frame, status_text, (10, 30), font, font_scale_large, (150, 150, 150), font_thickness_large)
                            
                            # Tampilkan judul hasil klasifikasi
                            y_prob_start = max(615 + len(feature_names)*20, h//2)
                            cv2.putText(combined_frame, "Hasil Klasifikasi SVM:", 
                                       (w + 10, y_prob_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                            cv2.line(combined_frame, (w + 10, y_prob_start + 10), (w + 290, y_prob_start + 10), (0, 255, 255), 1)
                              # Tampilkan probabilitas prediksi dalam bentuk bar chart
                            max_bar_width = 150
                            for i, prob in enumerate(probabilities):
                                if i < len(gesture_classes):
                                    # Teks kelas
                                    class_text = f"{gesture_classes.get(i, 'Unknown')}"
                                    cv2.putText(combined_frame, class_text, 
                                               (w + 20, y_prob_start + 35 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                               (255, 255, 255), 1)
                                    
                                    # Bar probabilitas
                                    bar_width = int(prob * max_bar_width)
                                    bar_color = (0, 0, 255) if prob > 0.4 else (0, 165, 255)
                                    
                                    # Background bar (gray)
                                    cv2.rectangle(combined_frame, 
                                                 (w + 170, y_prob_start + 25 + i*25), 
                                                 (w + 170 + max_bar_width, y_prob_start + 45 + i*25), 
                                                 (70, 70, 70), -1)
                                    
                                    # Foreground bar (colored by probability)
                                    if bar_width > 0:
                                        cv2.rectangle(combined_frame, 
                                                     (w + 170, y_prob_start + 25 + i*25), 
                                                     (w + 170 + bar_width, y_prob_start + 45 + i*25), 
                                                     bar_color, -1)
                                    
                                    # Probabilitas sebagai persentase
                                    prob_text = f"{prob:.1%}"
                                    text_x = w + 175 + bar_width if bar_width > 0 else w + 175
                                    cv2.putText(combined_frame, prob_text, 
                                               (text_x, y_prob_start + 40 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.5, (255, 255, 255), 1)
                            
                            # Tampilkan tabel fitur di sudut kanan bawah frame kamera
                            if max_prob > 0.3:  # Tampilkan fitur jika ada probabilitas yang signifikan
                                # Buat area untuk tabel fitur (kotak semi-transparan)
                                feature_table_height = 160
                                feature_table_width = 200
                                feature_table_x = w - feature_table_width - 10
                                feature_table_y = h - feature_table_height - 10
                                
                                # Gambar latar belakang tabel
                                overlay = frame.copy()
                                cv2.rectangle(overlay, 
                                             (feature_table_x, feature_table_y), 
                                             (feature_table_x + feature_table_width, feature_table_y + feature_table_height), 
                                             (30, 30, 30), -1)
                                
                                # Terapkan transparansi
                                alpha = 0.7
                                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                                
                                # Judul tabel
                                cv2.putText(frame, "FITUR UTAMA", 
                                           (feature_table_x + 10, feature_table_y + 20), 
                                           font, font_scale_small, (0, 255, 255), font_thickness_small)
                                
                                # Tampilkan fitur-fitur utama dengan nilai yang sudah dibulatkan
                                feature_names_short = ["Area", "Convexity", "Fingers", "Aspect", "Angle"]
                                feature_values = [
                                    f"{features[0]:.4f}",    # Normalized area
                                    f"{features[1]:.4f}",    # Convexity
                                    f"{int(features[2])}",   # Finger count
                                    f"{features[8]:.4f}",    # Aspect ratio
                                    f"{features[9]:.4f}"     # Orientation
                                ]
                                
                                # Gambar garis header
                                cv2.line(frame, 
                                        (feature_table_x + 5, feature_table_y + 25), 
                                        (feature_table_x + feature_table_width - 5, feature_table_y + 25), 
                                        (0, 255, 255), 1)
                                
                                # Tampilkan fitur dalam format tabel
                                for i, (name, value) in enumerate(zip(feature_names_short, feature_values)):
                                    y_pos = feature_table_y + 50 + i * 20
                                    # Nama fitur
                                    cv2.putText(frame, name, 
                                               (feature_table_x + 10, y_pos), 
                                               font, font_scale_small, (200, 200, 200), font_thickness_small)
                                    
                                    # Nilai fitur
                                    cv2.putText(frame, value, 
                                               (feature_table_x + 100, y_pos), 
                                               font, font_scale_small, (255, 255, 255), font_thickness_small)
                            
                            # Jalankan aksi berdasarkan gerakan yang diprediksi
                            if max_prob > 0.4:
                                current_time = time.time()  # Waktu saat ini
                                
                                # Variabel kontrol volume
                                volume_adjustment_interval = 0.3  # Interval penyesuaian volume
                                
                                if prediction == 1 :  # Play/Pause
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('playpause')  # Tekan tombol play/pause
                                        last_action_time = current_time  # Update waktu aksi
                                        print(f"Aksi: Play/Pause (keyakinan: {max_prob:.2f})")
                                elif prediction == 2 :  # Stop
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('stop')  # Tekan tombol stop
                                        last_action_time = current_time  # Update waktu aksi
                                        print(f"Aksi: Stop (keyakinan: {max_prob:.2f})")
                                elif prediction == 3 :  # Next Track
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('nexttrack')  # Tekan tombol lagu berikutnya
                                        last_action_time = current_time  # Update waktu aksi
                                        print(f"Aksi: Next Track (keyakinan: {max_prob:.2f})")
                                elif prediction == 4 :  # Previous Track
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('prevtrack')  # Tekan tombol lagu sebelumnya
                                        last_action_time = current_time  # Update waktu aksi
                                        print(f"Aksi: Previous Track (keyakinan: {max_prob:.2f})")
                                elif prediction == 5:  # Volume Up
                                    if current_time - last_action_time > volume_adjustment_interval:
                                        pyautogui.press('volumeup')  # Tekan tombol volume up
                                        last_action_time = current_time  # Update waktu aksi
                                        print(f"Aksi: Volume Up (keyakinan: {max_prob:.2f})")
                                elif prediction == 6:  # Volume Down
                                    if current_time - last_action_time > volume_adjustment_interval:
                                        pyautogui.press('volumedown')  # Tekan tombol volume down
                                        last_action_time = current_time  # Update waktu aksi
                                        print(f"Aksi: Volume Down (keyakinan: {max_prob:.2f})")
                                
                                # Update gerakan sebelumnya hanya untuk prediksi kepercayaan tinggi
                                previous_gesture = status_text  # Simpan gerakan saat ini
                        except Exception as e:
                            print(f"Error selama prediksi: {e}")
                            cv2.putText(combined_frame, f"Error: {str(e)}", 
                                       (w + 20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Tampilkan mode pengumpulan data jika aktif
        if collecting_data:
            collection_text = f"Pengumpulan Data: Kelas {current_collection_class} - {gesture_classes[current_collection_class]}"
            # Latar belakang untuk teks pengumpulan data
            cv2.rectangle(frame, (0, h-60), (w, h-25), (0, 0, 150), -1)
            cv2.putText(frame, collection_text, (10, h-35), font, font_scale_medium, (255, 255, 255), font_thickness_medium)
            
            # Hitung sampel untuk kelas saat ini
            current_class_count = sum(1 for label in training_labels if label == current_collection_class)
            cv2.putText(frame, f"Sampel kelas: {current_class_count}", 
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Tampilkan judul untuk skin mask
        cv2.putText(frame, "Mask Kulit", (w - w//4, h - h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Tampilkan frame gabungan
        cv2.imshow("Kontrol Media Gerakan Tangan dengan SVM Kustom", combined_frame)  # Tampilkan frame
        
        # Periksa penekanan tombol
        key = cv2.waitKey(1) & 0xFF  # Tunggu tombol
        
        # Tombol ESC untuk keluar
        if key == 27:
            running = False  # Hentikan program
        
        # Tombol 't' untuk toggle pengumpulan data pelatihan
        elif key == ord('t'):
            collecting_data = not collecting_data  # Toggle status pengumpulan
            print(f"Pengumpulan data {'dimulai' if collecting_data else 'dihentikan'}")
            # Reset sample cooldown saat toggling
            sample_cooldown = 0
        
        # Tombol 'm' untuk melatih model
        elif key == ord('m'):
            print("Melatih model SVM...")
            model_loaded = train_model()  # Latih model
        
        # Tombol 's' untuk menyimpan data tanpa melatih ulang
        elif key == ord('s'):
            print("Menyimpan data pelatihan...")
            save_training_data()  # Simpan data
            save_training_data_to_csv()  # Simpan ke CSV
            print("Data pelatihan disimpan")
        
        # Tombol 'c' untuk menghapus semua data pelatihan yang dikumpulkan
        elif key == ord('c'):
            print("Menghapus semua data pelatihan...")
            training_data = []  # Kosongkan data
            training_labels = []  # Kosongkan label
            print("Data pelatihan dihapus")
        
        # Tombol angka untuk memilih kelas gerakan untuk pelatihan
        elif key >= ord('0') and key <= ord('6'):
            current_collection_class = key - ord('0')  # Tentukan kelas
            print(f"Kelas yang dipilih: {current_collection_class} - {gesture_classes[current_collection_class]}")
        
        # Tombol 'q' sebagai opsi keluar alternatif
        elif key == ord('q'):
            running = False  # Hentikan program
            
except Exception as e:
    print(f"Error tak terduga: {str(e)}")
    import traceback
    traceback.print_exc()  # Cetak stack trace

# Sebelum keluar, simpan data pelatihan jika tidak kosong
if len(training_data) > 0:
    print("Menyimpan data sebelum keluar...")
    save_training_data()  # Simpan data

# Pembersihan
cap.release()  # Lepaskan kamera
cv2.destroyAllWindows()  # Tutup semua jendela
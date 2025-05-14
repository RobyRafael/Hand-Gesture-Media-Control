import cv2  # Library untuk pemrosesan gambar
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
cap = cv2.VideoCapture(1)  # Mulai webcam dengan indeks 1 (ganti ke 0 jika tidak berfungsi)
running = True  # Flag untuk menjalankan program

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
        
        # Teks status untuk ditampilkan
        status_text = "No gesture detected"  # Status default
        prediction = 0  # Prediksi awal
        
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
        
        # Tampilkan skin mask di pojok kanan bawah
        skin_display = cv2.resize(skin_mask, (w//4, h//4))  # Ubah ukuran mask
        frame[h-h//4:h, w-w//4:w] = cv2.cvtColor(skin_display, cv2.COLOR_GRAY2BGR)  # Tempatkan di kanan bawah
        
        if hand_contour is not None:
            # Gambar kontur
            cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)  # Gambar kontur tangan
            
            # Temukan convex hull dan defects
            contour, hull, defects = find_convex_hull_and_defects(hand_contour)
            
            if contour is not None:
                # Gambar hull
                if hull is not None:
                    hull_points = [contour[h[0]] for h in hull]  # Dapatkan titik-titik hull
                    cv2.drawContours(frame, [np.array(hull_points)], 0, (0, 0, 255), 3)  # Gambar hull
                
                # Gambar defects
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]  # Start, end, far point, dan distance
                        start = tuple(contour[s][0])  # Titik awal
                        end = tuple(contour[e][0])  # Titik akhir
                        far = tuple(contour[f][0])  # Titik terjauh
                        
                        # Gambar garis dari start ke end
                        cv2.line(frame, start, end, [0, 255, 0], 2)  # Gambar garis
                        
                        # Gambar lingkaran di titik terjauh
                        cv2.circle(frame, far, 5, [0, 0, 255], -1)  # Gambar lingkaran
                
                # Ekstrak fitur dari tangan
                features = extract_features(contour, defects, frame.shape)  # Ekstrak fitur
                
                if features:
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
                            else:
                                # Jika tidak ada kelas dengan keyakinan di atas 40%, jangan kenali gerakan
                                prediction = 0  # Reset prediksi
                                status_text = "No gesture detected (low confidence)"  # Teks status
                            
                            # Tampilkan probabilitas prediksi
                            for i, prob in enumerate(probabilities):
                                if i < len(gesture_classes) and prob > 0.05:  # Hanya tampilkan probabilitas signifikan
                                    prob_text = f"{gesture_classes.get(i, 'Unknown')}: {prob:.2f}"  # Teks probabilitas
                                    # Sorot prediksi terpilih dengan warna berbeda
                                    text_color = (0, 0, 255) if prob > 0.4 else (255, 0, 0)  # Warna teks
                                    cv2.putText(frame, prob_text, 
                                                (w - 300, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                            
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
                    
                    # Tampilkan fitur yang diekstrak
                    for i in range(min(6, len(features))):
                        cv2.putText(frame, f"F{i}: {features[i]:.3f}", 
                                    (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Tampilkan teks status
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Teks status utama
        cv2.putText(frame, "Klasifikasi SVM", (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Label SVM
        
        # Tampilkan judul untuk skin mask
        cv2.putText(frame, "Mask Kulit", (w - w//4, h - h//4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Tampilkan frame
        cv2.imshow("Kontrol Media Gerakan Tangan dengan SVM Kustom", frame)  # Tampilkan frame
        
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
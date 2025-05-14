import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create data visualization directory in the same folder as the script
visual_dir = os.path.join(script_dir, 'custom_data_visualizations')
if not os.path.exists(visual_dir):
    os.makedirs(visual_dir)
    print(f"Created directory: {visual_dir}")

# Function to create complete path for saving figures
def get_save_path(filename):
    return os.path.join(visual_dir, filename)

# Membaca data 
csv_path = os.path.join(script_dir, 'custom_svm_training_data.csv')
df = pd.read_csv(csv_path)

# Mengubah gesture_class ke tipe kategorikal untuk visualisasi yang lebih baik
gesture_names = {
    0: "No gesture",
    1: "Play/Pause",
    2: "Stop",
    3: "Next Track",
    4: "Previous Track",
    5: "Volume Up",
    6: "Volume Down"
}
df['gesture_name'] = df['gesture_class'].map(gesture_names)

# 1. Karakteristik Data

# 1.1 Informasi dasar dataset
print("Informasi Dataset:")
print(f"Jumlah sampel: {df.shape[0]}")
print(f"Jumlah fitur: {df.shape[1] - 2}")  # -2 untuk kolom kelas dan nama kelas
print(f"Distribusi kelas:")
print(df['gesture_class'].value_counts())

# 1.2 Visualisasi distribusi kelas
plt.figure(figsize=(12, 7))
ax = sns.countplot(x='gesture_class', data=df, palette='viridis')
plt.title('Distribusi Kelas Gerakan Tangan', fontsize=14)
plt.xlabel('Kelas Gerakan', fontsize=12)
plt.ylabel('Jumlah Sampel', fontsize=12)

# Tambahkan label pada setiap bar
for i, p in enumerate(ax.patches):
    ax.annotate(f"{p.get_height()}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom', fontsize=10)

# Tambahkan nama gerakan di bawah angka
plt.xticks(range(len(gesture_names)), [f"{i}\n{name}" for i, name in gesture_names.items()], 
           rotation=45, ha='right')
plt.tight_layout()
plt.savefig(get_save_path('class_distribution.png'))
plt.close()

# 1.3 Pair plot untuk fitur penting
# Pilih subset fitur untuk pair plot (terlalu banyak fitur akan membuat plot sulit dibaca)
selected_features = ['normalized_area', 'convexity', 'finger_count', 'aspect_ratio', 'orientation']
pair_plot_df = df[selected_features + ['gesture_class']].sample(min(500, len(df)))  # Sample untuk performa

plt.figure(figsize=(15, 12))
sns.pairplot(pair_plot_df, hue='gesture_class', palette='viridis', diag_kind='kde', 
             plot_kws={'alpha': 0.6, 's': 30})
plt.suptitle('Pair Plot Fitur Utama', y=1.02, fontsize=16)
plt.savefig(get_save_path('feature_pairplot.png'))
plt.close()

# 1.4 Heatmap korelasi
plt.figure(figsize=(14, 12))
feature_cols = df.columns[:-2]  # Semua kolom kecuali kelas dan nama kelas
correlation = df[feature_cols].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', 
            linewidths=0.5, vmin=-1, vmax=1, fmt='.2f', annot_kws={"size": 8})
plt.title('Matriks Korelasi Fitur', fontsize=14)
plt.tight_layout()
plt.savefig(get_save_path('correlation_heatmap.png'))
plt.close()

# 1.5 Box plots untuk fitur yang berbeda berdasarkan kelas
# Fitur geometri tangan
plt.figure(figsize=(15, 8))
sns.boxplot(x='gesture_name', y='normalized_area', data=df, palette='viridis')
plt.title('Area Tangan Ternormalisasi berdasarkan Kelas Gerakan', fontsize=14)
plt.xlabel('Gerakan Tangan', fontsize=12)
plt.ylabel('Area Ternormalisasi', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(get_save_path('boxplot_area.png'))
plt.close()

# Fitur jari
plt.figure(figsize=(15, 8))
sns.boxplot(x='gesture_name', y='finger_count', data=df, palette='viridis')
plt.title('Jumlah Jari Terdeteksi berdasarkan Kelas Gerakan', fontsize=14)
plt.xlabel('Gerakan Tangan', fontsize=12)
plt.ylabel('Jumlah Jari', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(get_save_path('boxplot_finger_count.png'))
plt.close()

# Rasio aspek
plt.figure(figsize=(15, 8))
sns.boxplot(x='gesture_name', y='aspect_ratio', data=df, palette='viridis')
plt.title('Rasio Aspek Tangan berdasarkan Kelas Gerakan', fontsize=14)
plt.xlabel('Gerakan Tangan', fontsize=12)
plt.ylabel('Rasio Aspek', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(get_save_path('boxplot_aspect_ratio.png'))
plt.close()

# 1.6 Visualisasi distribusi fitur untuk setiap kelas
# Convexity - violin plot
plt.figure(figsize=(15, 8))
sns.violinplot(x='gesture_name', y='convexity', data=df, palette='viridis', inner='quartile')
plt.title('Distribusi Convexity berdasarkan Kelas Gerakan', fontsize=14)
plt.xlabel('Gerakan Tangan', fontsize=12)
plt.ylabel('Convexity', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(get_save_path('violin_convexity.png'))
plt.close()

# Orientasi - violin plot
plt.figure(figsize=(15, 8))
sns.violinplot(x='gesture_name', y='orientation', data=df, palette='viridis', inner='quartile')
plt.title('Distribusi Orientasi berdasarkan Kelas Gerakan', fontsize=14)
plt.xlabel('Gerakan Tangan', fontsize=12)
plt.ylabel('Orientasi', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(get_save_path('violin_orientation.png'))
plt.close()

# 1.7 Visualisasi PCA
# Standarisasi data
X = df.drop(['gesture_class', 'gesture_name'], axis=1)
y = df['gesture_class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA untuk visualisasi 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA dengan warna berdasarkan kelas dan bentuk yang berbeda
plt.figure(figsize=(12, 10))
markers = ['o', 's', '^', 'D', 'v', 'p', '*']
colors = plt.cm.viridis(np.linspace(0, 1, len(gesture_names)))

for i, (gesture_id, gesture_name) in enumerate(gesture_names.items()):
    # Filter untuk kelas saat ini
    mask = y == gesture_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=[colors[i]], 
                marker=markers[i % len(markers)], 
                s=70, alpha=0.7, edgecolors='w',
                label=f"{gesture_id}: {gesture_name}")

plt.title('Visualisasi PCA: Proyeksi 2D Data Gerakan Tangan', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Kelas Gerakan', fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.savefig(get_save_path('pca_visualization.png'))
plt.close()

# PCA untuk visualisasi 3D (jika memungkinkan)
if len(X.columns) >= 3:
    pca3d = PCA(n_components=3)
    X_pca3d = pca3d.fit_transform(X_scaled)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (gesture_id, gesture_name) in enumerate(gesture_names.items()):
        # Filter untuk kelas saat ini
        mask = y == gesture_id
        ax.scatter(X_pca3d[mask, 0], X_pca3d[mask, 1], X_pca3d[mask, 2],
                   c=[colors[i]], 
                   marker=markers[i % len(markers)], 
                   s=70, alpha=0.7,
                   label=f"{gesture_id}: {gesture_name}")
    
    ax.set_title('Visualisasi PCA: Proyeksi 3D Data Gerakan Tangan', fontsize=14)
    ax.set_xlabel(f'PC1 ({pca3d.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca3d.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca3d.explained_variance_ratio_[2]:.2%})', fontsize=12)
    plt.legend(title='Kelas Gerakan', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(get_save_path('pca_3d_visualization.png'))
    plt.close()

# 2. Visualisasi SVM

# Fungsi untuk plot decision boundary pada PCA 2D
def plot_decision_boundary(X, y, model, title, filename):
    plt.figure(figsize=(14, 10))
    
    # Membuat grid untuk visualisasi decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Prediksi untuk setiap titik di grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot data points dengan warna dan bentuk berbeda berdasarkan kelas
    for i, (gesture_id, gesture_name) in enumerate(gesture_names.items()):
        # Filter untuk kelas saat ini
        mask = y == gesture_id
        plt.scatter(X[mask, 0], X[mask, 1], 
                    c=[colors[i]], 
                    marker=markers[i % len(markers)], 
                    s=70, alpha=0.7, edgecolors='k',
                    label=f"{gesture_id}: {gesture_name}")
    
    # Plot support vectors jika ada
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                   s=100, linewidth=1, facecolors='none', edgecolors='red',
                   label='Support Vectors')
    
    plt.title(title, fontsize=14)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.legend(title='Kelas Gerakan', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(get_save_path(filename))
    plt.close()

# 2.1 Train model SVM dengan berbagai kernel
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Kernel yang akan diuji
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Training dan visualisasi untuk setiap kernel
for kernel in kernels:
    svm = SVC(kernel=kernel, gamma='scale', C=1.0)
    svm.fit(X_train, y_train)
    plot_decision_boundary(X_pca, y, svm, 
                          f'SVM dengan Kernel {kernel.capitalize()}',
                          f'svm_{kernel}_boundary.png')

# 2.2 Visualisasi performa dengan berbagai parameter C
C_values = [0.1, 1, 10, 100]
train_scores = []
test_scores = []

for C in C_values:
    svm = SVC(kernel='rbf', C=C, gamma='scale')
    svm.fit(X_train, y_train)
    
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    # Plot decision boundary untuk berbagai nilai C
    plot_decision_boundary(X_pca, y, svm, 
                          f'SVM RBF dengan C={C}',
                          f'svm_rbf_C{C}_boundary.png')

# Plot performa vs parameter C
plt.figure(figsize=(12, 8))
plt.plot(C_values, train_scores, 'o-', label='Train Score', linewidth=2, markersize=10)
plt.plot(C_values, test_scores, 's-', label='Test Score', linewidth=2, markersize=10)
plt.xscale('log')
plt.xlabel('Parameter C', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Performa SVM vs Parameter C', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(C_values)

# Tambahkan label nilai akurasi di atas setiap titik
for i, (c, train_score, test_score) in enumerate(zip(C_values, train_scores, test_scores)):
    plt.annotate(f"{train_score:.3f}", (c, train_score), 
                 textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{test_score:.3f}", (c, test_score), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig(get_save_path('svm_C_performance.png'))
plt.close()

# 2.3 Confusion Matrix untuk best model
best_C = C_values[np.argmax(test_scores)]
best_svm = SVC(kernel='rbf', C=best_C, gamma='scale')
best_svm.fit(X_train, y_train)
y_pred = best_svm.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
           xticklabels=[f"{i}: {name}" for i, name in gesture_names.items()],
           yticklabels=[f"{i}: {name}" for i, name in gesture_names.items()])
plt.title(f'Confusion Matrix (C={best_C})', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig(get_save_path('confusion_matrix.png'))
plt.close()

# 2.4 ROC Curve untuk multiclass (one-vs-rest)
# Mendapatkan jumlah kelas unik
n_classes = len(np.unique(y))

# Buat classifier one-vs-rest
clf = OneVsRestClassifier(SVC(kernel='rbf', C=best_C, gamma='scale', probability=True))
clf.fit(X_train, y_train)

# Menghitung probabilitas prediksi (jika memungkinkan)
try:
    y_score = clf.predict_proba(X_test)
    
    # Compute ROC curve dan ROC area untuk setiap kelas
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # Untuk setiap kelas, hitung fpr dan tpr
        y_test_binary = np.array([1 if y_val == i else 0 for y_val in y_test])
        if i < len(y_score[0]):  # Check if we have probability for this class
            fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC Curves
    plt.figure(figsize=(14, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    for i, color in zip(range(min(n_classes, len(colors))), colors):
        if i in roc_auc:
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve {i}: {gesture_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(get_save_path('roc_curve.png'))
    plt.close()
except (AttributeError, ValueError) as e:
    print(f"Tidak dapat membuat ROC curve: {e}")

# 2.5 Visualisasi fitur untuk setiap kelas berdasarkan jenis gerakan
# Radar chart untuk nilai rata-rata fitur per kelas
features_for_radar = ['normalized_area', 'convexity', 'finger_count', 'aspect_ratio']
class_means = df.groupby('gesture_class')[features_for_radar].mean()

# Normalisasi nilai untuk radar chart
from sklearn.preprocessing import MinMaxScaler
scaler_radar = MinMaxScaler()
class_means_scaled = pd.DataFrame(
    scaler_radar.fit_transform(class_means),
    index=class_means.index,
    columns=class_means.columns
)

# Membuat radar chart
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            
        def fill(self, *args, **kwargs):
            return super().fill_between(*args, **kwargs)
            
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            self._close_line(lines[0])
            return lines
            
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
                
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            
        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, 
                                      radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
                
        def draw(self, renderer):
            if frame == 'circle':
                patch = Circle((0.5, 0.5), 0.5)
                patch.set_transform(self.transAxes)
                patch.set_clip_on(False)
                patch.set_fill(False)
                self.add_patch(patch)
                self.set_frame_on(False)
            
            PolarAxes.draw(self, renderer)
            self._update_line_labels()
            
        def _update_line_labels(self):
            lines = []
            for child in self.get_children():
                if isinstance(child, Line2D):
                    lines.append(child)
            
    register_projection(RadarAxes)
    return theta

def unit_poly_verts(theta):
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def radar_patch(r, theta):
    verts = unit_poly_verts(theta)
    return np.array([(r[i] * (verts[i][0] - 0.5) + 0.5,
                      r[i] * (verts[i][1] - 0.5) + 0.5) for i in range(len(r))])

# Mendapatkan jumlah fitur dan menghitung sudut
N = len(features_for_radar)
theta = radar_factory(N, frame='polygon')

# Plot radar chart
fig = plt.figure(figsize=(15, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(class_means_scaled)))

for i, (idx, row) in enumerate(class_means_scaled.iterrows()):
    ax = fig.add_subplot(2, 4, i+1, projection='radar')
    ax.plot(theta, row.values, color=colors[i], linewidth=2)
    ax.fill(theta, row.values, color=colors[i], alpha=0.25)
    ax.set_varlabels(features_for_radar)
    ax.set_title(f"{idx}: {gesture_names[idx]}", pad=20, fontsize=12)
    
    # Tambahkan lingkaran grid
    for j in range(1, 6):
        ax.plot(np.linspace(0, 2*np.pi, 100), 
                 np.ones(100) * j/5, 
                 color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig(get_save_path('feature_radar_chart.png'))
plt.close()

# 2.6 Visualisasi Feature Importance
# Menggunakan koefisien dari model SVM linear untuk mengukur pentingnya fitur
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_scaled, y)  # Menggunakan data asli yang telah distandarisasi

try:
    # Ambil koefisien
    if hasattr(linear_svm, 'coef_'):
        # Ambil magnitude koefisien untuk setiap kelas
        coef_magnitudes = np.abs(linear_svm.coef_)
        # Ambil rata-rata magnitude untuk semua kelas (untuk multiclass)
        avg_coef_magnitudes = np.mean(coef_magnitudes, axis=0)
        
        # Plot kepentingan fitur
        plt.figure(figsize=(14, 8))
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': avg_coef_magnitudes
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance dari Model SVM Linear', fontsize=14)
        plt.xlabel('Importance (Magnitude Koefisien Rata-rata)', fontsize=12)
        plt.ylabel('Fitur', fontsize=12)
        plt.tight_layout()
        plt.savefig(get_save_path('feature_importance.png'))
        plt.close()
except (AttributeError, ValueError) as e:
    print(f"Tidak dapat menampilkan feature importance: {e}")

print("Semua visualisasi selesai dibuat!")
print(f"Visualisasi tersimpan di: {visual_dir}")
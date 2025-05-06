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
from scipy import interp
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create data visualization directory in the same folder as the script
visual_dir = os.path.join(script_dir, 'data_visualizations')
if not os.path.exists(visual_dir):
    os.makedirs(visual_dir)
    print(f"Created directory: {visual_dir}")

# Function to create complete path for saving figures
def get_save_path(filename):
    return os.path.join(visual_dir, filename)

# Membaca data - fix the path to use the script directory
csv_path = os.path.join(script_dir, 'hand_gesture_data_svm.csv')
df = pd.read_csv(csv_path)

# 1. Karakteristik Data

# Fungsi pembantu untuk membuat plot
def create_figure(figsize=(12, 8)):
    return plt.figure(figsize=figsize)

# 1.1 Informasi dasar dataset
print("Informasi Dataset:")
print(f"Jumlah sampel: {df.shape[0]}")
print(f"Jumlah fitur: {df.shape[1] - 1}")  # -1 untuk kolom kelas
print(f"Distribusi kelas:")
print(df['gesture_class'].value_counts())

# 1.2 Visualisasi distribusi kelas
plt.figure(figsize=(10, 6))
sns.countplot(x='gesture_class', data=df)
plt.title('Distribusi Kelas Gerakan Tangan')
plt.xlabel('Kelas Gerakan')
plt.ylabel('Jumlah Sampel')
plt.savefig(get_save_path('class_distribution.png'))
plt.close()

# 1.3 Pair plot untuk beberapa fitur penting
# Memilih beberapa fitur untuk pair plot (akan terlalu crowded jika menggunakan semua)
selected_features = ['THUMB_TIP_x', 'INDEX_FINGER_TIP_x', 'MIDDLE_FINGER_TIP_x', 
                     'THUMB_TIP_y', 'INDEX_FINGER_TIP_y', 'MIDDLE_FINGER_TIP_y']
pair_plot_df = df[selected_features + ['gesture_class']].sample(min(500, len(df)))  # Sample untuk performa
sns.pairplot(pair_plot_df, hue='gesture_class', palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
plt.savefig(get_save_path('feature_pairplot.png'))
plt.close()

# 1.4 Heatmap korelasi
plt.figure(figsize=(14, 12))
feature_cols = df.columns[:-1]  # Semua kolom kecuali kelas
correlation = df[feature_cols].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriks Korelasi Fitur')
plt.tight_layout()
plt.savefig(get_save_path('correlation_heatmap.png'))
plt.close()

# 1.5 Box plots untuk fitur
plt.figure(figsize=(15, 10))
feature_subset = ['THUMB_TIP_x', 'INDEX_FINGER_TIP_x', 'MIDDLE_FINGER_TIP_x', 
                  'RING_FINGER_TIP_x', 'PINKY_TIP_x']
df_melted = pd.melt(df, id_vars=['gesture_class'], value_vars=feature_subset,
                   var_name='Feature', value_name='Value')
sns.boxplot(x='Feature', y='Value', hue='gesture_class', data=df_melted)
plt.title('Box Plot Fitur Koordinat X Ujung Jari')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(get_save_path('boxplot_x_tips.png'))
plt.close()

# 1.6 Visualisasi PCA
# Standarisasi data
X = df.drop('gesture_class', axis=1)
y = df['gesture_class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA untuk visualisasi 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                     alpha=0.7, s=50, edgecolors='w')
plt.colorbar(scatter, label='Kelas Gerakan')
plt.title('Visualisasi PCA: Proyeksi 2D Data Gerakan Tangan')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(get_save_path('pca_visualization.png'))
plt.close()

# PCA untuk visualisasi 3D
pca3d = PCA(n_components=3)
X_pca3d = pca3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3d[:, 0], X_pca3d[:, 1], X_pca3d[:, 2],
                     c=y, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, ax=ax, label='Kelas Gerakan')
ax.set_title('Visualisasi PCA: Proyeksi 3D Data Gerakan Tangan')
ax.set_xlabel(f'PC1 ({pca3d.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca3d.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({pca3d.explained_variance_ratio_[2]:.2%})')
plt.tight_layout()
plt.savefig(get_save_path('pca_3d_visualization.png'))
plt.close()

# 2. Visualisasi SVM

# Fungsi untuk plot decision boundary pada PCA 2D
def plot_decision_boundary(X, y, model, title, filename):
    plt.figure(figsize=(12, 10))
    
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
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                         edgecolors='k', alpha=0.8)
    plt.colorbar(scatter, label='Kelas Gerakan')
    
    # Plot support vectors jika ada
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                   s=100, linewidth=1, facecolors='none', edgecolors='red')
    
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
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
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_scores, 'o-', label='Train Score')
plt.plot(C_values, test_scores, 's-', label='Test Score')
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.title('Performa SVM vs Parameter C')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(get_save_path('svm_C_performance.png'))
plt.close()

# 2.3 Confusion Matrix untuk best model
best_C = C_values[np.argmax(test_scores)]
best_svm = SVC(kernel='rbf', C=best_C, gamma='scale')
best_svm.fit(X_train, y_train)
y_pred = best_svm.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (C={best_C})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(get_save_path('confusion_matrix.png'))
plt.close()

# 2.4 ROC Curve untuk multiclass (one-vs-rest)
# Mendapatkan jumlah kelas unik
n_classes = len(np.unique(y))

# Buat classifier one-vs-rest
clf = OneVsRestClassifier(SVC(kernel='rbf', C=best_C, gamma='scale', probability=True))
clf.fit(X_train, y_train)

# Menghitung probabilitas prediksi
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
plt.figure(figsize=(12, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
for i, color in zip(range(min(n_classes, 10)), colors):  # Plot max 10 classes
    if i in roc_auc:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(get_save_path('roc_curve.png'))
plt.close()

# 2.5 Visualisasi Tangan 3D (beberapa sampel)
# Plot 3D scatter untuk posisi jari tangan (sampel per kelas)
unique_classes = df['gesture_class'].unique()
samples_per_class = 3
fig = plt.figure(figsize=(15, 15))

for i, cls in enumerate(unique_classes[:min(6, len(unique_classes))]):  # Plot max 6 classes
    class_samples = df[df['gesture_class'] == cls].iloc[:samples_per_class]
    
    for j, sample in enumerate(class_samples.iterrows()):
        sample = sample[1]
        ax = fig.add_subplot(len(unique_classes[:min(6, len(unique_classes))]), 
                            samples_per_class, 
                            i*samples_per_class + j + 1, 
                            projection='3d')
        
        # Extract finger positions
        fingers_x = [sample['THUMB_TIP_x'], sample['INDEX_FINGER_TIP_x'], 
                   sample['MIDDLE_FINGER_TIP_x'], sample['RING_FINGER_TIP_x'], 
                   sample['PINKY_TIP_x'], sample['WRIST_x']]
        fingers_y = [sample['THUMB_TIP_y'], sample['INDEX_FINGER_TIP_y'], 
                   sample['MIDDLE_FINGER_TIP_y'], sample['RING_FINGER_TIP_y'], 
                   sample['PINKY_TIP_y'], sample['WRIST_y']]
        fingers_z = [sample['THUMB_TIP_z'], sample['INDEX_FINGER_TIP_z'], 
                   sample['MIDDLE_FINGER_TIP_z'], sample['RING_FINGER_TIP_z'], 
                   sample['PINKY_TIP_z'], sample['WRIST_z']]
        
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'Wrist']
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'black']
        
        # Plot points
        for k in range(len(fingers_x)):
            ax.scatter(fingers_x[k], fingers_y[k], fingers_z[k], 
                      c=colors[k], s=100, label=finger_names[k] if j == 0 and i == 0 else "")
        
        # Connect fingers to wrist
        for k in range(5):  # 5 fingers
            ax.plot([fingers_x[k], fingers_x[5]], 
                   [fingers_y[k], fingers_y[5]], 
                   [fingers_z[k], fingers_z[5]], c='gray', alpha=0.5)
        
        ax.set_title(f'Class {cls} - Sample {j+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=30)
        
        # Add legend only for the first plot
        if i == 0 and j == 0:
            ax.legend()

plt.tight_layout()
plt.savefig(get_save_path('hand_3d_visualization.png'))
plt.close()

print("Semua visualisasi selesai dibuat!")
print(f"Visualisasi tersimpan di: {visual_dir}")
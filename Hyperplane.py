import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# 1. Load data
df = pd.read_csv('hand_gesture_data.csv')
X = df.drop('gesture_class', axis=1).values
y = df['gesture_class'].values

# 2. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. PCA ke 2 komponen
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# 5. Latih SVM linear
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train_pca, y_train)

# 6. Buat mesh grid untuk keputusan
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 7. Prediksi setiap titik grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 8. Plot region keputusan & sampel
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                      c=y_train, edgecolor='k', s=20)
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('Linear SVM Decision Regions on PCA-Reduced Hand Gesture Data')
plt.legend(*scatter.legend_elements(), title="Gesture Class",
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 9. Simpan sebagai PNG
output_path = 'svm_pca_decision.png'
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved to {output_path}")

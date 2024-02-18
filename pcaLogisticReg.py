import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Veriyi yükle
data = pd.read_csv("C:/Users/ABREBO/Desktop/csv/StressLevelDataset.csv")

# Bağımsız değişkenleri ve hedef değişkeni ayır
x = data[['sleep_quality']]
y = data['stress_level']

# Veriyi standartlaştır
sc = StandardScaler()
x_std = sc.fit_transform(x)

# PCA ile boyut azaltma
pca = PCA(n_components=1)  # 1 bileşen seçiyoruz
x_pca = pca.fit_transform(x_std)

# Eğitim ve test setlerine ayırma
xtrain_pca, xtest_pca, ytrain, ytest = train_test_split(x_pca, y, test_size=0.33, random_state=0)
# Logistic regresyon modelini eğitme
lr_pca = LogisticRegression()
lr_pca.fit(xtrain_pca, ytrain)

# PCA ile tahmin yapma
y_pred_pca = lr_pca.predict(xtest_pca)
# Doğruluk oranını hesaplama
accuracy_pca = accuracy_score(ytest, y_pred_pca)
conf_matrix_pca = confusion_matrix(ytest, y_pred_pca)
# Sonuçları yazdırma
print("Doğruluk Oranı (PCA):", accuracy_pca)
print("Karışıklık Matrisi (PCA):\n", conf_matrix_pca)
# Karışıklık matrisini görselleştirme
sns.heatmap(conf_matrix_pca, annot=True, fmt='d', cmap='Blues', xticklabels=lr_pca.classes_, yticklabels=lr_pca.classes_)
plt.xlabel('Tahmin Edilen Değer')
plt.ylabel('Gerçek Değer')
plt.title('Karışıklık Matrisi (PCA)')
plt.show()

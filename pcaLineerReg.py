import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Veri setini yükle
data = pd.read_csv("C:/Users/ABREBO/Desktop/csv/StressLevelDataset.csv")


# Bağımsız değişkenleri ve hedef değişkeni ayır
x = data.iloc[:, 0:20].values
y = data.stress_level.values.reshape(-1, 1)

# Verileri standartlaştır
scaler = StandardScaler()
x_std = scaler.fit_transform(x)


# Veriyi PCA ile boyut azaltma
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# Boyut azaltılmış veriyi eğitim ve test setlerine ayırma
xtrain_pca, xtest_pca, ytrain_pca, ytest_pca = train_test_split(x_pca, y, test_size=0.33, random_state=0)

# Lineer regresyon modelini boyut azaltılmış veri üzerinde eğitme
lr_pca = LinearRegression()
lr_pca.fit(xtrain_pca, ytrain_pca)

# Boyut azaltılmış veri ile tahmin yapma
ypred_pca = lr_pca.predict(xtest_pca)

# Boyut azaltılmış veri için hata hesaplama
mse_pca = mean_squared_error(ytest_pca, ypred_pca)
res_pca = 1 - mse_pca
print(f'Mean Squared Error (PCA): {mse_pca}')
print("Dogruluk orani (PCA): " + str(res_pca))

# Farkları görselleştirme (örneğin, histogram olarak)
differences_pca = ytest_pca - ypred_pca
plt.hist(differences_pca, bins=20)
plt.xlabel('Gerçek - Tahmin (PCA)')
plt.ylabel('Frekans')
plt.title('PCA ile Boyut Azaltılmış Veride Gerçek ve Tahmin Arasındaki Farklar')
plt.show()

# Verilen girdiyi kullanarak PCA ile tahmin yapma
input_values_pca = pca.transform(np.array([14, 20, 0, 11, 2, 1, 2, 4, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2]).reshape(1, -1))
prediction_pca = lr_pca.predict(input_values_pca)

# Tahmin sonucunu yazdırma (PCA)
print("Tahmin Edilen Stres Seviyesi (PCA):", prediction_pca[0])


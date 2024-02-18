import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/ABREBO/Desktop/csv/StressLevelDataset.csv")

x = data[['sleep_quality']]  #bağımsız değişkenler dataframe olmalı çift köşeli parantez dataframe yapıyor
y = data['stress_level']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=0)

    
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

xtrain1=sc.fit_transform(xtrain)
xtest1=sc.transform(xtest)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain1,ytrain)

y_pred=lr.predict(xtest1)

print(lr.score(xtest1, ytest))
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error
import seaborn as sns

accuracy = accuracy_score(ytest, y_pred)
conf_matrix = confusion_matrix(ytest, y_pred)
print("Doğruluk Oranı:", accuracy)
print("Karışıklık Matrisi:\n", conf_matrix)

# Karışıklık matrisini görselleştirme
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=lr.classes_, yticklabels=lr.classes_)
plt.xlabel('Tahmin Edilen Değer')
plt.ylabel('Gerçek Değer')
plt.title('Karışıklık Matrisi')
plt.show()


sifir = data[data.stress_level == 0]
bir = data[data.stress_level == 1]
iki = data[data.stress_level == 2]



plt.figure(figsize=(10, 6))

# Stress Level 0 için KDE plot
sns.kdeplot(data=sifir['sleep_quality'], label='Stress Level 0', color='green', shade=True)

# Stress Level 1 için KDE plot
sns.kdeplot(data=bir['sleep_quality'], label='Stress Level 1', color='blue', shade=True)

# Stress Level 2 için KDE plot
sns.kdeplot(data=iki['sleep_quality'], label='Stress Level 2', color='red', shade=True)

plt.xlabel('Uyku Kalitesi')
plt.ylabel('Yoğunluk')
plt.legend()
plt.title('Uyku Kalitesi ve Stres Seviyesi Yoğunluk İlişkisi')
plt.show()



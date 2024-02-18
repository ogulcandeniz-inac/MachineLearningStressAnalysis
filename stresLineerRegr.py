import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/ABREBO/Desktop/csv/StressLevelDataset.csv")

eksik_veri = data.isnull().any()

        
correlation_matrix = data.corr()
print(correlation_matrix)


x=data.iloc[:,0:20].values
y=data.stress_level.values.reshape(-1,1)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)

y_pred=lr.predict(xtest)


from sklearn.metrics import  classification_report, confusion_matrix,mean_squared_error,accuracy_score

#lineer reg de genellikle mse kullanılır
mse = mean_squared_error(ytest, y_pred)
print(f'Mean Squared Error: {mse}')
res=1-mse
print("Dogruluk orani: "+str(res))

#histogram olarak farkları görselleştirme
differences = ytest - y_pred
plt.hist(differences, bins=20)
plt.xlabel('Gerçek - Tahmin')
plt.ylabel('Frekans')
plt.title('Gerçek ve Tahmin Arasındaki Farklar')
plt.show()

# Verilen girdiyi kullanarak tahmin yapma
input_values = np.array([14, 20, 0, 11, 2, 1, 2, 4, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2]).reshape(1, -1)
prediction = lr.predict(input_values)
print("Tahmin Edilen Stres Seviyesi:", prediction[0])
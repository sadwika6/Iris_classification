# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:21:29 2023

@author: sadwika sabbella
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\sadwika sabbella\Desktop\pyfh\iris.csv")
print(data.shape)
#print(data)
print(data.head())

#print(data.isna())
print(data.isna().sum())
#print(data.describe())



data['Extra']=range(1,len(data)+1)
data.columns
data['Extra']



#data['pl'].fillna((data['pl']).mean(),inplace=True) # 100

data['pl'].fillna((data['pl']).mode()[0],inplace=True) #100 97.7(test_size = 0.3)

#data.dropna()

#data['pl'].fillna(0,inplace=True) # 100

#data['pl'].fillna(10,inplace=True) # 100    97.7(test_size = 0.3)


#print(data.corr())
a=data.corr()
plt.matshow(a)
plt.colorbar()

#x=np.array(data.iloc[:,:-1]) # converts dataframe into 2D array
#y=data.iloc[:,-1].values # converts series into 1D array

x=np.array(data.iloc[:,[0,1,2,3,5]])
y=data.iloc[:,-2].values 
print(x.shape,y.shape)


#np.array() are .values are similar
#print(x)
#print(y)
#print(x.shape,y.shape)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
#test_size ---> ratio of test data and train data (80,20)
#random_state --> same set of random trained data will be considered

print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)

##If the dataset contains empty cells ValueError raises at this line

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, ypred)*100)

#print(model.predict([[5.4,4.5,3.4,1.6]]))
#print(model.predict([[1.2,0.9,1.0,0.8]]))
#print(model.predict([[6.5,4.9,5.6,3.9]]))






















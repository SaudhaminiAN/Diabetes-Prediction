# -*- coding: utf-8 -*-
"""Diabetes prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PR1eiJ15zQYzq6g4QII_mVuTxN3Epvwk

# Diabetes Prediction using Support Vector machine(ML)

Importing the dependencies
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data collection and Data processing"""

#loading the dataset to the pandas dataframe
data= pd.read_csv('/content/diabetes.csv')
data.head()

#describe-->statistical measures of data
data.describe()

#number of rows and columns
data.shape

data['Outcome'].value_counts()

data.groupby('Outcome').mean()

#seperating data and labels
x=data.drop(columns='Outcome',axis=1)
y=data['Outcome']

print(x)

print(y)

"""Data Standardization"""

scaler= StandardScaler()
scaler.fit(x)
standardized_data=scaler.transform(x)
print(standardized_data)

x=standardized_data
y=data['Outcome']
print(x)
print(y)

"""Train Test Split"""

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=.2, stratify=y, random_state=2)
print(x.shape,x_train.shape,x_test.shape)

"""Training the model"""

classifier=svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

"""Model Evaluation

"""

#accuracy on training data
x_train_prediction=classifier.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediction,y_train)
print('accuracy of training data:',training_data_accuracy)

x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('accuracy of training data:',test_data_accuracy)

input_data=(6,148,72,35,0,33.6,0.627,50)
#changing the input_data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if (prediction[0])==0:
  print('The person is not diabetes')
else:
  print('The person is diabetes')

"""Saving the trained model"""

import pickle
filename='trained_model.sav'
pickle.dump(classifier,open(filename,'wb'))

#loading the saved model
loaded_model=pickle.load(open('trained_model.sav','rb'))

input_data=(6,148,72,35,0,33.6,0.627,50)
#changing the input_data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0])==0:
  print('The person is not diabetes')
else:
  print('The person is diabetes')


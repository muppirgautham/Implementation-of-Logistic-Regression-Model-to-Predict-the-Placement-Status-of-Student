# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown value. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GAUTHAM M
RegisterNumber:  212221230027
*/
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## Original data(first five columns):
![image](https://user-images.githubusercontent.com/94810884/167667281-dfee026d-b6a4-49f8-9db2-b193c3948a11.png)

## Data after dropping unwanted columns(first five):
![image](https://user-images.githubusercontent.com/94810884/167667490-5a669fab-94df-4c0b-943c-6f22ffaf7e06.png)

## Checking the presence of null values:
![image](https://user-images.githubusercontent.com/94810884/167667599-87660ca9-894c-498f-b2d8-f87c093893ae.png)

## Checking the presence of duplicated values:
![image](https://user-images.githubusercontent.com/94810884/167667710-eda42f5a-0740-4db2-8cc1-3040184f133a.png)

## Data after Encoding:
![image](https://user-images.githubusercontent.com/94810884/167669140-718f9055-0243-4c92-95c0-22d337a94c77.png)

## X Data:
![image](https://user-images.githubusercontent.com/94810884/167668108-e0ce7ebd-d57c-4a3b-aa3c-d6010eedba53.png)

## Y Data:
![image](https://user-images.githubusercontent.com/94810884/167668213-78f799d5-8691-49f5-b51f-179ff152c6bf.png)

## Predicted Values:
![image](https://user-images.githubusercontent.com/94810884/167668309-f5d485fe-2995-4cbe-8192-4078d04a3c91.png)

## Accuracy Score:
![image](https://user-images.githubusercontent.com/94810884/167668432-668e7f3d-eacf-4aac-a067-03245d19c6f5.png)

## Confusion Matrix:
![image](https://user-images.githubusercontent.com/94810884/167668587-2f3f2754-88f1-46a4-b526-a60bb9629d4a.png)

## Classification Report:
![image](https://user-images.githubusercontent.com/94810884/167668691-46788d79-ed78-4d24-85a3-05d178dd9a0a.png)

## Predicting output from Regression Model:
![image](https://user-images.githubusercontent.com/94810884/167668809-1db2c666-7252-486c-9755-8271a3bc1067.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

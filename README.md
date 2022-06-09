# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

```
## Algorithm
#step1:
import pandas module.
#step2:
Read the required csv file using pandas
#step3:
Import LabEncoder module.
#step4:
From sklearn import logistic regression.
#step5:
Predict the values of array.
#step6:
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
#step7:
print the required values
#step8:
End the program.

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:k sucharitha
RegisterNumber:212221240021
*/
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))

```
## Output:
![3 1](https://user-images.githubusercontent.com/94166007/172882697-840b6e46-56c6-415a-aee1-342b028c8989.jpeg)

![3 2](https://user-images.githubusercontent.com/94166007/172882795-f3e6f110-7b8a-4a66-a351-04e24b2ed906.jpeg)

![3 3](https://user-images.githubusercontent.com/94166007/172882846-f434f7e4-b602-42fa-92c6-a7b095eb1131.jpeg)

![3 4](https://user-images.githubusercontent.com/94166007/172882889-4e2c0a92-312f-4965-afef-9f4fb325aa42.jpeg)

![3 5](https://user-images.githubusercontent.com/94166007/172882909-86111997-e799-4bef-90cf-dd92c6c82b5e.jpeg)

![3 6](https://user-images.githubusercontent.com/94166007/172883176-e0d894f5-0e74-4227-aeb6-918970bba688.jpeg)

![3 7](https://user-images.githubusercontent.com/94166007/172883073-2d631155-5970-4ab5-84d5-ef5d22128ad4.jpeg)

![3 8](https://user-images.githubusercontent.com/94166007/172883248-8e78128e-a8a1-4bd0-a9bd-609fddc2b697.jpeg)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

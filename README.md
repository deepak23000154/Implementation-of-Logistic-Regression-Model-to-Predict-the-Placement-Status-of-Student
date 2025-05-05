# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Prepare the Dataset Load Placement_Data.csv into a DataFrame.
Make a copy of the original data to work on (data1).

Drop unnecessary columns: sl_no (just an index) and salary (not known before placement).

2.Handle Missing and Duplicate Data Check for missing values with isnull().sum().
Check for duplicate rows with duplicated().sum().

3.Encode Categorical Columns Use LabelEncoder to convert categorical features into numeric values.
The following columns are encoded:

gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation, and status.

4.Split Features and Target x = all columns except status (input features).
y = status column (target variable: 1 = placed, 0 = not placed).

5.Train-Test Split Use train_test_split() to divide data:
80% training

20% testing

Set random_state=0 for reproducibility

6.Train Logistic Regression Model Use LogisticRegression with liblinear solver.
Fit the model on training data (x_train, y_train).

7.Predict and Evaluate Predict placement status for x_test.
Use:

accuracy_score to compute accuracy

confusion_matrix to see TP, FP, FN, TN

classification_report for precision, recall, F1-score

8.Make a Custom Prediction
Predict placement status for a new student record:

[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85] These values represent:

Encoded gender, ssc %, board, hsc %, board, stream, degree %, type, workex, etc.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DEEPAK.R
RegisterNumber:  2122223040031
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
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

![image](https://github.com/user-attachments/assets/dd52df2b-7a7a-4396-b3a3-dccf74b12078)


![image](https://github.com/user-attachments/assets/3f4cb1cb-d233-4de9-9fdb-f0ecab0d873c)


![image](https://github.com/user-attachments/assets/dcf01bcf-89b3-4f78-b99f-7bbb42af9303)


![image](https://github.com/user-attachments/assets/c7142b89-99e1-4690-97c6-d37f6580566b)


![image](https://github.com/user-attachments/assets/8551d3ab-6032-4334-8293-58101acb27b1)

![image](https://github.com/user-attachments/assets/f0819c93-7d65-47af-ad9f-318337dbe050)

![image](https://github.com/user-attachments/assets/89c3aeac-002d-4e05-8897-f9b056091fd8)

![image](https://github.com/user-attachments/assets/622b040c-7567-4feb-b827-99b8544c9bd0)

![image](https://github.com/user-attachments/assets/63f590e6-6d6e-4adb-bcfa-b359ed71e454)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.




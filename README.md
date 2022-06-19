# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Parveen Fathima M
RegisterNumber: 212219040103  
*/
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="latin-1")
data.head()
data.info()
df.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy= metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Dataset:
![dataset](https://user-images.githubusercontent.com/87666371/174471746-30c56224-cd18-4e57-b99b-5aa9936e0f97.png)

## Dataset information:
![datasetinfo](https://user-images.githubusercontent.com/87666371/174471769-c02c1af0-158f-45f5-a1ef-af33e47aeefd.png)

## Detected spam:
![detected](https://user-images.githubusercontent.com/87666371/174471794-c0252ee8-95c4-4fec-a0ac-2c5d54bd82ca.png)

## Accuracy score of the model:
![accuracy](https://user-images.githubusercontent.com/87666371/174471818-975685b5-8ba6-4788-b92b-c20fef045796.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
   
2.Read the dataset and separate the independent and dependent variables.

3.Split the dataset into training and testing.

4.Do preprocessing if needed, in this case vectorization is needed which is done using CountVectorizer()

5.Train the model using SVC() algorithm and .fit()

6.Predict the model on x_test.

7.Measure its accuracy
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Kanamarlapudi Sai Charithanjali
RegisterNumber:212224240069  
*/

import pandas as pd
    data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
    data.info()
    
    x=data['v2'].values
    y=data['v1'].values
    x.shape
    y.shape
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer()
    x_train=cv.fit_transform(x_train)
    x_test=cv.transform(x_test)
    x_train
    
    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x_train,y_train)
    y_pred=svc.predict(x_test)
    y_pred
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,y_pred)
    acc


```

## Output:

![WhatsApp Image 2025-11-06 at 08 24 18_8f2667ac](https://github.com/user-attachments/assets/55b67cb2-480b-4438-8653-f66d6a3b56a7)

![WhatsApp Image 2025-11-06 at 08 22 34_6b594507](https://github.com/user-attachments/assets/35da8043-e5d0-4acd-a434-477739dcc81b)

![WhatsApp Image 2025-11-06 at 08 22 48_fafe8e55](https://github.com/user-attachments/assets/42b8b8e5-e6ae-4865-a264-05747b9028ce)

![WhatsApp Image 2025-11-06 at 08 23 08_d52fb24e](https://github.com/user-attachments/assets/1549404b-d2a8-4c10-9eed-f256605858b8)

![WhatsApp Image 2025-11-06 at 08 23 19_ca018988](https://github.com/user-attachments/assets/6fe3447b-8d9e-40d8-89e8-884fbfaf25d5)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

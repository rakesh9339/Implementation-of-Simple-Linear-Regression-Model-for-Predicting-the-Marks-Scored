# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("student_scores.csv") 
df.head()
df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color="orangered",s=60)
plt.plot(x_train,regressor.predict(x_train),color="darkviolet",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()

plt.scatter(x_test,y_test,color="seagreen",s=60)
plt.plot(x_test,regressor.predict(x_test),color="cyan",linewidth=4)
plt.title("hours vs scores(training set)",fontsize=24)
plt.xlabel("Hours",fontsize=18)
plt.ylabel("scores",fontsize=18)
plt.show()


mse=mean_squared_error(_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```


## Output:
### Head:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/18f3af86-9ae2-4494-bb61-636b83a7bcd5)
### Tail:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/0a120341-9b3f-4ee2-8740-1ea5cdc71610)
### Array value of X:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/e296242a-26a5-4ba3-9736-86bc8fe4e85c)
### Array value of Y:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/c25d4900-5e51-4bf7-8db7-1ec80106ab55)
### Values of Y prediction:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/611f79c2-9b85-47f4-9a9f-6fbe4dc4e7ee)
### Array values of Y test:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/df0a81e5-6894-4e45-9037-6dce6cb3a7d7)
### Training Set Graph:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/42c59aa0-503e-4dc8-b41a-315bdd9680c1)
### Test Set Graph:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/8e868618-b584-4339-a91e-3b07e48301ba)
### Values of MSE, MAE and RMSE:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/08d3470e-d71f-43e9-971f-e4871b589271)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

##  Supervised learning  ##

# Logistic regression algorithm'

 #collect data: import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


titanic_data = pd.read_csv("D:\Titanic.csv")
print(titanic_data)

#print number of passenger s in the list
print("Number of passenger in the original data:" + str(len(titanic_data.index)))


## Now we analyzing the data
sns.countplot(x="Survived", data=titanic_data)
#plt.show()


#passengers who died (how many were male and female
sns.countplot(x="Survived", hue="Sex", data=titanic_data)
#plt.show()

#which class passengers died or survive more
sns.countplot(x="Survived", hue="Pclass", data=titanic_data)
#plt.show()


#Age distribution
titanic_data['Age'].plot.hist()
#plt.show()


#Fare
titanic_data['Fare'].plot.hist(bins=20, figsize=(10,5))
#plt.show()

# info of the data
titanic_data.info()

#number of suiblings aboard the titanic
sns.countplot(x="SibSp",data=titanic_data)
#plt.show

# Data Wrangling (data cleaning)
print(titanic_data.isnull())
print(titanic_data.isnull().sum())

#heat map for analyzing
sns.heatmap(titanic_data.isnull())
#plt.show()

#remove missing values
sns.boxplot(x="Pclass", y="Age", data=titanic_data)
#plt.show()

print(titanic_data.head(5))

#Drop the unnecessary column
titanic_data.drop("Cabin", axis=1, inplace=True)
print(titanic_data.head(5))

titanic_data.dropna(inplace=True)
#now we check  null values using heat map
sns.heatmap(titanic_data.isnull(),cbar=False)
#plt.show()

#calculate the sum and see null values
print(titanic_data.isnull().sum())


#Now we convert All string values to categorical variable for logistic regression
sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
print(sex.head(5))
#1 is male and 0 is female

#similarly for embark column
embark = pd.get_dummies(titanic_data["Embarked"],drop_first=True)
print(embark.head(5))

#similar for Pclass column
Pcl = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
print(Pcl.head())

titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
print(titanic_data.head(5))

#now we drop the unnecessary column
titanic_data.drop(['Sex','Embarked','PassengerId','Name','Pclass','Ticket'],axis=1, inplace=True)
print(titanic_data.head())
####finally we have cleaned the data

#Train and Test the data
x=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3 ,random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(max_iter=10000)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
"""confusion matrix has 4 values
-- predictive no =
-- predictive yes = 
-- actual  =
-- actual yes = 
so to calculate accuracy we add 102+63 and divide it 
by the whole by sum """

#now we get accuracy
from sklearn.metrics import accuracy_score
"""Hence accuracy is 77%"""

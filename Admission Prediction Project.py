## Admission Prediction Project

#Importing Essential Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Loading the dataset
df = pd.read_csv("D:\Admission_Prediction.csv")
print(df.head())

#returns number of rows and columns of the dataset
print(df.shape)

#returns the first x numberof rows when tail(num. Without a number it returns 5
print(df.tail())

#returns all the coulmns of the dataset
print(df.columns)

#returns the basic information of the coulmn
print(df.info())

#returns the basic statistics on the numeric columns
print(df.describe().T)

#returns the different datatypes of the dataset
print(df.dtypes)

#checking null values
#returns true if null value is present, else false
print(df.isnull().any())

#Renaming the columns with appropriate names
data = df.rename(columns={'GRE Score': 'GRE','TOEFL Score':'TOEFL','LOR ': 'LOR', 'Chance of Admit ': 'Probability'})
print(data.head())


#Data Visualization

#visualizing the feature GRE
fig = plt.hist(data['GRE'], rwidth=0.7)
plt.title("Distribution of GRE Scores")
plt.xlabel("GRE Scores")
plt.ylabel("Count")
#plt.show()

#Visualizing the feature TOEFL
fig = plt.hist(data['TOEFL'], rwidth=0.7)
plt.title("Distribution of TOEFL Scores")
plt.xlabel("TOEFL Scores")
plt.ylabel("Count")
#plt.show()

#Visualizing the feature university rating
fig = plt.hist(data['University Rating'], rwidth=0.7)
plt.title("Distribution of University Rating")
plt.xlabel("University Rating")
plt.ylabel("Count")
#plt.show()

#Visualizing the feature SOP rating
fig = plt.hist(data['SOP'], rwidth=0.7)
plt.title("Distribution of SOP Rating")
plt.xlabel("SOP Rating")
plt.ylabel("Count")
#plt.show()

#Visualizing the feature LOR rating
fig = plt.hist(data['LOR'], rwidth=0.7)
plt.title("Distribution of LOR Rating")
plt.xlabel("LOR Rating ")
plt.ylabel("Count")
#plt.show()

#Visualizing the feature CGPA
fig = plt.hist(data['CGPA'], rwidth=0.7)
plt.title("Distribution of CGPA")
plt.xlabel("CGPA")
plt.ylabel("Count")
#plt.show()

#Visualizing the feature Research
fig = plt.hist(data['Research'], rwidth=0.7)
plt.title("Distribution of Research Papers")
plt.xlabel("Research")
plt.ylabel("Count")
#plt.show()

#Data cleaning
data.drop("Serial No.",axis='columns',inplace=True)
print(data.head())

#Replacing the '0' values from al columns to NaN
data_copy = data.copy(deep=True)
data_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']] = data_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']].replace(0, np.NaN)
print(data_copy.isnull().sum())


# Model Building

#Splitting the dataset in features and label
X = data_copy.drop('Probability', axis='columns')
y = data_copy['Probability']


#Using GridSearchCV TO find the best algorithm for this problem
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

#creating a function to calculate the best model for this problem
#(like which algorithm fits the best)
def find_best_model(X,y):
    models = {
        'linear_regression':{
            'model': LinearRegression(),
            'parameters':{
                'normalize':[True,False]
            }
        },


        'lasso':{
            'model':Lasso(),
            'parameters':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },

        'svr':{
            'model': SVR(),
            'parameters':{
                'gamma':['auto','scale']
            }
        },

        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'parameters':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        },

        'random_forest':{
            'model':RandomForestRegressor(criterion='mse'),
            'parameters':{
                'n_estimators':[5,10,15,20]
            }
        },

        'knn':{
            'model':KNeighborsRegressor(algorithm='auto'),
            'parameters':{
                'n_neighbours':[2,5,10,20]
            }
        }
    }

    scores=[]
    for model_name,model_parameters in models.items():
        gs=GridSearchCV(model_parameters['model'],model_parameters['parameters'],cv=5, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'Model':model_name,
            'Best_parameters':gs.best_params_,
            'Score':gs.best_score_
        })
    return pd.DataFrame(scores,columns=['Model','Best_parameters','Score'])
print(find_best_model(X,y))

"""Here the above code will show which algorithm is best for this prediction
   We create 'def' as find_best_model to get the actual output"""



# Using cross_val_score for gaining highest accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(normalize=True), X, y, cv=5)
print('Highest Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))

#Now we split the dataset into test and train data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=5)
print(len(X_train),len(X_test))

#Creating a Linear Regression Model
model=LinearRegression(normalize=True)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

"""Now The last part we Predict the values using our trained model"""
#prediction1
#Input inthe form: GRE,TOEFL,University Rating, SOP, LOR, CGPA, Research
print('Chance of getting into University of California, Los Angeles(UCLA) is {}%'.format(round(model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]])[0]*100, 3)))

#prediction2
#Input inthe form: GRE,TOEFL,University Rating, SOP, LOR, CGPA, Research
print('Chance of getting into University of California, Los Angeles(UCLA) is {}%'.format(round(model.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]])[0]*100, 3)))

#   SUPERVISED LEARNING #

#Linear regression algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#reading file
data=pd.read_csv("D:\headbrain.csv")
print(data)
#data.head()

#now collecting X & Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

#Mean X and Y
mean_x=np.mean(X)
mean_y=np.mean(Y)
#total number of values
m=len(X)
#using the formula to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

#print Coefficients
print(b1, b0)

#brain weight= 0.26342933948939945
#head size = 325.57342104944223 is the value of C

#Plotting values and regression line
max_x=np.max(X) + 100
min_x=np.min(X) - 100

#calculating line values x and y
x=np.linspace(min_x,max_x,1000)
y=b0 + b1 * x

#plotting line
plt.plot(x,y,color='#58b970', label='Regression Line')
#plotting scatter points
plt.scatter(X,Y,c='#ef5423',label='Scatter Plot')

plt.xlabel('Head Size in cm^3')
plt.ylabel('Brain Weight in grams')
plt.legend()
#plt.show()


# now we find how good our model is!!
# rootmeansquaremethod, r squaredmethod, coefficient of determination

# we calculate R^2 value
ss_t=0   #total sum of squares
ss_r=0   #total sum of residuals
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)



#simple linear regression model using least square method

# linear regression using machine learning library skit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cannot use Rank 1 matrix in scikit learn

X = X.reshape((m, 1))
# Creating Model
reg=LinearRegression()

# Fitting training data
reg=reg.fit(X,Y)
# Y Prediction
Y_pred=reg.predict(X)

# Calculating RMSE and R2 Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)

print (np.sqrt(mse))
print (r2_score)
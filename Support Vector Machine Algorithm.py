#SUPERVISED LEARNING

# Support Vector Machine
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#loading the Data
cancer_data = datasets.load_breast_cancer()
print(cancer_data)
print(cancer_data['target']) 

x_train,x_test,y_train,y_test = train_test_split(cancer_data.data, cancer_data.target,test_size=0.4, random_state=209)

#Generating the Model
cls = svm.SVC(kernel="linear")

#train the model
cls.fit(x_train,y_train)

#predict the response
pred = cls.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred=pred))

#precision score
print("Precision:",metrics.precision_score(y_test,y_pred=pred))

#Recall score
print("Recall:", metrics.recall_score(y_test,y_pred=pred))
print(metrics.classification_report(y_test,y_pred=pred))




#   SUPERVISED LEARNING #

## Raandom Forest Algorithm

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("D:\Social_Network_Ads.csv")
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
(dataset.head())
print(dataset.shape)
print(dataset.info)


#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

print(X_train.shape)
print(X_test[:3])
print(Y_train)
print(Y_test)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train[0:10])
print(X_test[0:5])

#Training the random forest classification model on the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion='entropy', random_state=0)
"""estimators=5 is here we are creating only 5 decision trees"""
print(classifier.fit(X_train, Y_train))

#predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

#predicting the test set results
Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))


#making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(accuracy_score(Y_test,Y_pred))

#Visualizing the trees
import PIL
import pydotplus
from glob import glob
from IPython.display import display,Image
from sklearn.tree import export_graphviz

def save_trees_as_png(clf, iteration, feature_name, target_name):
    file_name = "Purchased_" + str(iteration) + ".png"
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_name,
        class_names=target_name,
        rounded=True,
        proportion=False,
        precision=2,
        filled=True,
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(file_name)
    print("Decision Tree {} saved as png file".format(iteration + 1))

    col = dataset.columns.tolist()
    feature_names = col[:2]
    target_names = col[2]

    print(target_names)
    print(classifier.estimators_)
    print(classifier.estimators_[0])

    for i in range(len(classifier.estimators_)):
        save_trees_as_png(classifier.estimators_[i], i, feature_names, target_names)
        images = [PIL.Image.open(f) for f in glob('./*.png')]
        print(images)
        for im in images:
            print(display(Image(filename=im.filename, retina=True)))




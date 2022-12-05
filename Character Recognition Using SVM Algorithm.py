#Character Recognition Using SVM

#Loading the dataset
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

Digits = datasets.load_digits()

#Generating the Model
clf = svm.SVC(gamma=0.001, C=100)
X,Y = Digits.data[:-10],Digits.target[:-10]
clf.fit(X,Y)

print(clf.predict(Digits.data[:-10]))
plt.imshow(Digits.images[9], interpolation='nearest')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets._samples_generator import make_blobs

"""We create 40 separate variables"""
X,y = make_blobs(n_samples=40,centers=2,random_state=20)

"""Fit the model, don't regularize for illustration purpose"""
clf = svm.SVC(kernel = 'linear',C=1)
clf.fit(X,y)

"""Display the data in the graphical form"""
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)
plt.show()

"""Using to predict unknown data"""
newData = [[3,4],[5,6]]
print(clf.predict(newData))

"""Fit the model, don't regularize for illustration"""
clf = svm.SVC(kernel='linear',C=1000)
clf.fit(X,y)
plt.scatter(X[:,0], X[:,1], c=y, s=30,cmap=plt.cm.Paired)
#plt.show

"""Plot the decision function"""
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
"""create grid to evaluate model"""
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY,XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(),YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

"""Plot decision boundary and margins"""
ax.contour(XX,YY,Z, color='k',levels=[-1,0,1],
    alpha=0.5,
    linestyles=['--','-','--'])
"""Plot Support Vectors"""
ax.scatter(clf.support_vectors_[:,0],
    clf.support_vectors_[:,1],s=100,
    linewidth=1,facecolors='none')
plt.ylabel("Size of the Body")
plt.xlabel("Width of the Snout")
plt.show()
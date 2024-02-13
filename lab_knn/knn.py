from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

irisData = load_iris()

feature_names = irisData.feature_names
print("Feature names: ", feature_names)

target_names = irisData.target_names
print("Target names: ", target_names)

X = irisData.data
y = irisData.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=44)

neighbors = np.arange(1,9)
train_accurancy = np.empty(len(neighbors))
test_accurancy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)

    train_accurancy[i] = knn.score(X_train,y_train)
    test_accurancy[i] = knn.score(X_test,y_test)

plt.plot(neighbors, test_accurancy, label='Testing dataset accurancy')
plt.plot(neighbors, train_accurancy, label='Trainging dataset accurancy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accurancy')
plt.show()

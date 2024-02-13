from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cancer_data = datasets.load_breast_cancer()

print("Data Point : ", cancer_data.data[5])
print("Data Shape : ", cancer_data.data.shape)
print("Target Value : ", cancer_data.target)

X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=109)

cls = svm.SVC(kernel="linear")
cls.fit(X_train,y_train)
pred = cls.predict(X_test)

# Calculate accurancy
accuracy = metrics.accuracy_score(y_test, y_pred=pred)
print("Accuracy : ", accuracy)

# Calculate precision
precision = metrics.precision_score(y_test, y_pred=pred)
print("Precision : ", precision)

# Calculate recall
recall = metrics.recall_score(y_test, y_pred=pred)
print("Recall : ", recall)

print(metrics.classification_report(y_test, y_pred=pred))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:/Users/KimThanachok/Workspace/Machine-Learning-Lab/lab4/possum.csv")
print(df.sample(5, random_state=44))

df = df.dropna()

X = df.drop(["case","site","Pop","sex"], axis=1)
y = df["sex"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=44)

rf_model = RandomForestClassifier(n_estimators=50,max_features=44)
rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)
print(predictions)

predictprob = rf_model.predict_proba(X_test)
print(predictprob)

importances = rf_model.feature_importances_
print(importances)

columns = X.columns
i=0
while i<len(columns):
    print(f"The impotance of feature'{columns[i]}' is {round(importances[i] * 100, 2)}%. ")
    i+=1

new_possum = [[8.0,94.1,60.4,89.0,36.0,74.5,54.5,15.2,28.0,36]]
print(rf_model.predict(new_possum))
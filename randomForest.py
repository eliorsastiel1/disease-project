import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def getDiseases(df):
    x = df.get('disease').unique()
    return x

def RandomForest(X_train, X_test, y_train, y_test):

    trees = RandomForestClassifier(n_estimators=50)
    trees.fit(X_train, y_train)
    y_pred = trees.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


df = pd.read_csv("reducedDataset.csv")
y = df[df.columns[1]].values
X = df.drop(['disease', 'index'], axis=1).values
y = [item for item in y]
featureNames =df.columns[2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


RandomForest(X_train, X_test, y_train, y_test)

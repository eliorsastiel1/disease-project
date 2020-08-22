from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split

def convertDiseaseToIndex(diseases):
    maxIndex = 0
    dic={}
    indexDiseaseArray = []
    for disease in diseases:
        if disease not in dic:
            dic[disease] = maxIndex
            maxIndex += 1

    for disease in diseases:
        i=dic.get(disease)
        indexDiseaseArray.append(i)

    return indexDiseaseArray


def LRegression(X_train, X_test, y_train, y_test):

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


df = pd.read_csv("reducedDataset.csv")
y = df[df.columns[1]].values
X = df.drop(['disease', 'index'], axis=1).values
y = [item for item in y]
y = convertDiseaseToIndex(y)
featureNames =df.columns[2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


LRegression(X_train, X_test, y_train, y_test)



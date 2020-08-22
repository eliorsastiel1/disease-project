import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

import xgboost as xgb



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


def xgboost(X_train, X_test, y_train, y_test):

    train=xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)

    param={
        'max_depth':15,
        'eta': 0.15,
        'objective':'multi:softmax',
        'num_class': 41
    }

    epochs=10

    model = xgb.train(param,train, epochs)
    prediction = model.predict(test)
    print(metrics.accuracy_score(y_test, prediction))


df = pd.read_csv("reducedDataset.csv")
y = df[df.columns[1]].values
X = df.drop(['disease', 'index'], axis=1).values
y = [item for item in y]
y = convertDiseaseToIndex(y)
featureNames =df.columns[2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


xgboost(X_train, X_test, y_train, y_test)
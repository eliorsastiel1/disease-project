import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
def knn(X_train, X_test, y_train, y_test,showPlot=False):
    print('started KNN')
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, test_size=0.3)
    score = 0
    x = []
    y = []
    knnMax=None
    for i in range (2,50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = knn.score(X_test, y_test)
        #print('accurate score: {}'.format(score))
        x.append(i)
        y.append(score*100)
        if i==2:
          knnMax  =knn
        elif y[i-2]>y[i-3]:
            knnMax  =knn
    if showPlot:
      plt.plot(x, y)
      plt.xlabel('x - Number of neighbors')
      plt.ylabel('y - Score')
      plt.title('Knn')
      plt.show()
      return
    y_pred = knnMax.predict(X_validate)
    score = knnMax.score(X_validate, y_validate)
    return knnMax,score
    


df = pd.read_csv("reducedDataset.csv")
y = df[df.columns[0]].values
X = df.drop(['disease'], axis=1).values
y = [item for item in y]
featureNames =df.columns[2:]

#knn(X_train, X_test, y_train, y_test,showPlot=True)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.1)
y = np.array(y)
print(y)
splits=5

kf = KFold(n_splits=splits)
#kf.get_n_splits(X)
finalscore=0
last_score=0
best_model=None
for train_index, test_index in kf.split(X):
  X_train, X_validate = X[train_index], X[test_index]
  y_train, y_validate = y[train_index], y[test_index]
  model,score=knn(X_train, X_validate, y_train, y_validate,showPlot=False)
  if score>last_score:
    best_model=model
    last_score=score
  finalscore=finalscore+score
print(finalscore/splits)
print(best_model.score(X_test, y_test))





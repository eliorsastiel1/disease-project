import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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
    return score
    


df = pd.read_csv("reducedDataset.csv")
y = df[df.columns[0]].values
X = df.drop(['disease'], axis=1).values
y = [item for item in y]
featureNames =df.columns[2:]

#knn(X_train, X_test, y_train, y_test,showPlot=True)

finalscore=0
kfolds=5
for fold in range(0,kfolds):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  finalscore=finalscore+knn(X_train, X_test, y_train, y_test,showPlot=False)
print(finalscore/kfolds)




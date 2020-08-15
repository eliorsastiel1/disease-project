import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def knn(X_train, X_test, y_train, y_test):
    print('started KNN')
    score = 0
    x = []
    y = []
    for i in range (2,50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = knn.score(X_test, y_test)
        #print('accurate score: {}'.format(score))
        x.append(i)
        y.append(score*100)

    plt.plot(x, y)
    plt.xlabel('x - Number of neighbors')
    plt.ylabel('y - Score')
    plt.title('Knn')
    plt.show()


df = pd.read_csv("reducedDataset.csv")
y = df[df.columns[1]].values
X = df.drop(['disease', 'index'], axis=1).values
y = [item for item in y]
featureNames =df.columns[2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


knn(X_train, X_test, y_train, y_test)

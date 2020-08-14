
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import neighbors
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import lda
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from IPython.display import Image
import xgboost as xgb
from sklearn.metrics import  accuracy_score

def knn(X_train, X_test, y_train, y_test):
    print('started KNN')
    score = 0

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)

    print('accurate score: {}'.format(score))



    # ax = plt.subplot()
    # sn.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 10})
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)
    # ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    # plt.title('Confusion Matrix - KNN', fontsize = 20)
    # ax.xaxis.set_ticklabels(['1','2','3']);ax.yaxis.set_ticklabels(['1','2','3'])
    # plt.tight_layout()
    # plt.show()


def xgboost(X_train, X_test, y_train, y_test, featureNames):

    train=xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)

    param={
        'max_depth':4,
        'eta': 0.3,
        'objective':'multi:softmax',
        'num_class': 3
    }

    epochs=10

    model = xgb.train(param,train, epochs)
    prediction = model.predict(test)
    print(accuracy_score(y_test, prediction))


df = pd.read_csv("converted_data.csv")
y = df[df.columns[0]].values
X = df.drop('disease', axis=1).values
y = [item for item in y]
featureNames =df.columns[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn(X_train, X_test, y_train, y_test)
#DecisionTree(X_train, X_test, y_train, y_test, featureNames)
#xgboost(X_train, X_test, y_train, y_test, featureNames)
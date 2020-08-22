import pandas as pd
import csv
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def calculate_Tree_match(X_train, X_test, y_train, y_test,depth=12,calc_depth=False):
    if calc_depth:
        train_accuracy = []
        test_accuracy = []
        for i in range (2,depth):
            clf = tree.DecisionTreeClassifier(max_depth=i)
            clf = clf.fit(X_train, y_train)
            y_predict =clf.predict(X_test) 
            y_train_predict =clf.predict(X_train) 
            test_accuracy.append(accuracy_score(y_test, y_predict))
            train_accuracy.append(accuracy_score(y_train, y_train_predict))      
        plt.plot(range (2,depth),test_accuracy,label='Test')
        plt.plot(range (2,depth),train_accuracy,label='Train')
        plt.legend()
        plt.title('Decision tree accuracy as a function of depth')
        plt.xlabel('Tree depth')
        plt.ylabel('Accuracy')
        
        plt.show()
    else:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X_train, y_train)
        y_predict =clf.predict(X_test)
        class_names = clf.classes_
        fig = plt.figure(figsize=(22,10))
        tree.plot_tree(clf,max_depth=3,filled=True,fontsize=6,class_names=class_names,proportion=True)
        plt.show()
        acc_score = accuracy_score(y_test, y_predict)
        print('Accuracy Score: {0:f}'.format(acc_score))
        conf_matrix = confusion_matrix(y_test, y_predict)
        labels = ['business', 'health']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf_matrix)
        ax.xaxis.set_ticks_position('bottom')
        plt.title('Confusion matrix of the classifier')
        #ttl = ax.title
        #ttl.set_position([.5, 1.2])
        #ax.set_title('Confusion matrix of the classifier', pad=60)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names,fontsize=8,rotation='vertical')
        ax.set_yticklabels([''] + class_names,fontsize=8)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        #plt.show()
        #print_cm(conf_matrix,class_names)



dataset_reduced = pd.read_csv("reducedDataset.csv")

Y=dataset_reduced['disease']
X=dataset_reduced.loc[:, dataset_reduced.columns != 'disease']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#calculate_Tree_match(X_train, X_test, y_train, y_test,depth=40,calc_depth=True)
calculate_Tree_match(X_train, X_test, y_train, y_test,depth=12,calc_depth=False)



import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from collections import Counter

dataset = pd.read_csv('converted_data.csv')
Y=dataset['disease']
print(Y.unique().shape)
X=dataset.loc[:, dataset.columns != 'disease']


def PCA_Analysis():

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X.values)

    df=X.copy()
    df['y'] = Y
    df['label'] = df['y'].apply(lambda i: str(i))

    print('Size of the dataframe: {}'.format(df.shape))

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


    print(df['pca-one'][0])

    plt.figure(figsize=(16,10))
    p1=sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 41),
        data=df,
        legend="full",
        alpha=0.3
    )

    #for dot in range(0,df.shape[0]):
    #     p1.text(df['pca-one'][dot], df['pca-two'][dot], df['label'][dot], horizontalalignment='left', size='medium', color='black', weight='semibold')

    plt.title('PCA colored by disease')
    plt.legend(ncol=2,loc='upper right',fontsize='x-small')
    plt.show()


def TSNE_Analysis():
    diseases = Counter()
    idxs=[]
    # TSNE_Analysis()
    for i, disease in enumerate(Y.unique()):
        if diseases[disease] == 0:
            mask = Y == disease
            idx = next(iter(mask.index[mask]), 'not exist')
            idxs.append(idx)
            # print(disease)
            diseases[disease] = 1

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X.values)

    df = X.copy()
    df['y'] = Y
    df['label'] = df['y'].apply(lambda i: str(i))

    print('Size of the dataframe: {}'.format(df.shape))

    df['tsne-first-component'] = tsne_results[:, 0]
    df['tsne-second-component'] = tsne_results[:, 1]


    plt.figure(figsize=(16, 10))
    p1 = sns.scatterplot(
        x="tsne-first-component", y="tsne-second-component",
        hue="y",
        palette=sns.color_palette("hls", 41),
        data=df,
        legend=False,
        alpha=0.3
    )

    for dot in idxs:
        p1.text(df['tsne-first-component'][dot], df['tsne-second-component'][dot], df['label'][dot], horizontalalignment='left', size='medium', color='black', weight='semibold')

    plt.title('tSNE colored by disease')
    plt.legend(ncol=2, loc='upper right', fontsize='x-small')
    plt.show()


#PCA_Analysis()
#TSNE_Analysis()
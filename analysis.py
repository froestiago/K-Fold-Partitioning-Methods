from random import random
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from kfoldmethods.splitters.CBDSCV import CBDSCVSplitter
import timeit


def plot_ds(X, y):
    classes = np.unique(y)
    fig, ax = plt.subplots()
    for c in classes:
        ax.scatter(X[y==c, 0], X[y==c, 1], label=c)
    plt.show()


def plot_folds(X, folds):
    fig, ax = plt.subplots()
    for fold, indices in enumerate(folds):
        ax.scatter(X[indices, 0], X[indices, 1], label=fold)
    plt.legend()
    plt.show()


def main():
    n_samples = 20000
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=5, random_state=123)
    n_folds = 10
    n_clusters = round(n_samples / n_folds)
    print(n_clusters)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=123)
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    
    
    start = timeit.default_timer()
    y_pred = agg_cluster.fit_predict(X)
    end = timeit.default_timer()

    print("Elapsed time for agg cluster: %.2f" % (end - start))

    start = timeit.default_timer()
    y_pred = kmeans.fit_predict(X)
    end = timeit.default_timer()
    print("Elapsed time for kmeans: %.2f" % (end - start))


    start = timeit.default_timer()
    y_pred = minibatch_kmeans.fit_predict(X)
    end = timeit.default_timer()
    print("Elapsed time for minibatch_kmeans: %.2f" % (end - start))

    # plot_ds(X, y_pred)
    


if __name__ == "__main__":
    main()
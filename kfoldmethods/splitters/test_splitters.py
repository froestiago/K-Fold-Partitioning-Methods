import numpy as np
import matplotlib.pyplot as plt
from kfoldmethods.splitters.DBSVC import dbsvc


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


def test_dbscv():
    X = [
        [0.05, 0.1], [0.1, 0.1], [0.16, 0.1], [0.23, 0.1], [0.31, 0.1], [0.41, 0.1], 
        [0.515, 0.1], [0.625, 0.1], [0.74, 0.1], [0.86, 0.1],
        [0.70, 0.85], [0.73, 0.85], [0.77, 0.85], [0.82, 0.85], [0.88, 0.85]]
    X = np.array(X)

    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y = np.array(y)

    # plot_ds(X, y)

    rng = np.random.RandomState(0)
    folds = dbsvc(X, y, k=5, rng=rng)
    print(folds)
    plot_folds(X, folds)
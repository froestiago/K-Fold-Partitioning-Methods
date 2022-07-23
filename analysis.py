from pathlib import Path
from random import random
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from pmlb import fetch_data

from kfoldmethods.datasets import pmlb_api
from kfoldmethods.experiments import configs
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


def datasets_info():
    df_datasets = pd.read_csv("kfoldmethods/datasets/pmlb_datasets.csv")
    selected = configs.datasets_balanced
    path_dataset_info = Path(configs.dataset_info__output_dir)
    path_dataset_info.mkdir(exist_ok=True, parents=True)
    
    df_selected = df_datasets.loc[
        df_datasets['Dataset'].isin(selected), 
        ['Dataset', 'n_observations', 'n_features', 'n_classes', 'Imbalance']]
    df_selected = df_selected.sort_values(by='Dataset')
    df_selected.to_csv(Path(configs.dataset_info__output_dir) / 'datasets_balanced.csv', index=False)
    print(df_selected)

    selected_imb = configs.datasets_imb
    df_selected = df_datasets.loc[
        df_datasets['Dataset'].isin(selected_imb), 
        ['Dataset', 'n_observations', 'n_features', 'n_classes', 'Imbalance']]
    df_selected = df_selected.sort_values(by='Dataset')
    df_selected.to_csv(Path(configs.dataset_info__output_dir) / 'datasets_imbalanced.csv', index=False)
    print(df_selected)

    selected = configs.datasets
    df_selected = df_datasets.loc[
        df_datasets['Dataset'].isin(selected), 
        ['Dataset', 'n_observations', 'n_features', 'n_classes', 'Imbalance']]
    df_selected = df_selected.sort_values(by='Dataset')
    df_selected.to_csv(Path(configs.dataset_info__output_dir) / 'datasets.csv', index=False)
    print(df_selected)


def min_instance_class():
    selected = configs.datasets

    for ds_name in selected:
        X, y = fetch_data(ds_name, return_X_y=True)
        _, n_per_class = np.unique(y, return_counts=True)
        
        small_class = min(n_per_class)
        print("{}: {}".format(ds_name, small_class))


if __name__ == "__main__":
    datasets_info()

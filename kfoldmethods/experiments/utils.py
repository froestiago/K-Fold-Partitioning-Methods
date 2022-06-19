from pathlib import Path
import numpy as np
from sklearn import clone
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kfoldmethods.experiments import configs


def load_best_classifier_for_dataset(
        ds, clf_class_name, hp_dir='run_data/classifier_hyperparameters'):
    path_search = Path(hp_dir) / ds / clf_class_name / 'search.joblib'
    search = joblib.load(path_search)
    return clone(search.best_estimator_)


def estimate_n_clusters(X, n_iters=50, sample_size=None, random_state=123, return_all=False):
    """Estimate number of clusters based on agglomerative clustering.

    The number of clusters is estimated by retaining only merges that had a distance
    smaller than a threshold. The threshold is the mean distance plus two times the 
    standard deviation of the distances.

    This process is repeated multiple times with small sample sizes.
    """
    if sample_size is None:
        sample_size = min(100, X.shape[0] - 1)
    
    splitter = ShuffleSplit(n_splits=n_iters, train_size=sample_size, random_state=random_state)
    
    n_clusters_list = []
    for train, test in splitter.split(X):
        model = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage='average', 
            compute_full_tree=True)
        model = model.fit(X[train])
        distances = model.distances_

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # stop when distance > mean + 2*std
        th = mean_distance + 2*std_distance
        n_clusters = np.sum(1*(distances > th)) + 1
        n_clusters_list.append(n_clusters)

    if return_all:
        return n_clusters_list
    
    return np.mean(n_clusters_list), np.std(n_clusters_list)


def bootstrap_step(X: np.ndarray, random_state=123):
    n_samples = X.shape[0]
    indices = np.arange(0, n_samples)
    indices_r = resample(indices, n_samples=n_samples, random_state=random_state)
    return indices_r


def g_name_to_str(g_name):
    return "_".join([str(s) for s in g_name])


def _compare_plot_overall(df):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for i, (g_name, g) in enumerate(groups):
        print(g_name)
        fig, ax = plt.subplots()
        sns.violinplot(data=g, y='splitter_method', x='bias', ax=ax)
        fig.savefig('figs/bias_splitter_{}'.format(g_name_to_str(g_name)))
        fig.tight_layout()
        plt.close(fig)


def _compare_plot_std_overall(df):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for i, (g_name, g) in enumerate(groups):
        print(g_name)
        fig, ax = plt.subplots()
        sns.violinplot(data=g, y='splitter_method', x='estimate_std', ax=ax)
        fig.savefig('figs/std_splitter_{}'.format(g_name_to_str(g_name)))
        fig.tight_layout()
        plt.close(fig)


def _compare_plot_balance(df):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for g_name, g in groups:
        fig, ax = plt.subplots()
        g_balanced = g.loc[g['dataset_name'].isin(configs.datasets_balanced), :]
        sns.violinplot(data=g_balanced, y='splitter_method', x='bias', ax=ax)
        fig.tight_layout()
        fig.savefig('figs/balanced_bias_splitter_{}'.format(g_name_to_str(g_name)))
        plt.close(fig)

        fig, ax = plt.subplots()
        g_imbalanced = g.loc[g['dataset_name'].isin(configs.datasets_imb), :]
        sns.violinplot(data=g_imbalanced, y='splitter_method', x='bias', ax=ax)
        fig.tight_layout()
        fig.savefig('figs/imbalanced_bias_splitter_{}'.format(g_name_to_str(g_name)))
        plt.close(fig)


def _compare_plot_std_balance(df):
    groups = df.groupby(by=['metric_name', 'n_splits'])
    
    for g_name, g in groups:
        fig, ax = plt.subplots()
        g_balanced = g.loc[g['dataset_name'].isin(configs.datasets_balanced), :]
        sns.violinplot(data=g_balanced, y='splitter_method', x='estimate_std', ax=ax)
        fig.tight_layout()
        fig.savefig('figs/balanced_std_splitter_{}'.format(g_name_to_str(g_name)))
        plt.close(fig)

        fig, ax = plt.subplots()
        g_imbalanced = g.loc[g['dataset_name'].isin(configs.datasets_imb), :]
        sns.violinplot(data=g_imbalanced, y='splitter_method', x='estimate_std', ax=ax)
        fig.tight_layout()
        fig.savefig('figs/imbalanced_std_splitter_{}'.format(g_name_to_str(g_name)))
        plt.close(fig)

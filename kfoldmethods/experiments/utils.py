from pathlib import Path
import numpy as np
from sklearn import clone
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample


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

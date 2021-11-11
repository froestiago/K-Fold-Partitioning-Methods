import numpy as np
from pandas.core.indexing import need_slice
from sklearn.utils import indexable, check_random_state, shuffle
from common_func import circular_append
from sklearn.cluster import KMeans
import copy

from sklearn.datasets import load_digits


def CBDSCV(X, y, k, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    k_clusters = len(np.unique(y))  # extrating k, the k that will be used on clustering, from y

    kmeans = KMeans(n_clusters=k_clusters)
    X_new = kmeans.fit_transform(X)  # does not allow to choose the metric for distance

    cluster_index = np.argsort(X_new)
    cluster_index = [i[0] for i in cluster_index]

    clusters = [[] for _ in range(k_clusters)]  # list with k clusters (empty)
    i, size = 0, len(X_new)

    while i < size:
        element = (i, X_new[i][cluster_index[i]])
        clusters[cluster_index[i]].append(element)
        i += 1

    index_list = []
    dtype = [('index', int), ('distance', float)]
    for values in clusters:
        each_cluster = np.array(values, dtype=dtype)
        each_cluster = np.sort(each_cluster, order='distance')
        index_list.extend(each_cluster['index'])

    folds = [[] for _ in range(k)]
    folds = circular_append(index_list, folds, k)

    return folds


class CBDSCVSplitter:
    def __init__(self, n_splits=5, random_state=None, shuffle=True):
        """Split dataset indices according to the CBDSCV technique.

        Parameters
        ----------
        n_splits : int
            Number of splits to generate. In this case, this is the same as the K in a K-fold cross validation.
        random_state : any
            Seed or numpy RandomState. If None, use the singleton RandomState used by numpy.
        shuffle : bool
            Shuffle dataset before splitting.
        """
        # in sklearn, generally, we do not check the arguments in the initialization function.
        # There is a reason for this.
        self.n_splits = n_splits
        self.random_state = random_state  # used for enabling the user to reproduce the results
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        """Generate indices to split data according to the DBSCV technique.

        Parameters
        ----------
        X : array-like object of shape (n_samples, n_features)
            Training data.
        y : array-like object of shape (n_samples, )
            Target variable corresponding to the training data.
        groups : None
            Not implemented. It is here for compatibility.

        Yields
        -------
            Split with train and test indexes.
        """
        if groups:
            raise NotImplementedError("groups functionality is not implemented.")

        # just some validations that sklearn uses
        X, y = indexable(X, y)
        rng = check_random_state(self.random_state)

        if self.shuffle:
            X, y = shuffle(X, y, random_state=rng)

        folds = CBDSCV(X, y, self.n_splits, rng=rng)

        for k in range(self.n_splits):
            test_fold_index = self.n_splits - k - 1  # start by using the last fold as the test fold
            ind_train = []
            ind_test = []
            for fold_index in range(self.n_splits):
                if fold_index != test_fold_index:
                    ind_train += copy.copy(folds[fold_index])
                else:
                    ind_test = copy.copy(folds[fold_index])

            yield ind_train, ind_test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# def main():
#     blob_centers = np.array(
#     [[ 0.2,  2.3],
#      [-1.5 ,  2.3],
#      [-2.8,  1.8],
#      [-2.8,  2.8],
#      [-2.8,  1.3]])
#     blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

#     # X, y = make_blobs(n_samples=50, centers=blob_centers,
#     # cluster_std=blob_std, random_state=7)
#     X, y = load_digits(return_X_y=True)

#     CBDSCV(X, y)


# if __name__ == '__main__':
#     main()
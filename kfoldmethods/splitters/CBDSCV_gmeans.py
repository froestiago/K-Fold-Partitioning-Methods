import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import indexable, check_random_state, shuffle
from pyclustering.cluster.gmeans import gmeans
import copy

from .utils import circular_append


def CBDSCV_gmeans(X, y, rng=None, bad_case = False):
    if rng is None:
        rng = np.random.RandomState()

    gmeans_instance = gmeans(X, repeat = 10).process()

    clusters = gmeans_instance.get_clusters()
    centers = gmeans_instance.get_centers()

    k = len(clusters)
    index_list = []

    for i in range(k):
        cluster_indexes = clusters[i]
        cluster_instances = X[clusters[i]]
        cluster_center = centers[i]

        distances = euclidean_distances(cluster_instances, [cluster_center])
        distances = distances.reshape(-1)
        sorted_index_distances = distances.argsort()

        for x in sorted_index_distances:index_list.append(cluster_indexes[x])


    folds = [[] for _ in range(k)] #list with kfolds (empty)
    folds = circular_append(index_list, folds, k)

    return folds, k

class CBDSCV_gmeansSplitter:
    def __init__(self, random_state=None, shuffle=True, bad_case=False):
        """Split dataset indices according to the DBSCV technique.

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
        #self.n_splits = n_splits
        self.random_state = random_state  # used for enabling the user to reproduce the results
        self.shuffle = shuffle
        self.bad_case = bad_case

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

        if self.bad_case == False:
            folds, n_splits = CBDSCV_gmeans(X, y, rng=rng, bad_case=False)
        else:
            folds, n_splits = CBDSCV_gmeans(X, y, rng=rng, bad_case=True)


        for k in range(n_splits):
            test_fold_index = n_splits - k - 1  # start by using the last fold as the test fold
            ind_train = []
            ind_test = []
            for fold_index in range(n_splits):
                if fold_index != test_fold_index:
                    ind_train += copy.copy(folds[fold_index])
                else:
                    ind_test = copy.copy(folds[fold_index])

            yield ind_train, ind_test

    '''def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits'''
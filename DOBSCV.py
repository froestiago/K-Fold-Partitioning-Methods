# import numpy as np
# from pandas.core.indexing import need_slice

# from scipy.spatial.kdtree import distance_matrix
# from sklearn.datasets import make_blobs
# from sklearn.metrics import pairwise_distances
# from common_func import aux_indexes, aux_indexes, sort_by_label, circular_append, data_slicing_by_label

import numpy as np
from pandas.core.indexing import need_slice

from sklearn.metrics import pairwise_distances
from sklearn.utils import indexable, check_random_state, shuffle
from common_func import aux_indexes, sort_by_label, circular_append, data_slicing_by_label
import copy

def dobscv(X, y, k, rng=None, bad_case = False):
    if rng is None:
        rng = np.random.RandomState()

    #sort dataset by labels
    X, y = sort_by_label(X, y)

    slicing_index, segment_shift, start_of_segment, end_of_segment = aux_indexes(y)

    X, y = data_slicing_by_label(X, y, slicing_index)

    index_list = [] #to be returned

    for each_class in X:
        distance_matrix = pairwise_distances(each_class, metric='euclidean')
        i = len(each_class)
        possible_rand = np.arange(len(distance_matrix[0]))
        while i > 0:
            n = k
            k_closest = []
            chosen_instance = np.random.choice(possible_rand)
            sorted_distances = np.sort(distance_matrix[chosen_instance])
            
            if bad_case == True:
                sorted_distances = sorted_distances[::-1]

            
            index_sorted_distances = np.argsort(distance_matrix[chosen_instance])

            if len(possible_rand) < k:
                n = len(possible_rand)

            k_closest.append(chosen_instance)
            k_closest.extend(index_sorted_distances[1:n])

            for zero_column_line in k_closest:
                distance_matrix[zero_column_line, :] = np.nan
                distance_matrix[:, zero_column_line] = np.nan
            
            possible_rand = np.setdiff1d(possible_rand, k_closest)

            index_list.extend(k_closest)
            i -= n

    #sum segment_shift on each position
    for i, j, x in zip(start_of_segment, end_of_segment, segment_shift):
        index_list[i:j] += x

    folds = [[] for _ in range(k)] #list with kfolds (empty)
    folds = circular_append(index_list, folds, k)

    return folds

class DOBSCVSplitter:
    def __init__(self, n_splits=5, random_state=None, shuffle=True, bad_case=False):
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
        self.n_splits = n_splits
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
            folds = dobscv(X, y, self.n_splits, rng=rng, bad_case=False)
        else:
            folds = dobscv(X, y, self.n_splits, rng=rng, bad_case=True)


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
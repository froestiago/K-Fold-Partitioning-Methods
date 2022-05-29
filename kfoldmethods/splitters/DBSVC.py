import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import indexable, check_random_state, shuffle
import copy

from .utils import aux_indexes, sort_by_label, circular_append, data_slicing_by_label


def dbsvc(X, y, k, bad_case = False, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    #sort dataset by labels
    X, y = sort_by_label(X, y)

    slicing_index, segment_shift, start_of_segment, end_of_segment = aux_indexes(y)

    X, y = data_slicing_by_label(X, y, slicing_index)

    i = 0
    index_list = [] #to be returned

    for j, each_class in enumerate(X):
        distance_matrix = pairwise_distances(each_class, metric='euclidean')
        np.fill_diagonal(distance_matrix, val=-1)

        # smallest_distance_index = 1 #to test
        smallest_distance_index = (rng.randint(len(distance_matrix[0])))
        index_list.append(smallest_distance_index)
        i = 0
        while i < len(distance_matrix[0])-1:  # while i < n_instances_in_class - 1
            zero_index = smallest_distance_index
            if bad_case == False:
                # get index of instance with smallest distance from previous instance
                # masked_where masks (omits) elements where first argument is true
                smallest_distance_index = np.argmin(
                    np.ma.masked_where(distance_matrix[smallest_distance_index] < 0, distance_matrix[smallest_distance_index]))
            else:
                smallest_distance_index = np.argmax(
                    np.ma.masked_where(distance_matrix[smallest_distance_index] < 0, distance_matrix[smallest_distance_index]))
                
            distance_matrix[zero_index, :] = -1
            distance_matrix[:, zero_index] = -1
            index_list.append(smallest_distance_index)
            i += 1

    #sum segment_shift on each position
    for i, j, x in zip(start_of_segment, end_of_segment, segment_shift):
        index_list[i:j] += x

    folds = [[] for _ in range(k)] #list with kfolds (empty)
    folds = circular_append(index_list, folds, k)
    return(folds) #return indexes


class DBSCVSplitter:
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
            folds = dbsvc(X, y, self.n_splits, rng=rng, bad_case=False)
        else:
            folds = dbsvc(X, y, self.n_splits, rng=rng, bad_case=True)

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




import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.utils import indexable, check_random_state, shuffle
import copy


def sortByLabel(X, y):
    sortedIndex = y.argsort()
    return X[sortedIndex], y[sortedIndex]

def circuilarAppend(inList, outList, k):
    i = 0
    for x in inList:
        outList[i].append(x)
        if i < k-1:
            i += 1
        else:
            i = 0
    return(outList)

def dataSlicing(X, y, indexes):
    XOut = np.split(X, indexes)
    XOut = XOut[:-1]
    yOut = np.split(y, indexes)
    yOut = yOut[:-1]

    return XOut, yOut

def dbsvc(X, y, k, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    #sort dataset by labels
    X, y = sortByLabel(X, y)
    
    counts = np.unique(y, return_counts=True)
    counts = counts[1]
    counts = np.cumsum(counts)
    indexShift = np.append(0, counts[:-1])
    outIndex = []
    #print(counts)
    
    # X, y = dataSlicing(X, y, counts) 

    distanceMatrix = pairwise_distances(X, metric='euclidean')

    i = 0
    indexList = []
    while i < len(indexShift):
        start = indexShift[i]
        finish = counts[i]
        subMatrix = distanceMatrix[start:finish, start:finish]
        j = 0
        # minIndex = 1 #to test
        minIndex = (rng.randint(len(subMatrix[0])))
        indexList.append(minIndex)
        while j < len(subMatrix[0])-1:
            zeroIndex = minIndex
            minIndex = np.argmin(np.ma.masked_where(subMatrix[minIndex] == 0, subMatrix[minIndex]))
            subMatrix[zeroIndex, :] = 0
            subMatrix[:, zeroIndex] = 0
            indexList.append(minIndex)
            j += 1
        i += 1


    counts = np.append(0, counts)
    start = counts[:-1]
    finish = counts[1:]

    #sum indexShift on each position
    for i, j, x in zip(start, finish, indexShift):
        indexList[i:j] += x

    folds = [[] for _ in range(k)] #list with kfolds (empty)
    folds = circuilarAppend(indexList, folds, k)

    return(folds) #return indexes


class DBSCVSplitter:
    def __init__(self, n_splits=5, random_state=None, shuffle=True):
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

        folds = dbsvc(X, y, self.n_splits, rng=rng)
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



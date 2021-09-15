import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances


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

def dbsvc(X, y, k):
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
        minIndex = (np.random.randint(len(subMatrix[0]))) 
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



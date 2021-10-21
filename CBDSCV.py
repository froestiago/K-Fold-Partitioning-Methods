import numpy as np
from sklearn.datasets import make_blobs, load_digits
from common_func import circular_append
from sklearn.cluster import KMeans, DBSCAN
np.random.seed(42)



def CBDSCV(X, y, k, metric):
    k_cluster = len(np.unique(y)) #extrating k, the k that will be used on clustering, from y

    kmeans = KMeans(n_clusters=k_cluster)
    X_new = kmeans.fit_transform(X) #does not allow to choose the metric for distance

    cluster_index = np.argsort(X_new)
    cluster_index = [i[0] for i in cluster_index]

    clusters = [[] for _ in range(k_cluster)] #list with k clusters (empty)
    i, size = 0, len(X_new)

    while i < size:
        element = (i, X_new[i][cluster_index[i]])
        clusters[cluster_index[i]].append(element)
        i+=1

    index_list = []
    dtype = [('index', int), ('distance', float)]
    for values in clusters:
        each_cluster = np.array(values, dtype=dtype)
        each_cluster = np.sort(each_cluster, order='distance')
        print('\nNEW')
        print(len(each_cluster))
        index_list.extend(each_cluster['index'])

    print(index_list)
    
    folds = [[] for _ in range(k)]
    folds = circular_append(index_list, folds, k)
    # print(folds)

    return folds

def main():
    blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    X, y = make_blobs(n_samples=50, centers=blob_centers, 
    cluster_std=blob_std, random_state=7)
    X, y = load_digits(return_X_y=True)

    CBDSCV(X, y, 10, 'euclidean')



if __name__ == '__main__':
    main()
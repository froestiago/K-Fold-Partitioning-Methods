import numpy as np
from scipy.spatial.kdtree import distance_matrix
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from common_func import aux_indexes, aux_indexes, sort_by_label, circular_append, data_slicing_by_label

np.random.RandomState(42)

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
        print('length: ', i)
        possible_rand = np.arange(len(distance_matrix[0]))
        print("NEW CLASS\n")
        while i > 0:
            print(distance_matrix)
            n = k
            print("\tnew index")
            k_closest = []
            chosen_instance = np.random.choice(possible_rand)
            print('STARTING INDEX = ', chosen_instance)
            
            sorted_distances = np.sort(distance_matrix[chosen_instance])
            index_sorted_distances = np.argsort(distance_matrix[chosen_instance])
            
            print('Sorted: ', sorted_distances)
            if len(possible_rand) < k:
                n = len(possible_rand)

            k_closest.append(chosen_instance)
            k_closest.extend(index_sorted_distances[1:n])

            for zero_column_line in k_closest:
                distance_matrix[zero_column_line, :] = np.nan
                distance_matrix[:, zero_column_line] = np.nan
            
            possible_rand = np.setdiff1d(possible_rand, k_closest)
            print('K - Closest: ', k_closest)
            print('Possible next element: ', possible_rand)
            #find how many nan there is


            index_list.extend(k_closest)
            i -= n

    #sum segment_shift on each position
    for i, j, x in zip(start_of_segment, end_of_segment, segment_shift):
        index_list[i:j] += x

    folds = [[] for _ in range(k)] #list with kfolds (empty)
    folds = circular_append(index_list, folds, k)
    
    print(folds)
    print(len(index_list))

def main():
    print("Testing DBSCVSplitter class...")

    random_state = np.random.RandomState(42)

    blob_centers = np.array(
        [[0.2, 2.3],  # y = 0
         [-1.5, 2.3],  # y = 1
         [-2.8, 1.8]])  # y = 2
    blob_std = np.array([0.4, 0.3, 0.1])

    X, y = make_blobs(n_samples=17, centers=blob_centers,
                      cluster_std=blob_std, shuffle=True, random_state=random_state)

    folds = dobscv(X, y, k = 3)
if __name__ == '__main__':
    main()
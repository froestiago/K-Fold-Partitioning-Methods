import DBSVC
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs


def main():
    #np.random.seed(42)

    blob_centers = np.array(
        [[ 0.2,  2.3],      #y = 0
         [-1.5 ,  2.3],     #y = 1
         [-2.8,  1.8],      #y = 2
         [-2.8,  2.8],      #y = 3
         [-2.8,  1.3]])     #y = 4
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    X, y = make_blobs(n_samples=25, centers=blob_centers,
                  cluster_std=blob_std, shuffle=True)

    
    folds = DBSVC.dbsvc(X, y, 5)
    print(folds)



if __name__ == '__main__':
    main()
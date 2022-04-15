import CBDSCV_gmeans
import DBSVC
import DOBSCV
import CBDSCV

import numpy as np
from sklearn.datasets import make_blobs, load_iris, load_digits, load_wine, load_breast_cancer, fetch_olivetti_faces
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from pmlb import fetch_data

import time

def timing_test():
    np.random.seed(42)

    blob_centers = np.array(
        [[ 0.2,  2.3, -1.5],    #y = 0
         [-1.5,  2.3, 1.8],     #y = 1
         [-2.8,  1.8, 1.3],     #y = 2
         [-2.8,  2.8, -2.8],    #y = 3
         [-2.8,  1.3, 2.3]])    #y = 4
    blob_std = np.array([0.7, 0.3, 0.6, 0.3, 0.2])

    X, y = make_blobs(n_samples=5000, centers=blob_centers,
                  cluster_std=blob_std, shuffle=True)
    
    startTimer = time.time()
    folds = DBSVC.dbsvc(X, y, 5)
    endTimer = time.time()
    totalTime = endTimer - startTimer
    print('O tempo de execução da função foi de: ', totalTime, 's')
    # print(folds)

def test_dbscv_splitter():
    print("Testing DBSCVSplitter class...")

    random_state = np.random.RandomState(0)

    blob_centers = np.array(
        [[0.2, 2.3],  # y = 0
         [-1.5, 2.3],  # y = 1
         [-2.8, 1.8],  # y = 2
         [-2.8, 2.8],  # y = 3
         [-2.8, 1.3]])  # y = 4
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    X, y = make_blobs(n_samples=15, centers=blob_centers,
                      cluster_std=blob_std, shuffle=True, random_state=random_state)

    print(X)
    
    splitter = DBSVC.DBSCVSplitter(n_splits=5, random_state=random_state)
    for ind_train, ind_test in splitter.split(X, y):
        print("Train indices: {} Test indices: {}".format(ind_train, ind_test))
    print("\nBAD CASE\n")
    splitter = DBSVC.DBSCVSplitter(n_splits=5, random_state=random_state, bad_case=True)
    for ind_train, ind_test in splitter.split(X, y):
        print("Train indices: {} Test indices: {}".format(ind_train, ind_test))

def test_dobscv_splitter():
    print("Testing DBSCVSplitter class...")

    random_state = np.random.RandomState(0)

    blob_centers = np.array(
        [[0.2, 2.3],  # y = 0
         [-1.5, 2.3],  # y = 1
         [-2.8, 1.8],  # y = 2
         [-2.8, 2.8],  # y = 3
         [-2.8, 1.3]])  # y = 4
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    X, y = make_blobs(n_samples=28, centers=blob_centers,
                      cluster_std=blob_std, shuffle=True, random_state=random_state)
    
    splitter = DOBSCV.DOBSCVSplitter(n_splits=5, random_state=random_state)
    for ind_train, ind_test in splitter.split(X, y):
        print("Train indices: {} Test indices: {}".format(ind_train, ind_test))

def test_cbdscv_splitter():
    print("Testing CBDSCVSplitter class")

    random_state = np.random.RandomState(0)

    blob_centers = np.array(
        [[0.2, 2.3],  # y = 0
         [-1.5, 2.3],  # y = 1
         [-2.8, 1.8],  # y = 2
         [-2.8, 2.8],  # y = 3
         [-2.8, 1.3]])  # y = 4
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    X, y = make_blobs(n_samples=500, centers=blob_centers,
                      cluster_std=blob_std, shuffle=True, random_state=random_state)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    splitter_dobscv = CBDSCV.CBDSCVSplitter(random_state=random_state)
    
    scores = cross_val_score(pipeline, X, y=y, cv=splitter_dobscv)
    print("-------------------")
    print("Results with DOBSCV:")
    print("Scores: ", scores)
    print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))

def test_cbdscv_gmeans_splitter():
    print("Testing CBD SCV gmeans Splitter class...")

    random_state = np.random.RandomState(42)

    blob_centers = np.array(
        [[0.2, 2.3],  # y = 0
         [-1.5, 2.3],  # y = 1
         [-2.8, 1.8],  # y = 2
         [-2.8, 2.8],  # y = 3
         [-2.8, 1.3]])  # y = 4
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    X, y = make_blobs(n_samples=5000, centers=blob_centers,
                      cluster_std=blob_std, shuffle=True, random_state=random_state)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    
    
    '''
    for ind_train, ind_test in splitter.split(X, y):
        print("Train indices: {} Test indices: {}".format(ind_train, ind_test))
    print("\nBAD CASE\n")
    '''
    splitter = CBDSCV_gmeans.CBDSCV_gmeansSplitter(random_state=random_state, bad_case=False)
    #for ind_train, ind_test in splitter.split(X, y):
    #    print("Train indices: {} Test indices: {}".format(ind_train, ind_test))
    
    scores = cross_val_score(pipeline, X, y=y, cv=splitter)
    print("-------------------")
    print("Results with CBD SCV gmeans:")
    print("Scores: ", scores)
    print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))

def sklearn_cv_example():
    print("Example of using the DBSCV splitter together with sklearn.")
    n_splits = 2
    X, y = load_iris(return_X_y=True)
    splitter_dbscv = DBSVC.DBSCVSplitter(n_splits=n_splits, random_state=0, shuffle=True)
    splitter_stratified_cv = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier())
    ])

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_dbscv)
    print("-------------------")
    print("Results with DBSCV:")
    print("Scores: ", scores)
    print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_stratified_cv)
    print("-------------------")
    print("Results with Stratified k-fold cross validation:")
    print("Scores: ", scores)
    print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))

def main():

    blob_centers = np.array(
        [[ 0.2,  2.3, -1.5],    #y = 0
         [-1.5,  2.3, 1.8],     #y = 1
         [-2.8,  1.8, 1.3],     #y = 2
         [-2.8,  2.8, -2.8],    #y = 3
         [-2.8,  1.3, 2.3]])    #y = 4
    blob_std = np.array([0.7, 0.3, 0.6, 0.3, 0.2])

    X, y = make_blobs(n_samples=10000, centers=blob_centers, cluster_std=blob_std, shuffle=True)   
    #X, y = load_digits(return_X_y = True)
    #X, y = fetch_data('mushroom', return_X_y=True)
    #X, y = load_wine(return_X_y=True)
    #X, y = load_iris(return_X_y=True)
    n_splits = 10

    bad_case_splitter_dbscv = DBSVC.DBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=True)
    splitter_dbscv = DBSVC.DBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False)
    
    bad_case_splitter_dobscv = DOBSCV.DOBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=True)
    splitter_dobscv = DOBSCV.DOBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False)

    splitter_stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

    splitter_cbdscv = CBDSCV.CBDSCVSplitter()

    splitter_cbdscv_gmeans = CBDSCV_gmeans.CBDSCV_gmeansSplitter(bad_case=False)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_cbdscv_gmeans)
    print("-------------------")
    print("Results with CBDSCV gmeans:")
    #print("Scores: ", scores)
    #print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print("STD: ", np.std(scores))

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_cbdscv)
    print("-------------------")
    print("Results with CBDSCV:")
    #print("Scores: ", scores)
    #print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print("STD: ", np.std(scores))

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_dobscv)
    print("-------------------")
    print("Results with DOBSCV:")
    #print("Scores: ", scores)
    #print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print("STD: ", np.std(scores))

    scores = cross_val_score(pipeline, X, y=y, cv=bad_case_splitter_dobscv)
    print("-------------------")
    print("Results with bad case DOBSCV:")
    #print("Scores: ", scores)
    #print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print("STD: ", np.std(scores))

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_dbscv)
    print("-------------------")
    print("Results with DBSCV:")
    #print("Scores: ", scores)
    #print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print("STD: ", np.std(scores))

    scores = cross_val_score(pipeline, X, y=y, cv=bad_case_splitter_dbscv)
    print("-------------------")
    print("Results with bad case DBSCV:")
    #print("Scores: ", scores)
    #print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print("STD: ", np.std(scores))

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_stratified_cv)
    print("-------------------")
    print("Results with Stratified k-fold cross validation:")
    #print("Scores: ", scores)
    #print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print("STD: ", np.std(scores))


if __name__ == '__main__':
    # timing_test()
    main()
    #test_dbscv_splitter()
    #test_dobscv_splitter()
    #test_cbdscv_splitter()
    #test_cbdscv_gmeans_splitter()

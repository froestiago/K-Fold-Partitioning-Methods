import time
import numpy as np
from sklearn.datasets import make_blobs, load_iris, load_wine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from ..splitters import DOBSCV, DBSVC, CBDSCV, CBDSCV_gmeans

def timing_test():
    np.random.seed(42)

    blob_centers = np.array(
        [[0.2, 2.3, -1.5],  # y = 0
         [-1.5, 2.3, 1.8],  # y = 1
         [-2.8, 1.8, 1.3],  # y = 2
         [-2.8, 2.8, -2.8],  # y = 3
         [-2.8, 1.3, 2.3]])  # y = 4
    blob_std = np.array([0.7, 0.3, 0.6, 0.3, 0.2])

    X, y = make_blobs(n_samples=5000, centers=blob_centers,
                      cluster_std=blob_std, shuffle=True)

    startTimer = time.time()
    folds = DBSVC.dbsvc(X, y, 5)
    endTimer = time.time()
    totalTime = endTimer - startTimer
    print('O tempo de execução da função foi de: ', totalTime, 's')

    # todo: make a test that has an expected result
    assert True


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

    # todo: make a test that has an expected result
    assert True


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

    # todo: make a test that has an expected result
    assert True


def test_cbdscv_splitter():
    print("Testing CBDSCVSplitter class")

    random_state = np.random.RandomState(0)

    # blob_centers = np.array(
    #     [[0.2, 2.3],  # y = 0
    #      [-1.5, 2.3],  # y = 1
    #      [-2.8, 1.8],  # y = 2
    #      [-2.8, 2.8],  # y = 3
    #      [-2.8, 1.3]])  # y = 4
    # blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    # X, y = make_blobs(n_samples=500, centers=blob_centers,
    #                   cluster_std=blob_std, shuffle=True, random_state=random_state)

    X, y = load_iris(return_X_y=True)    

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    splitter_dobscv = CBDSCV.CBDSCVSplitter(random_state=random_state, n_splits=5)

    scores = cross_val_score(pipeline, X, y=y, cv=splitter_dobscv)
    print("-------------------")
    print("Results with DOBSCV:")
    print("Scores: ", scores)
    print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))

    # todo: make a test that has an expected result
    assert True


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
    # for ind_train, ind_test in splitter.split(X, y):
    #    print("Train indices: {} Test indices: {}".format(ind_train, ind_test))

    scores = cross_val_score(pipeline, X, y=y, cv=splitter)
    print("-------------------")
    print("Results with CBD SCV gmeans:")
    print("Scores: ", scores)
    print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))

    # todo: make a test that has an expected result
    assert True


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

    # todo: make a test that has an expected result
    assert True


def test_all():
    blob_centers = np.array(
        [[0.2, 2.3, -1.5],  # y = 0
         [-1.5, 2.3, 1.8],  # y = 1
         [-2.8, 1.8, 1.3],  # y = 2
         [-2.8, 2.8, -2.8],  # y = 3
         [-2.8, 1.3, 2.3]])  # y = 4
    blob_std = np.array([0.7, 0.3, 0.6, 0.3, 0.2])

    X, y = make_blobs(n_samples=300, centers=blob_centers, cluster_std=blob_std, shuffle=True)
    # X, y = load_digits(return_X_y = True)
    # X, y = fetch_data('mushroom', return_X_y=True)
    # X, y = load_wine(return_X_y=True)
    # X, y = load_iris(return_X_y=True)
    n_splits = 10

    bad_case_splitter_dbscv = DBSVC.DBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=True)
    splitter_dbscv = DBSVC.DBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False)

    bad_case_splitter_dobscv = DOBSCV.DOBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=True)
    splitter_dobscv = DOBSCV.DOBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False)

    splitter_stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

    splitter_cbdscv = CBDSCV.CBDSCVSplitter(n_splits=n_splits, n_clusters=None)

    splitter_cbdscv_gmeans = CBDSCV_gmeans.CBDSCV_gmeansSplitter(n_splits=3, bad_case=False)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    print(len(X))
    scores = cross_val_score(pipeline, X, y=y, cv=splitter_cbdscv_gmeans)
    print("-------------------")
    print("Results with CBDSCV gmeans:")
    print("Scores: ", scores)
    print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    print('numero de folds: ', splitter_cbdscv_gmeans.n_splits)
    print('numero de clusters: ', splitter_cbdscv_gmeans.n_clusters)
    # print("STD: ", np.std(scores))

    # scores = cross_val_score(pipeline, X, y=y, cv=splitter_cbdscv)
    # print("-------------------")
    # print("Results with CBDSCV:")
    # print("Scores: ", scores)
    # print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    # print('numero de folds: ', splitter_cbdscv.n_splits)
    # print('numero de clusters: ', splitter_cbdscv.n_clusters)
    # print("STD: ", np.std(scores))

    # scores = cross_val_score(pipeline, X, y=y, cv=splitter_dobscv)
    # print("-------------------")
    # print("Results with DOBSCV:")
    # # print("Scores: ", scores)
    # # print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    # print("STD: ", np.std(scores))

    # scores = cross_val_score(pipeline, X, y=y, cv=bad_case_splitter_dobscv)
    # print("-------------------")
    # print("Results with bad case DOBSCV:")
    # # print("Scores: ", scores)
    # # print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    # print("STD: ", np.std(scores))

    # scores = cross_val_score(pipeline, X, y=y, cv=splitter_dbscv)
    # print("-------------------")
    # print("Results with DBSCV:")
    # # print("Scores: ", scores)
    # # print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    # print("STD: ", np.std(scores))

    # scores = cross_val_score(pipeline, X, y=y, cv=bad_case_splitter_dbscv)
    # print("-------------------")
    # print("Results with bad case DBSCV:")
    # # print("Scores: ", scores)
    # # print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    # print("STD: ", np.std(scores))

    # scores = cross_val_score(pipeline, X, y=y, cv=splitter_stratified_cv)
    # print("-------------------")
    # print("Results with Stratified k-fold cross validation:")
    # # print("Scores: ", scores)
    # # print("Mean {} Median {} STD {}".format(np.mean(scores), np.median(scores), np.std(scores)))
    # print("STD: ", np.std(scores))

    # todo: make a test that has an expected result
    assert True
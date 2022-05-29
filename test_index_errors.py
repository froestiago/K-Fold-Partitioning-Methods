from kfoldmethods.datasets.pmlb_api import pmlb_get_ds_list
from kfoldmethods.splitters import DOBSCV, DBSVC, CBDSCV, CBDSCV_gmeans
from kfoldmethods.splitters import test_splitters

from pmlb import fetch_data


def test_index_errors():
    ds = 'analcatdata_asbestos'
    cache_dir = 'cache'
    n_splits = 5

    splitter_methods = [
        DBSVC.DBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False, random_state=0),
        # DOBSCV.DOBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False, random_state=0),
        # CBDSCV.CBDSCVSplitter(n_splits=n_splits, shuffle=False, random_state=0),
        # CBDSCV_gmeans.CBDSCV_gmeansSplitter(shuffle=False, bad_case=False, random_state=0),
    ]

    X, y = fetch_data(ds, return_X_y=True, local_cache_dir=cache_dir)
    for splitter in splitter_methods:
        for train, test in splitter.split(X, y):
            print(len(train), len(test))
            exit(0)
        print(X.shape)


if __name__ == "__main__":
    test_splitters.test_dbscv()
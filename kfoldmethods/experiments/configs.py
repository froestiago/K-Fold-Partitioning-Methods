from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from kfoldmethods.datasets.pmlb_api import pmlb_get_ds_list
from kfoldmethods.splitters import CBDSCV, DBSVC, DOBSCV, CBDSCV_gmeans

run_data_dir = 'run_data'
pipeline = Pipeline([('scaler', MinMaxScaler()), ('clf', LogisticRegression())])
pipeline_params = [
    {'clf': [LogisticRegression(max_iter=100100, random_state=0, class_weight='balanced')], 
    'clf__C': [0.003, 0.03, 0.3, 3.0, 30.0]},

    {'clf': [SVC(kernel='rbf', max_iter=100100, random_state=0, class_weight='balanced')], 
        'clf__C': [0.3, 3.0, 30.0, 300.0],
        'clf__gamma': [0.00003, 0.0003, 0.003, 0.03, 0.3]},
        
    {'clf': [RandomForestClassifier(random_state=0, class_weight='balanced')], 
        'clf__max_depth': [1, 5, 10, 15, 50]},

    {'clf': [DecisionTreeClassifier(random_state=0, class_weight='balanced')], 
        'clf__max_depth': [1, 5, 10, 15, 50]}
]

tuning_folds = 10
tuning_grid_seach_n_jobs = 4
classifier_hyperparameters_output = "%s/classifier_hyperparameters" % run_data_dir

datasets = pmlb_get_ds_list(task='classification', n_samples=(200, 2000), verbose=False)
datasets = datasets[::3]

true_estimates_n_splits = 100
true_estimates_n_jobs = 5
true_estimates_random_state = 123

estimate_n_clusters_n_iters = 50
estimate_n_clusters_random_state = 123
estimate_n_clusters_n_jobs = 5

compare_splitters__n_repeats = 20
comapre_splitters__repeats_random_states = [123 + i for i in range(compare_splitters__n_repeats)]
compare_splitters__n_splits = 10
compare_splitters__n_jobs = 8
compare_splitters__path_n_clusters = "run_data/n_clusters_estimate/estimate_n_clusters.csv"

splitter_methods = [
    ('DBSCV', DBSVC.DBSCVSplitter, {
        'n_splits': compare_splitters__n_splits, 'shuffle': True, 'bad_case': False, 'random_state': 123}),

    ('DOBSCV', DOBSCV.DOBSCVSplitter, {
        'n_splits': compare_splitters__n_splits, 'shuffle': True, 'bad_case': False, 'random_state': 123}),

    ('CBDSCV', CBDSCV.CBDSCVSplitter, {
        'n_splits': compare_splitters__n_splits, 'shuffle': True, 'random_state': 123, 'minibatch_kmeans': False}),

    ('CBDSCV_Mini', CBDSCV.CBDSCVSplitter, {
        'n_splits': compare_splitters__n_splits, 'shuffle': True, 'random_state': 123, 'minibatch_kmeans': True}),

    # ('CBDSCV_gmeans', CBDSCV_gmeans.CBDSCV_gmeansSplitter, {
    #     'shuffle': True, 'bad_case': False, 'random_state': 123}),

    ('StratifiedKFold', StratifiedKFold, {
        'n_splits': compare_splitters__n_splits, 'shuffle': True, 'random_state': 123}),
    ('KFold', KFold, {
        'n_splits': compare_splitters__n_splits, 'shuffle': True, 'random_state': 123}),
    ('StratifiedShuffleSplit', StratifiedShuffleSplit, {
        'n_splits': compare_splitters__n_splits, 'test_size': 0.1, 'random_state': 123}),
    ('ShuffleSplit', ShuffleSplit, {
        'n_splits': compare_splitters__n_splits, 'test_size': 0.1, 'random_state': 123}),
]

need_n_clusters = ['CBDSCV', 'CBDSCV_Mini', 'CBDSCV_gmeans']
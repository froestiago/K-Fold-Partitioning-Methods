from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ..splitters import DOBSCV, DBSVC
from .loggers import LocalLogger, local_logger_to_long_frame

from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.base import clone
import seaborn as sns


def compare_variance(dir_output, run_name=None, **kwargs):
    logger = LocalLogger(dir_output=dir_output, use_tensor_board=True)
    path_cache = str((Path(dir_output) / Path('cachedir')).resolve())
    memory = Memory(path_cache)
    random_state = 0
    n_splits = 5
    n_runs = 100

    splitter_methods = [
        DBSVC.DBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False, random_state=0),
        DOBSCV.DOBSCVSplitter(n_splits=n_splits, shuffle=False, bad_case=False, random_state=0),
        StratifiedKFold(n_splits=n_splits),
        KFold(n_splits=n_splits),
        StratifiedShuffleSplit(n_splits=n_splits, random_state=0)
    ]
    datasets = [load_breast_cancer]

    pipeline = Pipeline([('scaler', MinMaxScaler()), ('clf', LogisticRegression())])
    pipeline_params = [
        {'clf': [LogisticRegression(max_iter=100100, random_state=0)], 'clf__C': [0.003, 0.03, 0.3, 3.0, 30.0]},
        {'clf': [SVC(kernel='rbf', max_iter=100100, random_state=0)], 'clf__C': [0.3, 3.0, 30.0, 300.0],
         'clf__gamma': [0.00003, 0.0003, 0.003, 0.03, 0.3]},
        {'clf': [DecisionTreeClassifier(random_state=0)], 'clf__max_depth': [1, 5, 10, 15, 50]}
    ]

    for ds in datasets:
        X, y = ds(return_X_y=True)

        # this will use cross validation to find the best hyperparameters for each classifier in this dataset
        # it will append to classifiers a estimator with unfitted parameters, but with the best hyperparameters
        classifiers = []
        for params in pipeline_params:
            print("Looking for best hyperparameters of %s ..." % params['clf'][0].__class__)
            search = GridSearchCV(pipeline, params, refit=True, cv=5)
            search.fit(X, y)
            estim = clone(search.best_estimator_)  # 'clone' will copy the unfitted best estimator.
            classifiers.append(estim)
            print("Found: {}".format(estim.get_params()))
            print("Results: {}".format(search.best_score_))

        # I use a rng_sampler so that the resampling works the same way always independently of how many ds and models
        # we use. Besides adding reproducibility, this also improves efficiency since we cache the results.
        rng_sampler = np.random.RandomState(random_state)
        for run in range(n_runs):
            print("Run: [%d/%d]" % (run + 1, n_runs))
            random_state_sampler = rng_sampler.randint(0, 99999)
            resample_indices = np.arange(0, X.shape[0])
            resample_indices = resample(resample_indices, random_state=random_state_sampler)
            logger.log_json(resample_indices.tolist(), '%s/indices.json' % ds.__name__)

            X_r = X[resample_indices]
            y_r = y[resample_indices]
            for splitter in splitter_methods:
                for train, test in splitter.split(X_r, y_r):
                    for model in classifiers:
                        model = model.fit(X_r[train], y_r[train])
                        y_pred = model.predict(X_r[test])

                        accuracy = accuracy_score(y_r[test], y_pred)
                        precision = precision_score(y_r[test], y_pred, average='macro')
                        recall = recall_score(y_r[test], y_pred, average='macro')
                        f1 = f1_score(y_r[test], y_pred, average='macro')

                        ns = '%s/%d/%s/%s' % (ds.__name__, run, splitter.__class__.__name__,
                                              model['clf'].__class__.__name__)
                        logger.log_metric(accuracy, '%s/accuracy' % ns)
                        logger.log_metric(precision, '%s/precision' % ns)
                        logger.log_metric(recall, '%s/recall' % ns)
                        logger.log_metric(f1, '%s/f1' % ns)


def compare_variance_analysis(path_run: str):
    path_run_obj = Path(path_run)

    sns.set_theme(style='darkgrid')

    if path_run_obj.is_dir():
        df = local_logger_to_long_frame(
            path_run,
            variables_names=('dataset', 'run', 'splitter', 'classifier', 'metric', 'value'),
            values_to_ignore=('tb', 'indices.json'),
            save_csv=True)

        df = df.drop(df.loc[df['splitter'] == 'StratifiedShuffleSplit'].index)

        df = df.replace(
            to_replace=['DBSCVSplitter', 'DOBSCVSplitter', 'KFold', 'StratifiedKFold'],
            value=['DBSCV', 'DOBSCV', 'KFold', 'StKFold']
        )

        df_runs = df.groupby(by=['dataset', 'run', 'splitter', 'classifier', 'metric'], as_index=False)\
            .agg(cv_mean=('value', 'mean'), cv_std=('value', 'std'))

        n_clf = len(df_runs['classifier'].unique())
        fig, ax = plt.subplots(n_clf, 2, figsize=(10, 9))
        for idx, clf in enumerate(df_runs['classifier'].unique()):
            metric = 'f1'
            df_clf_f_score = df_runs.loc[(df_runs['classifier'] == clf) & (df_runs['metric'] == metric)]

            sns.violinplot(data=df_clf_f_score, y='cv_mean', x='splitter', inner='quartile', ax=ax[idx, 0])
            ax[idx, 0].set_title('%s 5-fold CV mean' % clf)
            ax[idx, 0].set_ylabel('%s' % metric)
            ax[idx, 0].set_ylim([0.925, 1.00])

            sns.violinplot(data=df_clf_f_score, y='cv_std', x='splitter', inner='quartile', ax=ax[idx, 1])
            ax[idx, 1].set_title('%s 5-fold CV std' % clf)
            ax[idx, 1].set_ylabel('%s' % metric)
            ax[idx, 1].set_ylim([0.00, 0.045])

        fig.tight_layout()

        plt.show()

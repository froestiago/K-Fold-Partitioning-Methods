from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from pmlb import fetch_data
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed
from kfoldmethods.experiments import configs
from kfoldmethods.experiments.utils import load_best_classifier_for_dataset


class TrueMetricsEstimateResults:
    def __init__(self):
        # this design could be normalized, but maybe this won't improve usability in python
        self.records_splits = []
        self.records_classifiers = []
        self.records_metrics = []

    def insert_dataset_split(self, ds_name, split_id, clf_name, train, test):
        self.records_splits.append({
            'ds_name': ds_name,
            'split_id': split_id,
            'clf_name': clf_name,
            'train': train,
            'test': test,
        })

    def insert_classifier(self, ds_name, split_id, clf_name, classifier_object):
        self.records_classifiers.append({
            'ds_name': ds_name,
            'split_id': split_id,
            'classifier_name': clf_name,
            'classifier_object': classifier_object
        })

    def insert_metric_result(self, ds_name, split_id, clf_name, metric_name, metric_result):
        self.records_metrics.append({
            'ds_name': ds_name,
            'split_id': split_id,
            'classifier_name': clf_name,
            'metric_name': metric_name,
            'metric_result': metric_result
        })

    def insert_metric_results(self, ds_name, split_id, clf_name, metrics_name_value):
        for metric_name, metric_value in metrics_name_value:
            self.insert_metric_result(ds_name, split_id, clf_name, metric_name, metric_value)

    def select_metric_results(self):
        df = pd.DataFrame.from_records(self.records_metrics)
        return df


class TrueMetricsEstimate:
    def __init__(self, output_dir=None, ds_idx_0=None, ds_idx_last=None):
        self.results = TrueMetricsEstimateResults()
        self.ds_idx_0 = ds_idx_0 if ds_idx_0 is not None else 0
        self.ds_idx_last = ds_idx_last if ds_idx_last is not None else len(configs.datasets)-1

        if output_dir is None:
            self.path_results = Path(output_dir) / \
                Path('true_estimate') / \
                datetime.now().isoformat(timespec='seconds') / \
                'results_{}_to_{}.joblib'.format(self.ds_idx_0, self.ds_idx_last)
        else:
            self.path_results = Path(output_dir) / \
                'results_{}_to_{}.joblib'.format(self.ds_idx_0, self.ds_idx_last)
        
        self.path_results.parent.mkdir(exist_ok=True, parents=True)

    def compute_clf_estimates(self, ds_name, clf_class_name):
        X, y = fetch_data(ds_name, return_X_y=True)

        splitter = StratifiedShuffleSplit(
            n_splits=configs.true_estimates_n_splits, random_state=configs.true_estimates_random_state)

        for split_id, (train, test) in enumerate(splitter.split(X, y)):
            print("Split [%d/%d]" % (split_id, configs.true_estimates_n_splits - 1))
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[test], y[test]

            clf = load_best_classifier_for_dataset(ds_name, clf_class_name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            confusion_mat = confusion_matrix(y_test, y_pred)
            metric_results = [
                ('accuracy', accuracy), ('precision', precision), ('recall', recall),
                ('f1', f1), ('confusion_mat', confusion_mat)
            ]

            self.results.insert_dataset_split(ds_name, split_id, clf_class_name, train, test)
            self.results.insert_classifier(ds_name, split_id, clf_class_name, clf)
            self.results.insert_metric_results(ds_name, split_id, clf_class_name, metric_results)
            
            joblib.dump(self.results, self.path_results)

    def estimate_true_metrics(self):
        for ds_idx, ds_name in enumerate(configs.datasets):
            if self.ds_idx_0 <= ds_idx <= self.ds_idx_last:
                for params in configs.pipeline_params:
                    clf_class_name = params['clf'][0].__class__.__name__
                    print("Estimating metrics for {} with {}.".format(ds_name, clf_class_name))

                    self.compute_clf_estimates(ds_name, clf_class_name)


def analyze(args):
    #TODO: refactor
    path_true_estimates = Path('run_data/true_estimate')

    latest = None
    latest_date = None
    for dir in path_true_estimates.glob("*"):
        run_date = datetime.fromisoformat(str(dir.stem))
        
        if latest is None:
            latest = dir
            latest_date = run_date
        elif run_date > latest_date:
            latest = dir
            latest_date = run_date
    
    # results = joblib.load(str(latest / 'results_cp.joblib'))
    results = joblib.load(
        Path("run_data/true_estimate/2022-06-11T23:34:47/results_0_to_3.joblib"))
    metrics_df = results.select_metric_results()
    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv("metrics_df.csv")
    print(metrics_df)


def run_true_metrics_estimate(output_dir, ds_idx_0, ds_idx_last):
    print("Running datasets %d to %d" % (ds_idx_0, ds_idx_last))
    TrueMetricsEstimate(
        output_dir=output_dir, ds_idx_0=ds_idx_0, ds_idx_last=ds_idx_last).estimate_true_metrics()
    print("Finished datasets %d to %d" % (ds_idx_0, ds_idx_last))


def main(args):
    # TODO: refactor a bit the args
    if args.analyze:
        analyze(args)
        return
    ds_first, ds_last = args.ds_range

    output_dir = Path('run_data/true_estimate') / datetime.now().isoformat(timespec='seconds')
    n_datasets = len(configs.datasets)
    step = 4
    
    Parallel(n_jobs=configs.true_estimates_n_jobs)(
        delayed(run_true_metrics_estimate)(output_dir, i, min(i+step-1, n_datasets-1)) for i in range(0, n_datasets, step)
    )

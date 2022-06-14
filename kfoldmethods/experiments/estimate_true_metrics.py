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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


def select_metric_results():
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
    
    path_true_estimate_metrics = Path('true_estimate_metrics')
    path_true_estimate_metrics.mkdir(exist_ok=True, parents=True)
    for i in range(27):
        print("DS INDEX: [%d/%d]" % (i, 26))
        results = joblib.load(
            Path("run_data/true_estimate/2022-06-12T03:15:59/results_{}_to_{}.joblib".format(i, i)))
        metrics_df = results.select_metric_results()
        metrics_df = pd.DataFrame(metrics_df)

        path_true_estimate_run = path_true_estimate_metrics / "true_estimate_{}.csv".format(i)
        metrics_df.to_csv(str(path_true_estimate_run))


def plot_distributions_for_datasets(path_true_estimates_csv: str):
    # generate ds x clf figures distributions of the results. Each figure is 1xn_metrics
    # This is to give a visual idea of the true estimates accuracy
    path_true_estimate_distributions = Path("true_estimate_distributions")
    path_true_estimate_distributions.mkdir(exist_ok=True, parents=True)

    df_run = pd.read_csv(path_true_estimates_csv)

    groups = df_run.groupby(by=['ds_name'])
    for ds_name, df_group in groups:
        df_group = df_group[df_group['metric_name'] != 'confusion_mat']
        df_group.loc[:, 'metric_result'] = pd.to_numeric(df_group.loc[:, 'metric_result'])

        n_classifiers = len(df_group['classifier_name'].unique())
        n_metrics = len(df_group['metric_name'].unique())

        sns.set_theme()
        fig, ax = plt.subplots(n_metrics, n_classifiers, figsize=(8.3, 11.7))
        fig.suptitle(ds_name)
        for i, metric_name in enumerate(df_group['metric_name'].unique()):
            for j, classifier_name in enumerate(df_group['classifier_name'].unique()):
                df_plot = df_group[(df_group['metric_name'] == metric_name) & \
                                   (df_group['classifier_name'] == classifier_name)]

                sns.histplot(data=df_plot, x='metric_result', ax=ax[i, j])
                ax[i, j].set_title(classifier_name.replace("Classifier", ""))
                ax[i, j].set_xlabel(metric_name)
        
        fig.tight_layout()
        path_fig = str(path_true_estimate_distributions / "{}_distribution.pdf".format(ds_name))
        fig.savefig(path_fig)
        plt.close(fig)


def table_results(path_true_estimate_metrics):
    path_true_estimate_tables = Path("true_estimate_tables")
    path_true_estimate_tables.mkdir(exist_ok=True, parents=True)

    # Summary of mean performance
    for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
        df_summary = pd.DataFrame()

        for i in range(27):
            path_csv = str(path_true_estimate_metrics / "true_estimate_{}.csv".format(i))
            df_run = pd.read_csv(path_csv)
            # df_run = df_run[df_run['metric_name'] != 'confusion_mat']
            df_run = df_run[df_run['metric_name'] == metric_name]

            df_run.loc[:, 'metric_result'] = pd.to_numeric(df_run.loc[:, 'metric_result'])
            
            for ds_name in df_run['ds_name'].unique():
                groups = df_run[df_run['ds_name'] == ds_name].groupby(by=['metric_name', 'classifier_name'])
                results = groups.agg(mean=('metric_result', np.mean)).transpose().rename(index={'mean': ds_name})
                df_summary = pd.concat((df_summary, results), axis=0)
            
        df_summary.rename(
            columns={'RandomForestClassifier': 'RandomForest', 'DecisionTreeClassifier': 'DecisionTree'}, 
                    inplace=True)
        df_summary.to_csv(
            str(path_true_estimate_tables / "{}_mean.csv".format(metric_name)), float_format='%.4f')

    # Summary of std of performance
    for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
        df_summary = pd.DataFrame()

        for i in range(27):
            path_csv = str(path_true_estimate_metrics / "true_estimate_{}.csv".format(i))
            df_run = pd.read_csv(path_csv)
            # df_run = df_run[df_run['metric_name'] != 'confusion_mat']
            df_run = df_run[df_run['metric_name'] == metric_name]

            df_run.loc[:, 'metric_result'] = pd.to_numeric(df_run.loc[:, 'metric_result'])
            
            for ds_name in df_run['ds_name'].unique():
                groups = df_run[df_run['ds_name'] == ds_name].groupby(by=['metric_name', 'classifier_name'])
                results = groups.agg(std=('metric_result', np.std)).transpose().rename(index={'std': ds_name})
                df_summary = pd.concat((df_summary, results), axis=0)
            
        df_summary.rename(
            columns={'RandomForestClassifier': 'RandomForest', 'DecisionTreeClassifier': 'DecisionTree'}, 
                    inplace=True)
        df_summary.to_csv(
            str(path_true_estimate_tables / "{}_std.csv".format(metric_name)), float_format='%.4f')
        

def analyze(args):
    plot_distributions = False
    metric_tables = False
    build_true_estimate_summary = True
    path_true_estimate_metrics = Path("true_estimate_metrics")

    if plot_distributions:
        for i in range(27):
            path_csv = str(path_true_estimate_metrics / "true_estimate_{}.csv".format(i))
            plot_distributions_for_datasets(path_csv)
    
    if metric_tables:
        table_results(path_true_estimate_metrics)
    
    if build_true_estimate_summary:
        df_true_estimates = pd.DataFrame()
        for f in path_true_estimate_metrics.glob("*.csv"):
            df_run_true_estimates = pd.read_csv(f)
            df_true_estimates = pd.concat((df_true_estimates, df_run_true_estimates), axis=0)
        
        df_true_estimates = df_true_estimates[df_true_estimates['metric_name'] != 'confusion_mat']
        df_true_estimates.loc[:, 'metric_result'] = pd.to_numeric(df_true_estimates.loc[:, 'metric_result'])

        df_true_estimates_summary = df_true_estimates.groupby(
            by=['ds_name', 'classifier_name', 'metric_name']).agg(
                true_value=('metric_result', np.mean))
        df_true_estimates_summary.to_csv('true_estimates_summary.csv', float_format='%.4f')


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

    if args.select_metric_results:
        select_metric_results()
        return
    
    output_dir = Path('run_data/true_estimate') / datetime.now().isoformat(timespec='seconds')
    n_datasets = len(configs.datasets)
    step = 1
    
    Parallel(n_jobs=configs.true_estimates_n_jobs)(
        delayed(run_true_metrics_estimate)(output_dir, i, min(i+step-1, n_datasets-1)) for i in range(0, n_datasets, step)
    )

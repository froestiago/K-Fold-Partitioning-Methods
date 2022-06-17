from pathlib import Path
import joblib
from pmlb import fetch_data
from sklearn import clone
from sklearn.model_selection import GridSearchCV


from kfoldmethods.experiments import configs


def save_hp_tuning(ds_name: str, search):
    clf_name = str(search.best_estimator_['clf'].__class__.__name__)
    path_ds_classifiers = Path(configs.classifier_hyperparameters_output) / ds_name
    path_ds_classifiers = path_ds_classifiers / clf_name
    path_ds_classifiers.mkdir(exist_ok=True, parents=True)

    filepath = path_ds_classifiers / 'search.joblib'
    joblib.dump(search, filepath)
    

def tune(args):
    datasets = configs.datasets
    for ds_name in datasets:
        X, y = fetch_data(ds_name, return_X_y=True)
        print("Finding hyperparameters for {}. Shape: {}".format(ds_name, X.shape))

        for params in configs.pipeline_params:
            pipeline = clone(configs.pipeline)
            print("Looking for best hyperparameters of %s ..." % params['clf'][0].__class__)

            search = GridSearchCV(
                pipeline, params, refit=True, cv=configs.tuning_folds, n_jobs=configs.tuning_grid_seach_n_jobs,
                scoring=configs.tuning_grid_search_scoring)
            search.fit(X, y)

            params = search.best_estimator_.get_params()
            print("Results: {}".format(search.best_score_))

            save_hp_tuning(ds_name, search)
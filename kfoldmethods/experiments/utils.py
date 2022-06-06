from pathlib import Path
from sklearn import clone
import joblib


def load_best_classifier_for_dataset(
        ds, clf_class_name, hp_dir='run_data/classifier_hyperparameters'):
    path_search = Path(hp_dir) / ds / clf_class_name / 'search.joblib'
    search = joblib.load(path_search)
    return clone(search.best_estimator_)

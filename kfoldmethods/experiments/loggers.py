from pathlib import Path
from datetime import datetime
from typing import Tuple

import joblib
import json
import pandas as pd
import os


def logger_name_append(a, b):
    return '%s/%s' % (a, b)


class LocalLogger:
    """
    Just a basic local logger.
    """
    def __init__(self, dir_output, use_tensor_board=False):
        self._dir_output = dir_output
        self._use_tb = use_tensor_board

        # create root directory
        date_now = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-4]
        self._path_root = Path(dir_output) / Path(date_now)
        self._path_root.mkdir(parents=True, exist_ok=True)
        print("Files for this run will be logged to %s" % str(self._path_root.resolve()))
        if self._use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                path_tb = str((self._path_root / Path('tb')).resolve())
                self._tb_writer = SummaryWriter(log_dir=path_tb)
                print("Logs can be visualized using tensor board.")
                print("Run:\ntensorboard --logdir=%s" % path_tb)
            except ImportError:
                print("Failed to import Summary writer from torch.utils.tensorboard")
                print("Tensor board will not be used.")
                self._use_tb = False

    def log_object(self, obj, path):
        """Store given object to the given path.

        Parameters
        ----------
        obj : object
            Object to save. The object will be stored using joblib.
        path : str
            Path where to save the file relative to the root path of the logger.
        """
        path_obj = self._path_root / Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, str(path_obj.resolve()))
        # print("Object stored in %s" % str(path_obj.resolve()))

    def log_json(self, obj_json, path):
        """Store given object as a json file. The object is stored as an element in a list. If the file already exists,
        then the new object is appended to the list.

        Parameters
        ----------
        obj_json : object
            Object to save as a json file.
        path : str
            Path where to save object as json file. It should have a .json extension.
        """
        path_obj = self._path_root / Path(path)
        if path_obj.suffix != ".json":
            raise ValueError("The path given should finish with .json, but it is %s" % path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not path_obj.exists():
            data = []
            with open(str(path_obj.resolve()), 'w') as f:
                json.dump(data, f)

        with open(str(path_obj.resolve()), 'r') as f:
            data = json.load(f)
        data.append(obj_json)
        with open(str(path_obj.resolve()), 'w') as f:
            json.dump(data, f)

        # print("JSON appended to %s" % str(path_obj.resolve()))

    def log_metric(self, score, path):
        """Log a metric value to the given path.

        The path in this case should have no extension.
        If it exists, the new metric value will be appended to it. In disk, the values are json files.

        This methods is useful for instance when measures metrics across multiple runs.
        If one wants to measure the accuracy of the method 'svm' over multiple runs, one could log to the path 'svm/acc'
        after measuring the accuracy after each run. This will generate a list of values.

        Parameters
        ----------
        score : float or int
            Score to log.
        path : str
            Path where to log the metric score.
        """
        path_obj = self._path_root / Path(path)
        if path_obj.suffix != "":
            raise ValueError("The path given should have no extension, but it is %s" % path)
        path_obj_json = Path("%s.json" % str(path_obj.resolve()))
        path_obj_json.parent.mkdir(parents=True, exist_ok=True)

        # create empty list if file does not exist
        if not path_obj_json.exists():
            data = []
            with open(str(path_obj_json.resolve()), 'w') as f:
                json.dump(data, f)

        with open(str(path_obj_json.resolve()), 'r') as f:
            data = json.load(f)
        data.append(score)
        with open(str(path_obj_json.resolve()), 'w') as f:
            json.dump(data, f)

        if self._use_tb:
            self._tb_writer.add_scalar(path, scalar_value=score)

    @staticmethod
    def logger_name_append(a, b):
        return '%s/%s' % (a, b)


def local_logger_to_long_frame(run_dir: str,
                               variables_names: Tuple[str, ...],
                               values_to_ignore: Tuple[str, ...] = None,
                               save_csv: bool = False):
    """Convert a file structure created using LocalLogger to a data frame in long form.

    Notes
    -----
    The method currently ignores all non-json files. todo include joblib files

    Parameters
    ----------
    run_dir
    variables_names : List[str, ...]
        List with the name to give to each column. Each level of the nested structure
        gets the corresponding name in the list. The last level can have a file with
        a list containing multiple elements. In that case, the last element in the
        variable names is used as the name of the column.
    values_to_ignore : List[str, ...]
        These values are ignored when converting the structure.
    save_csv :
        Save resulting dataframe as csv

    Returns
    -------

    """
    run_dir_obj = Path(run_dir)

    data = {}
    for v in variables_names:
        data[v] = []

    for p in run_dir_obj.rglob("*"):
        ignore_path = False
        parts = os.path.normpath(str(p)).split(os.path.sep)
        for v in values_to_ignore:
            if v in parts:
                ignore_path = True

        if not p.is_file():
            ignore_path = True

        if not p.suffix == '.json':
            ignore_path = True

        if ignore_path:
            continue

        with open(p, 'r') as f:
            last_values = json.load(f)

        path_parts = os.path.splitext(
            str(p.relative_to(run_dir_obj))
        )[0].split(os.path.sep)

        for last_value in last_values:
            last_value_name = variables_names[-1]
            data[last_value_name].append(last_value)

            for name, value in zip(variables_names[:-1], path_parts):
                data[name].append(value)

    df = pd.DataFrame.from_dict(data)
    if save_csv:
        path_csv = "%s.csv" % run_dir
        df.to_csv(path_csv)

    return df

"""
Interface for filtering PMLB datasets.

"""

from typing import Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd


def pmlb_get_ds_list(task: Union[str, None] = 'classification',
                     n_samples: Tuple[Union[int, None], Union[int, None]] = (1, None),
                     n_features: Tuple[Union[int, None], Union[int, None]] = (1, None),
                     n_classes: Tuple[Union[int, None], Union[int, None]] = (2, None),
                     imbalance: Tuple[Union[int, None], Union[int, None]] = (None, None),
                     verbose: bool = True):
    path_pmlb = str(Path(__file__).parent / 'pmlb_datasets.csv')

    df = pd.read_csv(path_pmlb)
    df = df.drop(columns=['Metadata'])
    df['Imbalance'].loc[df['Imbalance'] == 0.0] = np.nan

    if task is not None:
        df = df.loc[df['Task'] == task]

    if n_samples[0] is not None:
        df = df.loc[df['n_observations'] >= n_samples[0]]
    if n_samples[1] is not None:
        df = df.loc[df['n_observations'] < n_samples[1]]

    if n_features[0] is not None:
        df = df.loc[df['n_features'] >= n_features[0]]
    if n_features[1] is not None:
        df = df.loc[df['n_features'] < n_features[1]]

    if n_classes[0] is not None:
        df = df.loc[df['n_classes'] >= n_classes[0]]
    if n_classes[1] is not None:
        df = df.loc[df['n_classes'] < n_classes[1]]

    if imbalance[0] is not None:
        df = df.loc[df['imbalance'] >= imbalance[0]]
    if imbalance[1] is not None:
        df = df.loc[df['imbalance'] < imbalance[1]]

    if verbose:
        print(df)
        print(df['Dataset'].to_numpy())
        print(df.describe())

    return df['Dataset'].to_list()


if __name__ == "__main__":
    ds_names = pmlb_get_ds_list(n_samples=(1, 10050), verbose=False)
    print(ds_names)

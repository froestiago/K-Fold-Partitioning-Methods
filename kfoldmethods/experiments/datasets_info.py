import pandas as pd
from pathlib import Path

from kfoldmethods.experiments import configs


def main(args):
    df_datasets = pd.read_csv(configs.dataset_info__pmlb_list_path)
    selected = configs.datasets_balanced
    path_dataset_info = Path(configs.dataset_info__output_dir)
    path_dataset_info.mkdir(exist_ok=True, parents=True)
    
    df_selected = df_datasets.loc[
        df_datasets['Dataset'].isin(selected), 
        ['Dataset', 'n_observations', 'n_features', 'n_classes', 'Imbalance']]
    df_selected = df_selected.sort_values(by='Dataset')
    df_selected.to_csv(Path(configs.dataset_info__output_dir) / 'datasets_balanced.csv', index=False)
    print(df_selected)

    selected_imb = configs.datasets_imb
    df_selected = df_datasets.loc[
        df_datasets['Dataset'].isin(selected_imb), 
        ['Dataset', 'n_observations', 'n_features', 'n_classes', 'Imbalance']]
    df_selected = df_selected.sort_values(by='Dataset')
    df_selected.to_csv(Path(configs.dataset_info__output_dir) / 'datasets_imbalanced.csv', index=False)
    print(df_selected)

    selected = configs.datasets
    df_selected = df_datasets.loc[
        df_datasets['Dataset'].isin(selected), 
        ['Dataset', 'n_observations', 'n_features', 'n_classes', 'Imbalance']]
    df_selected = df_selected.sort_values(by='Dataset')
    df_selected.to_csv(Path(configs.dataset_info__output_dir) / 'datasets.csv', index=False)
    print(df_selected)

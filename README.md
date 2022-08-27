# K-Fold-Partitioning-Methods

## Hyperparameter Tuning
We tune each classifier using the full dataset prior to the main experiments.
Since our goal is not to compare classifier performance, the data leakage resulting from this is not a problem.
The tuning only serves to guarantee that the classifiers have appropriate hyperparameters for each dataset.
To reproduce our results on the hyperparameters set, run: 

`python main.py hp-search`

The outputs are stored in the folder `classifier_hyperparameters`.

## Defining the Number of Clusters
The number of clusters in each dataset is computed prior to the main experiments and used as input to the cluster-based splitting strategies.
To reproduce these experiments, run

`python main.py n-clusters-estimate`, followed by

`python main.py n-clusters-estimate -a` to obtain a CSV file with the number of clusters in each dataset. 

The outputs are stored in the folder `n_clusters_estimate`.

## Estimates of the True Performance
This step requires that the instructions from *Hyperparameter Tuning* have been performed first.
We use the hyperparameters and to find estimates of the true performance of each classifier in each dataset.
To reproduce our results, run `python main.py true-estimate`, followed by `python main.py true-estimate -s` to retrieve CSV tables with the performance estimates for each dataset, classifier, and iteration.
Finally, `python main.py true-estimate -a` produces some CSV files with summary results as well as some figures omitted from the paper because of space. 

## Splitters Estimates of the Performance
This step requires that the instructions from *Hyperparameter Tuning* and *Defining the Number of Clusters* have been performed first.
To reproduce this experiment, run `python main.py compare-splitters`.
Note that this may take up to a few days to run.
After it, run `python main.py compare-splitters -s` to extract to CSV files the estimates from the metadata created through the previous command.
Finally, `python main.py compare-splitters -a` generates the box plots of the performances. 


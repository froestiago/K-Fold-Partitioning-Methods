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

## Estimate of the True Performance
This step requires the outputs from the *Hyperparameter Tuning* and *Defining the Number of Clusters*.


import argparse
from kfoldmethods.experiments import splitters_compare, tune_classifiers, estimate_true_metrics
from kfoldmethods.tests import test_splitters


def build_test_subparser(subparsers):
    parser_test = subparsers.add_parser('test', help='Run test function.')
    return parser_test


def build_experiment_subparser(subparsers):
    parser_experiment = subparsers.add_parser('run', help='Run experiments.')

    parser_experiment.add_argument(
        "-o", "--output-dir", type=str, default='run_data',
        help="Directory where to save results, if any. Default is %(default)s")

    parser_experiment.add_argument(
        "--rs", type=int, default=123, help="Random state (default: %(default)s).")

    parser_experiment.add_argument(
        "--n-splits", type=int, default=5, help="Number of folds in k-fold (default: %(default)s).")

    parser_experiment.add_argument(
        "--n-runs", type=int, default=2, help="Number of repetitions of the experiment (default: %(default)s).")

    parser_experiment.add_argument(
        "--start-from", type=int, default=0, help="Start from given dataset index (default: %(default)s).")

    return parser_experiment


def build_analysis_subparser(subparsers):
    parser_analysis = subparsers.add_parser('analyze', help='Analyze results of experiments.')

    parser_analysis.add_argument('-r', '--run-dir', required=True, type=str, help="Directory containing results.")

    return parser_analysis


def build_hyperparameters_search_subparsers(subparsers):
    parser_hp_search = subparsers.add_parser('hp-search', help='Analyze results of experiments.')
    return parser_hp_search


def build_estimate_true_metrics_subparsers(subparsers):
    parser_true_estimate = subparsers.add_parser('true-estimate', help='Estimate true metrics for each dataset and classifier.')
    parser_true_estimate.add_argument("-a", "--analyze", action="store_true", help="Analyze results of run")

    # TODO: ds_range is deprecated. Remove it
    parser_true_estimate.add_argument(
        "--ds-range", type=int, nargs=2, default=(0, None), help="Which datasets to process, starting from ds 0")
    parser_true_estimate.add_argument(
        "--select-metric-results", action="store_true", 
        help="Generate csv files containing only the metrics from the joblib files.")
    return parser_true_estimate


def main():
    parser = argparse.ArgumentParser("k-Fold Partitioning Methods")
    subparsers = parser.add_subparsers(dest='subparser_name')
    parser_test = build_test_subparser(subparsers)
    parser_experiment = build_experiment_subparser(subparsers)
    parser_analysis = build_analysis_subparser(subparsers)
    parser_hp = build_hyperparameters_search_subparsers(subparsers)
    parser_true_estimate = build_estimate_true_metrics_subparsers(subparsers)

    args = parser.parse_args()

    if args.subparser_name == "test":
        test_splitters.test_all()
        return

    if args.subparser_name == "run":
        splitters_compare.compare_variance(args)
        return
    
    if args.subparser_name == "analyze":
        splitters_compare.compare_variance_analysis(args)
        return

    if args.subparser_name == "hp-search":
        tune_classifiers.tune(args)
        return

    if args.subparser_name == "true-estimate":
        estimate_true_metrics.main(args)
        return
    

if __name__ == '__main__':
    main()

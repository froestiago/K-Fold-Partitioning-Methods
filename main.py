import argparse
from kfoldmethods.experiments import splitters_compare
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
        "--analysis", action="store_true", default=False,
        help="Run only the analysis of the metadata."
    )

    parser_experiment.add_argument(
        "--rs", type=int, default=123, help="Random state (default: %(default)s).")

    parser_experiment.add_argument(
        "--n-splits", type=int, default=5, help="Number of folds in k-fold (default: %(default)s).")

    parser_experiment.add_argument(
        "--n-runs", type=int, default=2, help="Number of repetitions of the experiment (default: %(default)s).")

    parser_experiment.add_argument(
        "--start-from", type=int, default=0, help="Start from given dataset index (default: %(default)s).")

    return parser_experiment


def main():
    parser = argparse.ArgumentParser("k-Fold Partitioning Methods")
    subparsers = parser.add_subparsers(dest='subparser_name')
    parser_test = build_test_subparser(subparsers)
    parser_experiment = build_experiment_subparser(subparsers)
    args = parser.parse_args()

    if args.subparser_name == "test":
        test_splitters.test_all()
        return

    if args.subparser_name == "run":
        if args.analysis:
            splitters_compare.compare_variance_analysis(args)
            return

        splitters_compare.compare_variance(args)
        splitters_compare.compare_variance_analysis(args)


if __name__ == '__main__':
    main()

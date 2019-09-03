import argparse

global_train_parser = argparse.ArgumentParser(add_help=False)

global_train_parser.add_argument(
    '-gpuid',
    default=-1,
    type=int,
    help ='GPU ID (-1 to disable ))')

global_train_parser.add_argument(
    '-dr_rate',
    default=0.3,
    type=float,
    help='dropout rate applied to LMs (default: 0.3)')

global_train_parser.add_argument(
    '-n_layer',
    default=2,
    type=int,
    help='number of layers (default: 2)')

global_train_parser.add_argument(
    '-emb_size',
    default=300,
    type=int,
    help='embedding size (default: 300)')

global_train_parser.add_argument(
    '-h_size',
    default=300,
    type=int,
    help='hidden state size (default: 300)')

global_train_parser.add_argument(
    '-batch_size',
    default=32,
    type=int,
    help='batch size (default: 32)')

global_train_parser.add_argument(
    '-epoch_size',
    default=30,
    type=int,
    help='number of epochs (default: 30)')

global_train_parser.add_argument(
    '-save_dir',
    type=str,
    required=True,
    help="directory to save models and embeddings")

global_train_parser.add_argument(
    '-data',
    type=str,
    required=True,
    help='preprocessed data name')

global_train_parser.add_argument(
    '-opt_type',
    default="ASGD",
    choices=['SGD', 'ASGD'],
    help='optimizer (SGD or ASGD) (default: ASGD)')

global_train_parser.add_argument(
    '-learning_rate',
    default=5.0,
    type=float,
    help='learning rate (default: 5.0)')

global_train_parser.add_argument(
    '-stop_threshold',
    default = float('inf'),
    type=float,
    help='threshold for early stopping (set as 0.99 in the paper; default: inf (= disable early stopping)')

global_train_parser.add_argument(
    '-remove_models',
    action='store_true',
    help='Only save the model at the last epoch (default: disabled)')

global_train_parser.add_argument(
    '-seed',
    default=0,
    type=int,
    help='random seed (default: 0)')

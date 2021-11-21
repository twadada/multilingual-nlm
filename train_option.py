import argparse

global_train_parser = argparse.ArgumentParser(add_help=False)

global_train_parser.add_argument(
    '-gpu',
    action='store_true',
    help ='Use GPU (recomended)')

global_train_parser.add_argument(
    '-dr_rate',
    default=0.5,
    type=float,
    help='dropout rate applied to LMs (default: 0.5)')

global_train_parser.add_argument(
    '-n_layer',
    default=1,
    type=int,
    help='number of layers (default: 1)')

global_train_parser.add_argument(
    '-emb_size',
    default=500,
    type=int,
    help='embedding size (default: 500)')

global_train_parser.add_argument(
    '-h_size',
    default=500,
    type=int,
    help='hidden state size (default: 500)')

global_train_parser.add_argument(
    '-batch_size',
    default=16,
    type=int,
    help='batch size (default: 16)')

global_train_parser.add_argument(
    '-epoch_size',
    type=int,
    required=True,
    help='number of epochs')

global_train_parser.add_argument(
    '-save_dir',
    type=str,
    required=True,
    help="directory where models and embeddings are saved")

global_train_parser.add_argument(
    '-data',
    type=str,
    required=True,
    help='preprocessed data name')

global_train_parser.add_argument(
    '-opt_type',
    default="Adam",
    choices=['SGD','Adam'],
    help='optimizer (SGD or Adam) (default: Adam)')

global_train_parser.add_argument(
    '-learning_rate_SGD',
    default=1.0,
    type=float,
    help='learning rate for SGD (default: 1.0)')

global_train_parser.add_argument(
    '-stop_threshold',
    default = float('inf'),
    type=float,
    help="threshold for unsupervised early stopping (the default is 'inf' (= disable early stopping)")

global_train_parser.add_argument(
    '-early_stop',
    action='store_true',
    help='enable early stop using validation data'
    )

global_train_parser.add_argument(
    '-remove_models',
    action='store_true',
    help='save the best/last model only (default: disabled)')

global_train_parser.add_argument(
    '-seed',
    default=0,
    type=int,
    help='random seed (default: 0)')

global_train_parser.add_argument(
    '-save_point',
    default=1,
    type=int,
    help='how often the model is saved (default: 1 epoch)')

global_train_parser.add_argument(
    '-eval_dict',
    default=None,
    nargs='+', type=str)

global_train_parser.add_argument(
    '-dico_max_size',
    type=int,
    default=3000,
    help='source train data path')

global_train_parser.add_argument(
    '-lang_class',
    nargs='+',
    type=int,
    help='decoder id'
)

global_train_parser.add_argument(
    '-pretrained_emb',
    nargs='+',
    type=str,
    help='language class based on word order (e.g SVO, SOV, VSO, etc.)'
)

global_train_parser.add_argument(
    '-share_vocab',
    choices=[0, 3, 4],
    default=0,
    type=int
)

global_train_parser.add_argument(
    '-swemb_size',
    type=int,
    help='backward training'
)
global_train_parser.add_argument(
    '-enc_dec_layer',
    nargs='+',
    type=int,
    help='number of layers of encoder and decoder'
)

global_train_parser.add_argument(
    '-subword',
    type=str,
    nargs='+',
    default=None,
    help='subword file')

global_train_parser.add_argument(
    '-dict_srclang',
    type=int,
    default=0,
)

global_train_parser.add_argument(
    '-dict_tgtlangs',
    type=int,
    nargs='+',
    default=None
)
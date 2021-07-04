import argparse
import jsonargparse
import os
from easydict import EasyDict


def add_pcae_args(parser):
    pcae_args = parser.add_argument_group('PCAE Parameters')
    pcae_args.add_argument(
        '--pcae.num_caps',
        type=int, default=16,
        metavar='PCAE_NCAPS',
        help='number of capsules')
    pcae_args.add_argument(
        '--pcae.caps_dim',
        type=int, default=6,
        help='number of dimensions per capsule')
    pcae_args.add_argument(
        '--pcae.feat_dim',
        type=int, default=16,
        help='number of feature dimensions per capsule')
    pcae_args.add_argument(
        '--pcae.lr',
        type=float, default=3e-3,
        help='learning rate')
    # .998 = 1-(1-.96)**1/20, equiv to .96 every 20 epochs
    pcae_args.add_argument(
        '--pcae.lr_decay',
        type=float, default=.998,
        help='learning rate decay (for exp schedule)')
    pcae_args.add_argument(
        '--pcae.lr_restart_interval',
        type=int, default=4000,
        help='number of steps between warm restarts (for cosrestarts schedule)')
    pcae_args.add_argument(
        '--pcae.weight_decay', type=float,
        help='weight decay')
    pcae_args.add_argument(
        '--pcae.decoder_lr_coeff', type=float,
        help='decoder learning rate coefficient')
    pcae_args.add_argument(
        '--pcae.optimizer', type=str.lower, choices=['sgd', 'radam'],
        help='optimizer algorithm')
    pcae_args.add_argument(
        '--pcae.lr_scheduler', type=str.lower, choices=['exp', 'cosrestarts'],
        help='learning rate scheduler')
    pcae_args.add_argument(
        '--pcae.loss_ll_coeff', type=float,
        help='log_likelihood loss contribution coefficient')
    pcae_args.add_argument(
        '--pcae.loss_temp_l1-coeff', type=float,
        help='template L1 norm loss contribution coefficient')
    pcae_args.add_argument(
        '--pcae.loss_mse_coeff', type=float,
        help='reconstruction MSE loss contribution coefficient')
    pcae_args.add_argument(
        '--pcae.loss_pres_l2_sparsity.batch', type=float,
        help='')
    pcae_args.add_argument(
        '--pcae.loss_pres_l2_sparsity.capsules', type=float,
        help='')
    pcae_args.add_argument(
        '--pcae.alpha_channel', type=bool,
        help='whether to add an alpha channel to the part templates')
    pcae_args.add_argument(
        '--pcae.inverse_space_transform', type=bool,
        help='learn part poses in non_inverse transform space')


def add_ocae_args(parser):
    ocae_args = parser.add_argument_group('OCAE Parameters')
    ocae_args.add_argument(
        '--ocae.lr',
        type=float, default=1e-1,
        help='learning rate')


def add_log_args(parser):
    log_args = parser.add_argument_group('Logger Parameters')
    log_args.add_argument(
        '--log.run_name',
        type=str, default='',
        help='W&B run name')
    log_args.add_argument(
        '--log.project',
        type=str, default='SCAE',
        help='W&B project name')
    log_args.add_argument(
        '--log.team',
        type=str, default=None,
        help='W&B team name')
    log_args.add_argument(
        '--log.upload',
        type=bool, default=False,
        help='upload logs to W&B')
    log_args.add_argument(
        '--log.frequency',
        type=int, default=1,
        help='logging frequency')


def parse_args(args=None):
    parser = jsonargparse.ArgumentParser(
        prog='scae',
        description='Training/evaluation/inference script for Stacked Capsule Autoencoders',
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        default_config_files=['scae/config/default.yaml'])

    parser.add_argument(
        '--cfg',
        action=jsonargparse.ActionConfigFile,
        help='Overides all args before this is passed')

    # todo(maximsmol): add inference and evaluation
    parser.add_argument(
        '-n', '--batch_size',
        type=int, default=128,
        help='number of samples per mini_batch')
    parser.add_argument(
        '-N', '--num_epochs',
        type=int, default=300,
        metavar='NEPOCHS',
        help='number of epochs')
    parser.add_argument(
        '--non_deterministic',
        action='store_true', default=False,
        help='allow non_deterministic operations for potentially higher performance')
    parser.add_argument(
        '-d', '--debug',
        action='store_true', default=False,
        help='enable autograd anomaly detection')
    parser.add_argument(
        '--model',
        type=str.lower, default='pcae',
        choices=['ccae', 'pcae', 'ocae', 'scae'],
        # todo(maximsmol): change default
        help='part of the model to run')
    parser.add_argument(
        '--dataset.name',
        type=str.lower, default='mnist',
        choices=['mnist', 'usps', 'cifar10', 'svhn'])
    parser.add_argument('--dataset.random_rotate', type=bool)
    parser.add_argument('--dataset.random_translate', type=bool)
    parser.add_argument(
        '--data_workers',
        type=int, default=len(os.sched_getaffinity(0)),
        metavar='NWORKERS',
        help='number of data loader workers')

    add_pcae_args(parser)
    add_ocae_args(parser)
    add_log_args(parser)

    return namespace_to_edict(parser.parse_args(args=args, _skip_check=True))


def namespace_to_edict(namespace):
    vars_dict = vars(namespace)
    for k in vars_dict:
        v = vars_dict[k]
        if isinstance(v, argparse.Namespace):
            vars_dict[k] = namespace_to_edict(v)
    return EasyDict(vars_dict)

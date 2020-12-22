import argparse
import os

from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser(
        prog='scae',
        description='Training/evaluation/inference script for Stacked Capsule Autoencoders',
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # todo(maximsmol): add inference and evaluation
    parser.add_argument(
        '-n', '--batch-size',
        type=int, default=128,
        help='number of samples per mini-batch')
    parser.add_argument(
        '-N', '--num-epochs',
        type=int, default=3000,
        metavar='NEPOCHS',
        help='number of epochs')
    parser.add_argument(
        '--non-deterministic',
        action='store_true', default=False,
        help='allow non-deterministic operations for potentially higher performance')
    parser.add_argument(
        '-d', '--debug',
        action='store_true', default=False,
        help='enable autograd anomaly detection')

    parser.add_argument(
        '--model',
        type=str.lower, default='ocae',
        choices=['ccae', 'pcae', 'ocae', 'scae'],
        # todo(maximsmol): change default
        help='part of the model to run')


    parser.add_argument(
        '--dataset',
        type=str.lower, default='mnist',
        choices=['mnist'])
    parser.add_argument(
        '--data-workers',
        # default = len(os.sched_getaffinity(0))
        type=int, default=0,
        metavar='NWORKERS',
        help='number of data loader workers')


    pcae_args = parser.add_argument_group('PCAE Parameters')
    pcae_args.add_argument(
        '--pcae-num-caps',
        type=int, default=16,
        metavar='PCAE_NCAPS',
        help='number of capsules')
    pcae_args.add_argument(
        '--pcae-caps-dim',
        type=int, default=6,
        help='number of dimensions per capsule')
    pcae_args.add_argument(
        '--pcae-feat-dim',
        type=int, default=16,
        help='number of feature dimensions per capsule')
    pcae_args.add_argument(
        '--pcae-lr',
        type=float, default=1e-4,
        help='learning rate')
    # .998 = 1-(1-.96)**1/20, equiv to .96 every 20 epochs
    pcae_args.add_argument(
        '--pcae-lr-decay',
        type=float, default=.998,
        help='learning rate decay')
    pcae_args.add_argument(
        '--pcae-weight-decay',
        type=float, default=.01,
        help='weight decay')
    pcae_args.add_argument(
        '--alpha-channel',
        action='store_true', default=False,
        help='expect input images to have an alpha channel')


    ocae_args = parser.add_argument_group('OCAE Parameters')
    ocae_args.add_argument(
        '--ocae-lr',
        type=float, default=1e-1,
        help='learning rate')
    ocae_args.add_argument(
        '--ocae-lr-decay',
        type=float, default=0,
        help='learning rate decay')
    ocae_args.add_argument(
        '--ocae-weight-decay',
        type=float, default=0,
        help='weight decay')


    logger_args = parser.add_argument_group('Logger Parameters')
    logger_args.add_argument(
        '--log-run-name',
        type=str, default=None,
        help='run name')
    logger_args.add_argument(
        '--log-project',
        type=str, default='SCAE',
        help='project name')
    logger_args.add_argument(
        '--log-upload',
        action='store_true', default=False,
        help='upload logs to W&B')
    logger_args.add_argument(
        '--log-frequency',
        type=int, default=1,
        help='logging frequency')

    return EasyDict(vars(parser.parse_args()))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import novel_objects.src as src

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        #train
        self.parser.add_argument('--project_home', default=src.top_dir(), help='project home')
        self.parser.add_argument('--dataset_path', default=src.top_dir() + "/data", help='data home')
        self.parser.add_argument('--ckpt_path', default=src.top_dir() + "/saved_models/", help='ckpt path')

        self.parser.add_argument('--img_height', default=64, type=int, help='img height')
        self.parser.add_argument('--img_width', default=64, type=int, help='img width')
        self.parser.add_argument('--use_augmentation', action='store_false', help='')

        self.parser.add_argument('--train_class_split', default=0.75, type=float, help='')
        self.parser.add_argument('--test_class_split', default=0.25, type=float, help='')

        self.parser.add_argument('--train_data_split', default=0.85, type=float, help='')
        self.parser.add_argument('--test_data_split', default=0.15, type=float, help='')

        self.parser.add_argument('--auto_lr_find', action='store_true', help='')
        self.parser.add_argument('--debug_lr', action='store_true', help='')

        self.parser.add_argument('--batch_size', default=32, type=int, help='')
        self.parser.add_argument('--max_epochs', default=1000, type=int, help='')
        self.parser.add_argument('--latent_dim', default=512, type=int, help='')
        self.parser.add_argument('--log_every_n_steps', default=50, type=int, help='')

        self.parser.add_argument('--loss_w1', default=0.66, type=float, help='')
        self.parser.add_argument('--loss_w2', default=0.34, type=float, help='')
        self.parser.add_argument('--use_l2', action='store_false', help='use weighted squared loss')
        self.parser.add_argument('--l2_scale', default=1.0, type=float, help='')

        self. parser.add_argument('--lr', default=1e-4, type=float, help='lr0 or init lr 1e-5 for Adam')
        self.parser.add_argument('--factor', default=0.75, type=float, help='plateau')
        self.parser.add_argument('--patience', default=100, type=float,
                                 help='patience forplateau, stepsize for step')
        self.parser.add_argument('--min_lr', default=5e-5, type=float, help='')
        self.parser.add_argument('--lrf', default=0.2, type=float, help='inal OneCycleLR learning rate (lr0 * lrf)')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='')
        self.parser.add_argument('--weight_decay', default=0.0005, type=float, help='')

        self.parser.add_argument('--scheduler_type', default='stepLR',
                                 help='reduce_lr_on_plateau, linear, one_cycle')

        self.parser.add_argument('--dataset', default='imagenet_mini', help='[imagenet_mini, CIFAR10]')
        self.parser.add_argument('--network', default='default', help='')

        self.parser.add_argument('--training_lib', default='lightning', help='pytorch or lightning')

        # eval
        self.parser.add_argument('--eval_exp_id', default=44, type=int, help='')
        self.parser.add_argument('--eval_ckpt_file', default='epoch=116-step=116648', help='')

        # debug
        self.parser.add_argument('--debug_data_mini', action='store_true', help='')
        self.parser.add_argument('--debug_visualize', action='store_true', help='')

        self.parser.add_argument('--use_mlflow', action='store_false', help='')

        # patch level
        self.parser.add_argument('--use_patch_op', action='store_true', help='')
        self.parser.add_argument('--patch_resolution', default=2, help='')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt

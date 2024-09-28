#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/18 上午10:49
# @Author  : CaoQixuan
# @File    : main.py
# @Description :
import argparse

import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="base_trans", type=str,
                    help='path to a yaml options file')
parser.add_argument('--lr', default=None, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--num_works', default=4, type=int, help='number of cpu')
parser.add_argument('--epochs', default=None, type=int, help='number of epochs')
parser.add_argument('--dataset', default="NWEP_RESISC45", type=str, help='dataset')
parser.add_argument('--hash_bit', default=None, type=int, help='dim')
parser.add_argument('--margin', default=None, type=float, help=' triplet margin')
parser.add_argument('--t', default=None, type=float, help='temperature')
parser.add_argument('--base', default=None, type=float, help='base')
parser.add_argument('--alpha', default=None, type=float, help='alpha')
parser.add_argument('--beta', default=None, type=float, help='beta')
parser.add_argument('--gamma', default=None, type=float, help='gamma')
parser.add_argument('--convert_weights', default=None, type=bool, help='convert_weights')
parser.add_argument('--save_state_dict', default=None, type=bool, help='save_state_dict')
parser.add_argument('--augment', action='store_true',
                    help="Whether to apply augmentation. Default is True, use '--augment' to set it to False.")
parser.add_argument('--aug_type', default='patchmask', help="Whether to apply augmentation type.")
parser.add_argument('--gpus', default="8, 7", type=str, help='gpu')
parser.add_argument('--teacher', default=None, type=str, help='base_distill teacher name')
parser.add_argument('--vit', default=None, type=str, help='base_distill teacher name')
parser.add_argument('--store_path', default="", type=str, help='base_distill teacher name')
parser.add_argument('--des', default=None, type=str, help='des')

opt = parser.parse_args()
options = yaml.load(open("models_data/" + opt.path + "/config.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
if opt.lr is not None:
    options['optimizer']['adam']['lr'] = opt.lr
if opt.batch_size is not None:
    options['train']['batch_size'] = opt.batch_size
if opt.num_works is not None:
    options['train']['num_works'] = opt.num_works
if opt.epochs is not None:
    options['train']['epoch'] = opt.epochs
if opt.dataset is not None:
    options['dataset']['type'] = opt.dataset
if opt.hash_bit is not None:
    options['model']['hash_bit'] = opt.hash_bit
if opt.margin is not None:
    options['loss']['margin'] = opt.margin
if opt.t is not None:
    options['loss']['T'] = opt.t
if opt.base is not None:
    options['loss']['base'] = opt.base
if opt.alpha is not None:
    options['loss']['alpha'] = opt.alpha
if opt.beta is not None:
    options['loss']['beta'] = opt.beta
if opt.gamma is not None:
    options['loss']['gamma'] = opt.gamma
if opt.convert_weights is not None:
    options['train']['convert_weights'] = opt.convert_weights
if opt.save_state_dict is not None:
    options['logs']['save_state_dict'] = opt.save_state_dict
if opt.teacher is not None:
    options['model']["teacher"]['save_name'] = opt.teacher
    options['model']["teacher"]['pre_train'] = True
if opt.augment is not None:
    options['dataset']['augment'] = opt.augment
    options["dataset"]["aug_type"] = opt.aug_type
if opt.store_path is not None:
    options['logs']['store_path'] = "logs/" + opt.path + "_" + opt.store_path + "/"
if opt.des is not None:
    options['logs']['describe'] = opt.des
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

if __name__ == "__main__":
    print(options)
    if opt.path == "base_distill":
        from train_student import main
    elif opt.path == "base_vit":
        from train_teacher import main
    else:
        print("Error")
        assert False
    if len(opt.gpus.split(',')) == 1:
        main(options)
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BS_ID"
        main(options, gpus=True)

# source activate CqxTorch
# cd /home/caoqixuan/GitImageRetrieval/
# python main.py --gpus 5,6 --batch_size 256 --path base_vit --hash_bit 128

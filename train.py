#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 下午5:00
# @Author  : CaoQixuan
# @File    : train.py
# @Description :
import gc
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from dataset import get_loader
from models.base_vit import base_vit
from models.base_distill import base_distall
from optimizer import Optimizer
from utils import save_log_txt, mark_validate, validate_with_no_cross, same_seeds, get_options, get_device


def train(opt, model, train_loader, test_loader, writer):
    model.train()
    optimizer = Optimizer(opt=opt["optimizer"], model=model, writer=writer)
    loss_memory = utils.LossMemory(writer)
    # train_loader, test_loder = get_loader(opt)
    model_name = opt["logs"]["store_path"] + "test.pt"
    best_metric_dict = {"map": 0}
    for epoch in range(opt['train']['epoch']):
        start_time = time.time()
        for data_pair, i in zip(train_loader, (range(len(train_loader)))):
            torch.cuda.empty_cache()
            gc.collect()
            loss_pair = model(data_pair, epoch)
            loss = loss_pair["total_loss"].mean()
            loss.backward()
            optimizer.step(epoch=epoch)
            loss_memory.append(epoch, loss_pair)
        end_time = time.time()
        mark_log = time.strftime(
            '[%Y-%m-%d %H:%M:%S]',
            time.localtime()) + " Epoch: [{:0>3d}/{:0>3d}] {} time: {:<5.2f}s\n".format(
            epoch, opt["train"]["epoch"], loss_memory.to_string(), end_time - start_time)
        save_log_txt(writer, mark_log)

        if epoch % opt["logs"]["eval_step"] == 0 and epoch >= opt["logs"]["start_eval_epoch"]:
            torch.cuda.empty_cache()
            metric_dict = validate_with_no_cross(opt, model, train_loader, test_loader)
            if metric_dict["map"] > best_metric_dict["map"]:
                best_metric_dict = metric_dict
                if opt["logs"]["save_state_dict"] and (epoch not in [2, 5, 10, 50]):
                    if os.path.isfile(model_name):
                        os.remove(model_name)
                    model_name = writer.log_dir + "%s_%s_%d_-%.4f.pt" % (
                        opt["model"]["name"], opt["dataset"]["type"], opt["model"]["hash_bit"],
                        best_metric_dict["map"])
                    if type(model) is torch.nn.DataParallel:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    torch.save(state_dict, model_name)
            mark_validate(opt, epoch, metric_dict, best_metric_dict, writer)


def main(opt):
    same_seeds(114514)
    # utils.generate_random_samples(opt)
    train_loader, test_loder = get_loader(opt)
    augment = "augment_" if opt["dataset"]["augment"] else ""
    filename_suffix = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    writer = SummaryWriter(log_dir=opt["logs"]["store_path"] + "%s%s_%s_%s_%d_%s/" % (
        augment, opt["model"]["name"], "float16" if opt["train"]["convert_weights"] else "float32",
        opt["dataset"]["type"], opt["model"]["hash_bit"], filename_suffix))
    if opt["model"]["name"] == "base_vit":
        model = base_vit.Network(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_distill":
        model = base_distall.Network(opt=opt, writer=writer)
    else:
        print("Unknown model")
        sys.exit()
    model.show()
    if opt["train"]["convert_weights"]:
        utils.convert_weights(model)
    model.to(get_device())
    utils.save_params_config(writer, opt)
    train(opt=opt, model=model, train_loader=train_loader, test_loader=test_loder, writer=writer)


if __name__ == "__main__":
    """
    base_vit train teacher network firstly
    base_distill train student network
    tensorboard --logdir=logs/model_name/mark_info/   # View training details
    """
    opt = get_options(model_name="base_distill")
    main(opt)

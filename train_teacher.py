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

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from dataset import get_loader
from models.base_cnn import base_cnn
from models.base_distill import base_distall
from models.base_distill2 import base_distall2
from models.base_vit import base_vit
from models.base_vit_plus import base_vit_plus
from models.base_work import base_work
from optimizer import Optimizer
from utils import save_log_txt, validate_with_no_cross, same_seeds, get_options, get_device


def train(opt, model, train_loader, test_loader, dataset_loader, writer):
    model.train()
    optimizer = Optimizer(opt=opt["optimizer"], model=model, writer=writer)
    loss_memory = utils.LossMemory(writer)
    # train_loader, test_loder = get_loader(opt)
    model_name = opt["logs"]["store_path"] + "test.pt"
    best_metric_dict = {"total_loss": 100000}
    for epoch in range(opt['train']['epoch']):
        start_time = time.time()
        for i, data_pair in enumerate(tqdm(train_loader, total=len(train_loader))):
            # break
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
            if loss_memory.average()["total_loss"] < best_metric_dict["total_loss"]:
                metric_dict = validate_with_no_cross(opt, model, dataset_loader, test_loader)
                save_log_txt(writer, metric_dict)
                save_log_txt(writer,"%.4f----->%.4f\n" % (best_metric_dict["total_loss"], loss_memory.average()["total_loss"]))
                best_metric_dict = loss_memory.average()
                if opt["logs"]["save_state_dict"]:
                    if os.path.isfile(model_name):
                        os.remove(model_name)
                    model_name = writer.log_dir + "%s_%s_%d_-%.4f.pt" % (
                        opt["model"]["name"], opt["dataset"]["type"], opt["model"]["hash_bit"],
                        best_metric_dict["total_loss"])
                    if type(model) is torch.nn.DataParallel:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()
                    torch.save(state_dict, model_name)
            # mark_validate(opt, epoch, metric_dict, best_metric_dict, writer)


def main(opt, gpus=False):
    same_seeds(114514)
    # utils.generate_random_samples(opt)
    train_loader, test_loader, dataset_loader = get_loader(opt)
    augment = "augment_" if opt["dataset"]["augment"] else ""
    filename_suffix = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    writer = SummaryWriter(log_dir=opt["logs"]["store_path"] + "%s%s_%s_%s_%d_%s/" % (
        augment, opt["model"]["name"], "float16" if opt["train"]["convert_weights"] else "float32",
        opt["dataset"]["type"], opt["model"]["hash_bit"], filename_suffix))
    if opt["model"]["name"] == "base_work":
        model = base_work.Network(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_vit":
        model = base_vit.Network(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_vit_plus":
        model = base_vit_plus.Network(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_distill":
        model = base_distall.Network(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_distill2":
        model = base_distall2.Network(opt=opt, writer=writer)
    elif opt["model"]["name"] == "base_cnn":
        model = base_cnn.Network(opt=opt, writer=writer)
    else:
        print("Unknown model")
        sys.exit()
    model.show()
    if opt["train"]["convert_weights"]:
        utils.convert_weights(model)
    if gpus:
        model = torch.nn.DataParallel(model, device_ids=utils.get_devices()).to(get_device())
    else:
        model.to(get_device())
    utils.save_params_config(writer, opt)
    start = time.time()
    train(opt=opt, model=model, train_loader=train_loader, test_loader=test_loader, dataset_loader=dataset_loader,
          writer=writer)
    end_time = time.time()
    save_log_txt(writer, "cost time: {} h".format((end_time - start) / 3600))


if __name__ == "__main__":
    """
    tensorboard --logdir=logs/base_clip/20231204-2015
    """
    opt = get_options(model_name="base_vit")
    main(opt)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 下午5:00
# @Author  : CaoQixuan
# @File    : optimizer.py
# @Description : learn from https://pytorch.org/docs/stable/optim.html
import torch.optim


class Optimizer:
    def __init__(self, opt, model, writer=None):
        self.writer = writer
        self.params = filter(lambda p: p.requires_grad, model.parameters())

        print("---------requires_grad=False---------")
        for param in filter(lambda p: not p[1].requires_grad, model.named_parameters()):
            print(param[0])
        print("---------requires_grad=False---------")

        self.opt = opt
        if opt["type"] == 'sgd':
            config = opt["sgd"]
            self.optimizer = torch.optim.SGD(self.params, lr=config["lr"], weight_decay=eval(config["weight_decay"]),
                                             momentum=config["momentum"])
        elif opt["type"] == 'adam':
            config = opt["adam"]
            self.optimizer = torch.optim.Adam(
                params=self.params,
                lr=config["lr"],
                betas=eval(config["betas"]),
                eps=eval(config["eps"]),
                weight_decay=eval(config["weight_decay"]),
            )
        else:
            raise NotImplementedError

        if self.opt["lr_scheduler_type"] == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=opt["step"]["decay_epoch"],
                gamma=opt["step"]["decay_param"],
            )
        elif self.opt["lr_scheduler_type"] == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=opt["cos"]["T_0"],
                T_mult=opt["cos"]["T_mult"],
                eta_min=opt["cos"]["eta_min"]
            )
        else:
            self.lr_scheduler = None

        if "norm_type" in opt:
            self.norm_type = opt["norm_type"]
        else:
            self.norm_type = 2
        if "max_grad_clip" in opt:
            self.max_grad_clip = opt["max_grad_clip"]
        else:
            self.max_grad_clip = 0

        self.stepNum = 0
        self.lastEpoch = 0
        self.optimizer.zero_grad()

    def step(self, epoch: int = None):
        if self.max_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.max_grad_clip, norm_type=self.norm_type)

        self.optimizer.step()

        if self.lr_scheduler is not None:
            if self.opt["lr_scheduler_type"] == "step":
                if (self.opt["step"]["restart_epoch"] != -1) and (epoch % self.opt["step"]["restart_epoch"] == 0) and (
                        epoch != 0):
                    self.lr_scheduler.__init__(optimizer=self.optimizer,
                                               step_size=self.opt["step"]["decay_epoch"],
                                               gamma=self.opt["step"]["decay_param"])
                if epoch != self.lastEpoch:
                    self.lr_scheduler.step()
                    self.lastEpoch = epoch
            else:
                self.lr_scheduler.step()

        self.optimizer.zero_grad()
        if self.writer is not None:
            self.writer.add_scalar("Optimizer", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=self.stepNum)
        self.stepNum += 1

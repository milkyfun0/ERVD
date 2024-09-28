# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 下午5:00
# @Author  : CaoQixuan
# @File    : optimizer.py
# @Description :
import torch
import torch.optim
from torch.optim import Optimizer as TorchOptimizer
from utils import save_log_txt


class Optimizer(TorchOptimizer):
    def __init__(self, opt, model, writer=None):
        # 将 params 转换为 filter 参数的格式
        self.opt = opt
        params = filter(lambda p: p.requires_grad, model.parameters())

        # 保存 writer 日志
        self.writer = writer
        save_log_txt(writer, "---------requires_grad=False---------\n")
        for param in filter(lambda p: not p[1].requires_grad, model.named_parameters()):
            save_log_txt(writer, str(param[0]) + "\n")
        save_log_txt(writer, "---------requires_grad=False---------\n")

        # 初始化 torch.optim.Optimizer 的参数字典
        defaults = {
            'lr': opt.get("lr", 0.001),
            'weight_decay': opt.get("weight_decay", 0)
        }
        super(Optimizer, self).__init__(params, defaults)

        # 初始化优化器类型
        if opt["type"] == 'sgd':
            config = opt["sgd"]
            self.optimizer = torch.optim.SGD(
                params=self.param_groups,
                lr=config["lr"],
                weight_decay=eval(config["weight_decay"]),
                momentum=config["momentum"]
            )
        elif opt["type"] == 'adam':
            config = opt["adam"]
            self.optimizer = torch.optim.Adam(
                params=self.param_groups,
                lr=config["lr"],
                betas=eval(config["betas"]),
                eps=eval(config["eps"]),
                weight_decay=eval(config["weight_decay"]),
            )
        else:
            raise NotImplementedError("Optimizer type not supported.")

        # 学习率调度器
        if opt["lr_scheduler_type"] == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=opt["step"]["decay_epoch"],
                gamma=opt["step"]["decay_param"],
            )
        elif opt["lr_scheduler_type"] == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=opt["cos"]["T_0"],
                T_mult=opt["cos"]["T_mult"],
                eta_min=opt["cos"]["eta_min"]
            )
        else:
            self.lr_scheduler = None

        # 设置梯度裁剪参数和范数类型
        self.norm_type = opt.get("norm_type", 2)
        self.max_grad_clip = opt.get("max_grad_clip", 0)

        self.stepNum = 0
        self.lastEpoch = 0
        self.optimizer.zero_grad()

    def step(self, epoch: int = None, closure=None):
        """
        Performs a single optimization step.
        """
        params = [p for group in self.param_groups for p in group['params'] if p.requires_grad]

        # 进行梯度裁剪
        if self.max_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_clip, norm_type=self.norm_type)

        # 调用底层优化器的 step 方法
        self.optimizer.step(closure)

        # 更新学习率调度器
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

        # 使用 writer 记录学习率
        if self.writer is not None:
            self.writer.add_scalar("Optimizer", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=self.stepNum)
        self.stepNum += 1

    def zero_grad(self, set_to_none=False):
        """
        Clears the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

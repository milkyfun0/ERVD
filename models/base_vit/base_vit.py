#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/12 下午3:55
# @Author  : CaoQixuan
# @File    : base_work.py
# @Description :
import torch.nn as nn

from dataset import Augment
from loss import calc_contrastive_loss, calc_triple_loss
from models.base_vit.tool import load_clip_from_json
from utils import params_count, save_log_txt


class Network(nn.Module):
    def __init__(self, opt, writer=None):
        super(Network, self).__init__()
        self.opt = opt
        self.visual, self.visual_config = load_clip_from_json(
            opt=opt,
            config_path=opt["model"]["CLIP"]["config_path"],
            pre_train=opt["model"]["CLIP"]["pre_train"],
            writer=writer
        )
        self.writer = writer
        self.augment = Augment(opt, self.visual_config["image_size"], self.visual_config["patch_size"])
        self.hash_proj = nn.Linear(self.visual.output_dim, opt["model"]["hash_bit"])
        self.initialize()

    def show(self):
        info = "--- Base ViT Network {}bits {} ---\n".format(self.opt["model"]["hash_bit"],
                                                             self.opt["dataset"]["type"])
        total = params_count(self)
        info += ' Model has {} parameters\n'.format(total)
        save_log_txt(self.writer, info)

    def initialize(self):
        nn.init.xavier_normal_(self.hash_proj.weight)

    def encode(self, data_pair):
        model_device = next(self.parameters()).device
        images = data_pair["image"].to(model_device)
        output = self.visual(images)
        output = self.hash_proj(output)
        return output

    def forward(self, data_pair, epoch):
        images, labels = self.augment(data_pair)
        output = self.visual(images)
        feature = self.hash_proj(output)
        total_loss = 0
        con_loss = calc_contrastive_loss(feature, feature, label_row=labels, label_col=labels, mask_diag=True,
                                         t=self.opt["loss"]["T"]) * self.opt["loss"]["base"]
        total_loss += con_loss

        triple_loss = calc_triple_loss(
            feature, labels, self.opt["loss"]["margin"]) * self.opt["loss"]["alpha"]
        total_loss += triple_loss

        return {
            "total_loss": total_loss,
            "con_loss": con_loss,
            "triple_loss": triple_loss,
        }

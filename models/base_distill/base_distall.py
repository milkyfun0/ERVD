#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 上午10:55
# @Author  : CaoQixuan
# @File    : base_distall.py
# @Description :
import torch.nn as nn

from dataset import Augment
from loss import calc_contrastive_loss, calc_triple_loss, calc_l2_loss
from models.base_distill import student
from models.base_distill import teacher
from models.common import AlignModulePart
from utils import params_count, save_log_txt, convert_weights


class AlignModule(nn.Module):
    def __init__(self, align_number, image_size, patch_size, teacher_width, student_width, alpha=0.5, share=False):
        super(AlignModule, self).__init__()
        self.align_number = align_number
        self.alignModule = nn.ModuleList(
            [AlignModulePart(image_size, patch_size, teacher_width, student_width, alpha, share=share) for _ in
             range(align_number)])

    def forward(self, teacher_low_features, student_low_features, global_token=None):
        skip = len(teacher_low_features) - self.align_number
        loss = 0
        for i in range(len(self.alignModule)):
            loss += self.alignModule[i].forward(teacher_low_features[i + skip], student_low_features[i + skip],
                                                global_token=global_token)
        return loss


class Network(nn.Module):
    def __init__(self, opt, writer=None):
        super(Network, self).__init__()
        self.opt = opt
        self.writer = writer
        self.visual_student, self.visual_config = student.load_clip_from_json(
            opt=opt,
            config_path=opt["model"]["student"]["config_path"],
            pre_train=opt["model"]["student"]["pre_train"],
            writer=writer
        )
        self.network_teacher = teacher.load_teacher_from_json(
            opt=opt,
            config_path=opt["model"]["teacher"]["config_path"],
            pre_train=opt["model"]["teacher"]["pre_train"],
            writer=writer
        )
        self.augment = Augment(self.visual_config["image_size"], self.visual_config["patch_size"])
        self.hash_proj_student = nn.Linear(self.visual_student.output_dim, opt["model"]["hash_bit"])
        self.alignModule = AlignModule(
            self.visual_config["layers"] - student.freeze_number - 1, self.visual_config["image_size"],
            self.visual_config["patch_size"], self.network_teacher.visual.width, self.visual_student.width)
        self.initialize()

        convert_weights(self.network_teacher)

    def show(self):
        info = "--- Distill ViT Network {}bits {} ---\n".format(self.opt["model"]["hash_bit"],
                                                                self.opt["dataset"]["type"])
        total = params_count(self)
        info += ' Model has {} parameters\n'.format(total)
        info += ' teacher has {} parameters\n'.format(params_count(self.network_teacher))
        info += ' student has {} parameters\n'.format(params_count(self.visual_student))
        info += ' align has {} parameters\n'.format(params_count(self.alignModule))
        save_log_txt(self.writer, info)

    def initialize(self):
        nn.init.xavier_normal_(self.hash_proj_student.weight)

    @property
    def dtype(self):
        return self.hash_proj_student.weight.dtype

    def encode(self, data_pair):
        model_device = self.augment.device
        images = data_pair["image"].to(model_device).type(self.dtype)
        output = self.visual_student(images)
        output = self.hash_proj_student(output)
        return output

    def forward(self, data_pair, epoch):
        images, labels = self.augment(data_pair)
        total_loss = 0

        #
        output_student = self.visual_student(images)
        feature_student = self.hash_proj_student(output_student)

        #
        con_loss_student = calc_contrastive_loss(
            feature_student, feature_student, label_row=labels, label_col=labels, mask_diag=True,
            t=self.opt["loss"]["T"]) * self.opt["loss"]["base"]
        total_loss += con_loss_student

        triple_loss_student = calc_triple_loss(
            feature_student, labels, self.opt["loss"]["margin"]) * self.opt["loss"]["alpha"]
        total_loss += triple_loss_student
        feature_teacher = self.network_teacher(images).detach()
        #
        global_align_loss = calc_l2_loss(feature_student, feature_teacher.detach()) * self.opt["loss"]["beta"]
        total_loss += global_align_loss

        low_align_loss = self.alignModule(
            self.network_teacher.half_feature, self.visual_student.half_feature) * self.opt["loss"]["gamma"]
        total_loss += low_align_loss

        return {
            "total_loss": total_loss,
            "con_loss_student": con_loss_student,
            "triple_loss_student": triple_loss_student,
            "global_align_loss": global_align_loss,
            "low_align_loss": low_align_loss
        }

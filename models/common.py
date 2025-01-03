#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/21 下午2:22
# @Author  : CaoQixuan
# @File    : common.py
# @Description :
import math

import torch
from torch import nn
from torch.nn import functional as F

from Collection import VarCollection
from loss import calc_triple_loss, calc_l2_loss
from utils import calc_distance


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LocalSelect(nn.Module):
    def __init__(self, image_size, patch_size, window_size, dim, hash_bit, share_linear=None):
        super(LocalSelect, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.image_height = image_size // patch_size

        self.index_select = nn.Parameter(
            data=self.get_index_select(), requires_grad=False)  # 1, shift_window_number, image_height**2,1
        self.shift_window_patch_numbers = nn.Parameter(
            data=torch.sum(self.index_select, dim=-2) * 1.0, requires_grad=False)  # 1, shift_window_number, 1
        self.gaussian_prob = nn.Parameter(data=self.get_gaussian_prob(sigma=self.image_height // 2),
                                          requires_grad=False)  # 1, shift_window_number, 1
        if share_linear is not None:
            self.linear = share_linear
        else:
            self.linear = nn.Linear(dim, hash_bit)

    def get_shift_window_matrix(self):
        shift_window_matrix = torch.zeros(self.image_height, self.image_height)
        for i in range(self.image_height):
            for j in range(self.image_height):
                row_times = (i + self.window_size // 2) // self.window_size
                col_times = (j + self.window_size // 2) // self.window_size
                shift_window_matrix[i, j] = row_times * (self.image_height // self.window_size + 1) + col_times + 1
        return shift_window_matrix

    def get_index_matrix(self):
        index_matrix = torch.arange(0, self.image_height ** 2).reshape(self.image_height, self.image_height)
        return index_matrix

    def get_index_select(self):
        shift_window_matrix = self.get_shift_window_matrix().reshape(1, self.image_height, self.image_height)
        shift_window_number = int(torch.max(shift_window_matrix).item())
        shift_window_index = torch.arange(1, shift_window_number + 1).reshape(-1, 1, 1)
        index_select = (shift_window_matrix == shift_window_index).to(
            torch.int32).reshape(1, shift_window_number, self.image_height ** 2, 1)  # 1 这个维度为batch广播准备
        return index_select

    def get_gaussian_prob(self, sigma):
        gaussian_1D = torch.Tensor(
            [math.exp(-(x - self.image_height // 2) ** 2 / float(2 * sigma ** 2)) for x in
             range(self.image_height)]).unsqueeze(1)
        gaussian_2D = torch.mm(gaussian_1D, gaussian_1D.T)
        gaussian_2D = gaussian_2D / gaussian_2D.sum()
        prob = torch.sum(self.index_select * gaussian_2D.reshape(1, 1, -1, 1), dim=2).reshape(1, -1, 1)
        return prob

    def similarity_weight(self, weight_type="prob"):
        if weight_type == "prob":
            weight = self.shift_window_patch_numbers / torch.sum(self.shift_window_patch_numbers)
        elif weight_type == "softmax":
            weight = F.normalize(self.shift_window_patch_numbers)
            weight = F.softmax(weight, dim=1)
        elif weight_type == "gaussian":
            weight = self.gaussian_prob
        else:
            raise NotImplementedError
        return weight.detach()

    def get_shift_window_embedding(self, tokens, proj=True):
        if proj:
            tokens = self.linear(tokens)
        window_tokens = tokens.unsqueeze(1) * self.index_select  # b, shift_window_number, patch_number, dim
        shift_window_embedding = torch.sum(
            window_tokens, dim=-2) / self.shift_window_patch_numbers  # patch pooling: b, shift_window_number, dim
        shift_window_embedding = shift_window_embedding.transpose(0, 1)
        return shift_window_embedding

    def forward(self, tokens, labels):
        """
        :param labels:
        :param tokens: batch, patch_number, dim
        """
        shift_window_embedding = self.get_shift_window_embedding(tokens)

        sim = calc_distance(
            shift_window_embedding, shift_window_embedding, dis_type="batch_dot")  # shift_window_patch_number, b, b
        weight = self.similarity_weight("prob").reshape(-1, 1, 1)  # shift_window_patch_number, 1, 1
        sim = torch.sum(weight * sim, dim=0)  # b, b
        # sim = torch.max(sim, dim=0)[0]
        #
        loss = calc_triple_loss(sim, labels, 0.5)
        #
        # mask = (labels.reshape(-1, 1) == labels.reshape(1, -1)).to(torch.int32)
        # mask = mask * (1 - torch.eye(tokens.shape[0], device=tokens.device))
        #
        # loss = calc_contrastive_loss_part(sim, mask)

        return loss


class AlignModulePart(nn.Module):
    def __init__(self, image_size, patch_size, teacher_width, student_width, alpha=0.5, share=False):
        super(AlignModulePart, self).__init__()

        self.alpha = alpha
        self.patch_number = (image_size // patch_size) ** 2
        self.need_proj = teacher_width != student_width
        share_linear = None
        if self.need_proj:
            self.linear = nn.Linear(teacher_width, student_width)
            if share:
                share_linear = self.linear
        self.localSelect = LocalSelect(image_size, patch_size, window_size=7, dim=teacher_width, hash_bit=student_width,
                                       share_linear=share_linear)

    def forward(self, teacher_tokens, student_tokens, global_token=None):
        if global_token is None:
            teacher_global_token = teacher_tokens[:, 0]
        else:
            teacher_global_token = global_token[:, 0]
        if self.need_proj:
            teacher_global_token = self.linear(teacher_global_token)
        student_global_token = student_tokens[:, 0]
        global_dis = calc_l2_loss(teacher_global_token, student_global_token)

        # b, shift_window_number, dim
        teacher_local_windows = self.localSelect.get_shift_window_embedding(
            teacher_tokens[:, 1:self.patch_number + 1], proj=self.need_proj).transpose(0, 1)
        student_local_windows = self.localSelect.get_shift_window_embedding(
            student_tokens[:, 1:self.patch_number + 1], proj=False).transpose(0, 1)
        local_dis = torch.norm(teacher_local_windows - student_local_windows, dim=-1, p=2)  # b, shift_window_number
        window_percent = self.localSelect.similarity_weight().squeeze(
            -1)  # 1, shift_window_number
        local_dis = torch.sum(local_dis * window_percent) / len(local_dis)

        loss = global_dis + local_dis * self.alpha
        return loss

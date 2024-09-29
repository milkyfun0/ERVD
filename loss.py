#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 下午5:00
# @Author  : CaoQixuan
# @File    : loss.py.py
# @Description :
import torch
from torch import nn

from utils import calc_distance


def calc_contrastive_loss(
        feature_row: torch.Tensor,
        feature_col: torch.Tensor,
        label_row: torch.Tensor,
        label_col: torch.Tensor,
        mask_diag: bool = False,
        div: int = 1,
        t: float = 0.1,
        eps: float = 1e-10
):
    """
    contrastive_loss
    :param div:
    :param feature_row: (b, n)
    :param feature_col: (b, n)
    :param label_row: (b, -1)
    :param label_col: (b, -1)
    :param mask_diag:  True: diag is not pair, False: diag is not pair
    :param t: temperature
    :param eps:
    :return:
    """
    assert feature_row.shape == feature_col.shape

    label_col = torch.div(label_col, div, rounding_mode="floor")

    feature_row = feature_row / feature_row.norm(dim=1, keepdim=True)
    feature_col = feature_col / feature_col.norm(dim=1, keepdim=True)
    # print(label_row.reshape(-1, 1) == label_col.reshape(1, -1))
    mask = (label_row.reshape(-1, 1) == label_col.reshape(1, -1)).to(torch.int32)

    mask_diag = (1 - torch.eye(feature_row.shape[0], device=feature_row.device)) if mask_diag else torch.ones_like(
        mask, device=feature_row.device)

    mask = mask * mask_diag
    row_col = feature_row @ feature_col.T / t * mask_diag
    col_row = feature_col @ feature_row.T / t * mask_diag

    row_col_loss = calc_contrastive_loss_part(sim=row_col, mask=mask, eps=eps)
    col_row_loss = calc_contrastive_loss_part(sim=col_row, mask=mask.T, eps=eps)

    return (row_col_loss + col_row_loss) / 2


def calc_contrastive_loss_part(
        sim: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-10
):
    """
    :param sim: (b, b)
    :param mask: (b, b)
    :param eps:
    :return:
    """

    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max
    exp_sim = torch.exp(sim)
    sim = sim * mask

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    mask_sum = mask.sum(dim=1)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

    loss = -1 * (mask * log_prob).sum(dim=1) / mask_sum.detach()
    return loss.mean()


def calc_triple_loss(features: torch.Tensor, labels, margin: torch.float):
    """
    learn from git from SCFR : https://github.com/xdplay17/SCFR
    :param features:
    :param labels:
    :param margin:
    :return:
    triple3
    """
    R = (labels.reshape(-1, 1) == labels.reshape(1, -1)).to(torch.int32)
    scores = calc_distance(features, features, dis_type="hamming")
    loss = R * scores + (1.0 - R) * (margin - scores).clamp(min=0.0)
    loss_mean = loss.sum() / (features.size(0) ** 2)
    return loss_mean


#     return loss


def calc_cluster_loss(feature: torch.Tensor, centers: torch.Tensor, labels: torch.Tensor, margin,
                      num_class) -> torch.Tensor:
    # git from SCFR : https://github.com/xdplay17/SCFR
    class_label = torch.tensor(range(num_class)).to(feature.device)
    # 化为 one hot
    R = (labels.unsqueeze(1) == class_label.unsqueeze(0)).float()

    # Hamming Distance
    features = feature.sign()
    centers = centers.sign()
    q = features.shape[1]
    distance = 0.5 * (q - torch.matmul(features, centers.T))

    loss = R * distance + (1.0 - R) * (margin - distance).clamp(min=0.0)
    loss_mean = loss.sum() / (features.size(0) * centers.size(0))

    return loss_mean


def calc_cls_loss(logistic: torch.Tensor, labels: torch.Tensor):
    """
    CrossEntropyLoss; if dataset distribute is not balanced please use focal loss: $-(1-p_t)^\\gamma log(p_t) $
    test ok
    """
    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    if labels is not torch.LongTensor:
        labels = labels.to(torch.long)
    loss = cross_entropy(logistic, labels.flatten())

    return loss


def calc_l2_loss(feature_student: torch.Tensor, feature_teacher: torch.Tensor):
    distance = torch.norm(feature_student - feature_teacher)
    return distance.sum() / feature_student.size(0)

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


class CodeBook(nn.Module):
    # learn from Vector [Quantization with Self-attention for Quality-independent Representation Learning]
    def __init__(self, dim, book_size, attn_mask=None, alpha=0.5):
        super(CodeBook, self).__init__()
        self.dim = dim
        self.attn_mask = attn_mask
        self.alpha = alpha
        self.book_size = book_size
        # self.code_book_value = torch.tensor(numpy.load("analysis/center-1.8M.npy"), dtype=torch.float32)
        self.code_book_dim = 64
        self.code_book = nn.Embedding(book_size, self.code_book_dim)

        self.K = nn.Parameter(torch.FloatTensor(dim, dim), requires_grad=True)
        self.Q = nn.Parameter(torch.FloatTensor(dim, dim), requires_grad=True)
        self.V = nn.Parameter(torch.FloatTensor(dim, dim), requires_grad=True)
        self.loss = None
        self.fc_fuse = nn.Sequential(nn.Linear(dim * 2, dim),
                                     QuickGELU(), )
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.code_book.weight, 0, 3)
        nn.init.kaiming_normal_(self.K)
        nn.init.kaiming_normal_(self.Q)
        nn.init.kaiming_normal_(self.V)

    @VarCollection("att_weight")
    def attention(self, tokens: torch.Tensor, attn_mask: torch = None):
        tokens = self.att_ln_post(tokens)
        q, k, v = self.proj_q(tokens), self.proj_k(tokens), self.proj_v(tokens)
        self.attn_mask = self.attn_mask.to(dtype=tokens.dtype,
                                           device=tokens.device) if self.attn_mask is not None else None
        self.attn_mask = None if attn_mask is None else self.attn_mask
        tokens, att_weight = self.attn(q, k, v, need_weights=True, key_padding_mask=attn_mask,
                                       average_attn_weights=False,
                                       attn_mask=self.attn_mask)
        return self.att_ln_past(tokens)

    def calc_loss(self, tokens: torch.Tensor, new_tokens: torch.Tensor, attn_mask: torch = None):
        q_latent_loss = F.mse_loss(new_tokens, tokens.detach())
        e_latent_loss = F.mse_loss(tokens, new_tokens.detach())
        loss = q_latent_loss + self.alpha * e_latent_loss

        if attn_mask is not None:
            loss = torch.masked_fill(loss, attn_mask, value=0)
        # self.loss = loss.mean() + calc_triple_loss(calc_distance(self.code_book.weight, self.code_book.weight, "dot"),
        #                                            0.2)
        self.loss = loss.mean()

    def get_loss(self):
        return self.image_loss

    def get_new_tokens(self, tokens):
        tokens = F.normalize(tokens.squeeze(), p=2, dim=-1)  # 有疑问？
        code = F.normalize(self.code_book.weight, p=2, dim=-1)

        # for batch in range(tokens.shape[0]):  ## 显存不够！！！！
        #     # tokens_ids = []
        #     # for token_id in range(tokens.shape[1]):
        #     #     temp = torch.norm(tokens[batch][token_id].unsqueeze(0) - code, dim=-1)
        #     #     tokens_ids.append(temp)
        #     # distance.append(torch.cat(tokens_ids, dim=0))
        #     distance.append(torch.norm(tokens[batch].unsqueeze(1) - code, dim=-1))
        distances = (
                torch.sum(tokens ** 2, dim=1, keepdim=True) +
                torch.sum(code ** 2, dim=1) -
                2. * torch.matmul(tokens, code.t())
        )  # [N, M]
        # distance = torch.cat(distance, dim=0)
        max_sim_id = torch.argmin(distances, dim=-1).reshape(-1, 1)
        new_token = self.code_book(max_sim_id).reshape(tokens.shape)

        return new_token

    @property
    def dtype(self):
        return self.code_book.weight.dtype

    def forward(self, tokens: torch.Tensor, attn_mask: torch = None):
        """
        class token 怎么做操作呢 需要进一步测试
        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        """
        bs = tokens.shape[0]
        tokens = tokens.type(self.dtype)
        cls_token = tokens[:, 0, :].reshape(-1, self.code_book_dim)

        new_tokens = self.get_new_tokens(cls_token)
        self.calc_loss(cls_token, new_tokens, attn_mask)
        # new_tokens = cls_token + (new_tokens - cls_token).detach()
        new_tokens = new_tokens
        # print(new_tokens.shape, cls_token.shape)
        fuse = torch.stack([cls_token.reshape(bs, self.dim), new_tokens.reshape(bs, self.dim)], dim=2)
        # self.calc_loss(tokens, new_tokens, attn_mask)

        # tokens = torch.cat([cls_token, tokens], dim=1)
        # new_tokens = torch.cat([cls_token, new_tokens], dim=1)

        K = torch.bmm(self.K.repeat(bs, 1, 1), fuse)  # b,d,2
        Q = torch.bmm(self.Q.repeat(bs, 1, 1), fuse)  # b.d.2
        V = torch.bmm(self.V.repeat(bs, 1, 1), fuse)  # b,d,2
        A = F.softmax(torch.bmm(K.permute(0, 2, 1), Q) / torch.sqrt(torch.tensor(2).to(fuse.device)), dim=1)  # b,2,2
        fuse = torch.bmm(V, A).permute(0, 2, 1).reshape(bs, -1).contiguous()
        fuse = self.fc_fuse(fuse)
        return fuse.unsqueeze(1)


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim, stride=2):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True))
        self.stride = stride
        self.ln_1 = LayerNorm(embed_dim)
        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            LayerNorm(embed_dim)
        )

    def tokens_attention(self, tokens):
        att = torch.mean(tokens, dim=2, keepdim=True)
        tokens = tokens * torch.sigmoid(att)
        return self.mlp(tokens)

    def forward(self, x):
        x = self.ln_2(x)
        x = x.permute(1, 0, 2)
        cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]

        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        tokens = self.proj(cnn_feat).flatten(2).transpose(1, 2)
        # tokens = tokens + self.tokens_attention(tokens)

        x = torch.cat((cls_token, tokens), dim=1)

        x = x.permute(1, 0, 2)
        x = self.ln_1(x)
        return x


class Adapter(nn.Module):
    def __init__(self, embed_dim):
        super(Adapter, self).__init__()

        self.down_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim * 4)
        self.up_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            QuickGELU(),
        )

    def init_weights(self):
        nn.init.kaiming_normal_(self.down_mlp[0].weight)
        nn.init.kaiming_normal_(self.up_mlp[0].weight)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output = self.down_mlp(x)
        output = self.layer_norm(output)
        output = self.up_mlp(output) + x
        return output.permute(1, 0, 2)


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

# class LocalSelect(nn.Module):
#     def __init__(self, image_size, patch_size, window_size, dim, hash_bit):
#         super(LocalSelect, self).__init__()
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.window_size = window_size
#         self.image_height = image_size // patch_size
#
#         self.index_select = nn.Parameter(
#             data=self.get_index_select(), requires_grad=False)  # 1, shift_window_number, image_height**2,1
#         self.shift_window_patch_numbers = nn.Parameter(
#             data=torch.sum(self.index_select, dim=-2) * 1.0, requires_grad=False)  # 1, shift_window_number, 1
#
#         self.linear = nn.Linear(dim, hash_bit)
#
#     def get_shift_window_matrix(self):
#         shift_window_matrix = torch.zeros(self.image_height, self.image_height)
#         for i in range(self.image_height):
#             for j in range(self.image_height):
#                 row_times = (i + self.window_size // 2) // self.window_size
#                 col_times = (j + self.window_size // 2) // self.window_size
#                 shift_window_matrix[i, j] = row_times * (self.image_height // self.window_size + 1) + col_times + 1
#         return shift_window_matrix
#
#     def get_index_matrix(self):
#         index_matrix = torch.arange(0, self.image_height ** 2).reshape(self.image_height, self.image_height)
#         return index_matrix
#
#     def get_index_select(self):
#         shift_window_matrix = self.get_shift_window_matrix().reshape(1, self.image_height, self.image_height)
#         shift_window_number = int(torch.max(shift_window_matrix).item())
#         shift_window_index = torch.arange(1, shift_window_number + 1).reshape(-1, 1, 1)
#         index_select = (shift_window_matrix == shift_window_index).to(
#             torch.int32).reshape(1, shift_window_number, self.image_height ** 2, 1)  # 1 这个维度为batch广播准备
#         return index_select
#
#     def similarity_weight(self, sim_type="prob"):
#         if sim_type == "prob":
#             weight = self.shift_window_patch_numbers / torch.sum(self.shift_window_patch_numbers)
#         elif sim_type == "softmax":
#             weight = F.normalize(self.shift_window_patch_numbers)
#             weight = F.softmax(weight, dim=1)
#         else:
#             raise NotImplementedError
#         # softmax
#         return weight
#
#     def get_shift_window_embedding(self, tokens, proj=True):
#         if proj:
#             tokens = self.linear(tokens)
#         window_tokens = tokens.unsqueeze(1) * self.index_select  # b, shift_window_number, patch_number, dim
#         shift_window_embedding = torch.sum(
#             window_tokens, dim=-2) / self.shift_window_patch_numbers  # patch pooling: b, shift_window_number, dim
#         shift_window_embedding = shift_window_embedding.transpose(0, 1)
#         return shift_window_embedding
#
#     def forward(self, tokens, labels):
#         """
#         :param labels:
#         :param tokens: batch, patch_number, dim
#         """
#         shift_window_embedding = self.get_shift_window_embedding(tokens)
#
#         sim = calc_distance(
#             shift_window_embedding, shift_window_embedding, dis_type="batch_dot")  # shift_window_patch_number, b, b
#         weight = self.similarity_weight("prob").reshape(-1, 1, 1)  # shift_window_patch_number, 1, 1
#         sim = torch.sum(weight * sim, dim=0)  # b, b
#         # sim = torch.max(sim, dim=0)[0]
#         #
#         loss = calc_triple_loss(sim, labels, 0.5)
#         print(loss)
#         #
#         # mask = (labels.reshape(-1, 1) == labels.reshape(1, -1)).to(torch.int32)
#         # mask = mask * (1 - torch.eye(tokens.shape[0], device=tokens.device))
#         #
#         # loss = calc_contrastive_loss_part(sim, mask)
#
#         return loss
#
#
# class AlignModulePart(nn.Module):
#     def __init__(self, image_size, patch_size, teacher_width, student_width, alpha=0.5):
#         super(AlignModulePart, self).__init__()
#
#         self.alpha = alpha
#         self.patch_number = (image_size // patch_size) ** 2
#         self.localSelect = LocalSelect(image_size, patch_size, window_size=7, dim=teacher_width, hash_bit=student_width)
#         self.proj = teacher_width != student_width
#         if self.proj:
#             self.linear = nn.Linear(teacher_width, student_width)
#
#     def forward(self, teacher_tokens, student_tokens):
#         teacher_global_token = teacher_tokens[:, 0]
#         if self.proj:
#             teacher_global_token = self.linear(teacher_global_token)
#         student_global_token = student_tokens[:, 0]
#         global_dis = calc_l2_loss(teacher_global_token, student_global_token)
#
#         # b, shift_window_number, dim
#         teacher_local_windows = self.localSelect.get_shift_window_embedding(
#             teacher_tokens[:, 1:self.patch_number + 1], proj=self.proj).transpose(0, 1)
#         student_local_windows = self.localSelect.get_shift_window_embedding(
#             student_tokens[:, 1:self.patch_number + 1], proj=False).transpose(0, 1)
#         local_dis = torch.norm(teacher_local_windows - student_local_windows, dim=-1, p=2)  # b, shift_window_number
#         window_percent = self.localSelect.similarity_weight().squeeze(-1)  # 1, shift_window_number
#         local_dis = torch.sum(local_dis * window_percent) / len(local_dis)
#
#         loss = global_dis + local_dis * self.alpha
#         return loss

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 上午10:55
# @Author  : CaoQixuan
# @File    : teacher.py
# @Description :
import json
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from Collection import VarCollection
from dataset import Augment
from utils import get_device


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self, embed_dim):
        super(Adapter, self).__init__()

        self.down_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            QuickGELU(),
        )
        self.layer_norm = LayerNorm(embed_dim * 2)
        self.up_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            QuickGELU(),
        )

    def init_weights(self):
        nn.init.kaiming_normal_(self.down_mlp[0].weight)
        nn.init.kaiming_normal_(self.up_mlp[0].weight)

    def forward(self, x):
        output = self.down_mlp(x)
        output = self.layer_norm(output)
        output = self.up_mlp(output)
        return output + x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super(ResidualAttentionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # n_head 头，d_model 表示维度。
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    @VarCollection("att_weight")
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x, att_weight = self.attn(x, x, x, need_weights=True, key_padding_mask=attn_mask, average_attn_weights=False,
                                  attn_mask=self.attn_mask)
        return x  # 三个x表示Q K V计算值，x最后维度=n_head*d_model

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, writer=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.writer = writer
        # self.peg = PosCNN(width, width)
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        # self.lowblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers // 5)])
        # self.linear = nn.Sequential(nn.Linear(2 * width, width), QuickGELU())

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        half_feature = []
        for i in range(self.layers):
            if i == self.layers // 2:
                x = x.detach()
            x = self.resblocks[i](x, attn_mask)
            if i % 2 == 1 and i >= self.layers // 2:
                half_feature.append(x.detach().permute(1, 0, 2))

        return x, half_feature


class VisionTransformer(nn.Module):
    # git from https://github.com/openai/CLIP
    def __init__(self, opt, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, writer=None):
        super(VisionTransformer, self).__init__()
        self.opt = opt
        self.width = width
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # width相当于transform中的d_model
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, writer=writer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.iter = 0

        # self.codebook = CodeBook(width, book_size=3096, alpha=0.5)

        self.writer = writer
        self.initialize_parameters()
        self.half_feature = None

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def get_tokens_avg(self):
        return torch.mean(self.x[:, 1:, :], dim=1, keepdim=False)

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: torch.Tensor):
        # x=[1,3,224,224]
        x = self.conv1(x)  # shape = [*, width, grid, grid] # 将图片分成[32,32]个patch [1,768,7,7]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2],合并高宽 [1,768,49]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] ，更换位置 [1,49,768]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width],添加cls token[1,50,768]
        x = x + self.positional_embedding.to(x.dtype)  # 这里位置编码是可学习的参数，可能是切了path顺序让模型自己学习吧  [1,50,768]
        x = self.ln_pre(x)  # [1,50,768]

        x = x.permute(1, 0, 2)  # NLD -> LND  # [pixel,b,d_model]=[50,1,768]
        # 当实例化时 batch——first默认维false
        x, self.half_feature = self.transformer(x)  # 多头transformer [50,1,768]
        x = x.permute(1, 0, 2)  # LND -> NLD  # [1,50,768]
        # x = self.codebook(x)

        x = self.ln_post(x[:, 0, :])  # x[:, 0, :] 将所有信息汇聚到cls token中，只需前面来做下游任务 [1,768]

        if self.proj is not None:  # self.proj是可学习参数，维度为[768,512]
            x = x @ self.proj  # 通过学习参数将维度再次融合变成512特征，最终为[1,512]

        return x


def load_clip_from_json(opt, config_path: str, pre_train: bool = True, writer=None):
    # print(config_path)
    with open(config_path + "config.json", 'r') as f:
        model_cfg = json.load(f)

    model = VisionTransformer(
        opt=opt,
        input_resolution=model_cfg["vision_cfg"]["image_size"],
        patch_size=model_cfg["vision_cfg"]["patch_size"],
        width=model_cfg["vision_cfg"]["width"],
        layers=model_cfg["vision_cfg"]["layers"],
        heads=model_cfg["vision_cfg"]["width"] // 64,
        output_dim=model_cfg["embed_dim"],
        writer=writer
    )
    return model, model_cfg["vision_cfg"]


class Network(nn.Module):
    def __init__(self, opt, writer=None):
        super(Network, self).__init__()
        self.opt = opt
        self.visual, self.visual_config = load_clip_from_json(
            opt=opt,
            config_path=opt["model"]["teacher"]["config_path"],
            pre_train=False,
            writer=writer
        )
        self.writer = writer
        self.augment = Augment(self.visual_config["image_size"], self.visual_config["patch_size"])
        self.hash_proj = nn.Linear(self.visual.output_dim, opt["model"]["hash_bit"])
        self.half_feature = None
        self.final_feature = None
        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.hash_proj.weight)

    def encode(self, data_pair):
        model_device = get_device()
        images = data_pair["image"].to(model_device)
        output = self.visual(images)
        self.final_feature = output
        output = self.hash_proj(output)
        return output

    def forward(self, images):
        target_dtype = images.dtype
        output = self.visual(images.type(self.visual.dtype))
        output = self.hash_proj(output).type(target_dtype)
        self.half_feature = [feature.type(target_dtype) for feature in self.visual.half_feature]
        self.final_feature = output.type(target_dtype)
        return output


def load_teacher_from_json(opt, config_path: str, pre_train: bool = True, writer=None):
    # print(config_path)
    with open(config_path + "config.json", 'r') as f:
        model_cfg = json.load(f)

    model = Network(opt, writer=writer)
    if pre_train:
        weights_map = torch.load(config_path + opt["model"]["teacher"]["save_name"])
        model.load_state_dict(weights_map, strict=False)

    return model

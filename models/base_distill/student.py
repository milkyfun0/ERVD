#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 下午12:24
# @Author  : CaoQixuan
# @File    : student.py
# @Description :
import json
from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from Collection import VarCollection
from models.common import Adapter

freeze_number = 3


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, is_adaptive: bool = False):
        super(ResidualAttentionBlock, self).__init__()

        self.is_adaptive = is_adaptive
        self.attn = nn.MultiheadAttention(d_model, n_head)  # n_head 头，d_model 表示维度。
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        if is_adaptive:
            self.adapter = Adapter(d_model)
            self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    @VarCollection("att_weight")
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x, att_weight = self.attn(x, x, x, need_weights=True, key_padding_mask=attn_mask, average_attn_weights=False,
                                  attn_mask=self.attn_mask)
        return x  # 三个x表示Q K V计算值，x最后维度=n_head*d_model

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if self.is_adaptive:
            step_1 = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
            step_2 = self.mlp(self.ln_2(step_1))
            step_3 = step_2 + step_1
            step_4 = self.adapter(step_3)
            output = self.ln_3(step_4 + step_1)
        else:
            x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
            output = x + self.mlp(self.ln_2(x))
        return output


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, writer=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.writer = writer
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for i in range(layers)])
        # self.prompt = nn.Parameter(width ** -0.5 * torch.randn(width))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        half_feature = []
        # x = x.permute(1, 0, 2)
        # prompt_token = self.prompt.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        # x = torch.cat([x, prompt_token], dim=1)
        # x = x.permute(1, 0, 2)
        for i in range(self.layers):
            if i == freeze_number:
                x = x.detach()
            x = self.resblocks[i](x, attn_mask)
            if i >= freeze_number:
                half_feature.append(x.permute(1, 0, 2))
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
        self.half_feature = None
        self.initialize_parameters()

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
        # self.half_feature = self.half_feature.permute(1, 0, 2).reshape(-1, self.width)
        # x = self.codebook(x)

        x = self.ln_post(x[:, 0, :])  # x[:, 0, :] 将所有信息汇聚到cls token中，只需前面来做下游任务 [1,768]

        if self.proj is not None:  # self.proj是可学习参数，维度为[768,512]
            x = x @ self.proj  # 通过学习参数将维度再次融合变成512特征，最终为[1,512]

        return x


def get_freeze_layers_names():
    names = [
        "positional_embedding",
        "class_embedding",
        "positional_embedding",
        "conv1.weight",
        "conv1.bias",
    ]
    for i in range(freeze_number):
        names.append("transformer.resblocks.{}.attn.in_proj_weight".format(i))
        names.append("transformer.resblocks.{}.attn.in_proj_bias".format(i))
        names.append("transformer.resblocks.{}.attn.out_proj.weight".format(i))
        names.append("transformer.resblocks.{}.attn.out_proj.bias".format(i))
        names.append("transformer.resblocks.{}.ln_1.weight".format(i))
        names.append("transformer.resblocks.{}.ln_1.bias".format(i))
        names.append("transformer.resblocks.{}.mlp.c_fc.weight".format(i))
        names.append("transformer.resblocks.{}.mlp.c_fc.bias".format(i))
        names.append("transformer.resblocks.{}.mlp.c_proj.weight".format(i))
        names.append("transformer.resblocks.{}.mlp.c_proj.bias".format(i))
        names.append("transformer.resblocks.{}.ln_2.weight".format(i))
        names.append("transformer.resblocks.{}.ln_2.bias".format(i))
    return names


def trans_standard_static_dict(state_dict: dict, layers=10):
    new_dict = {}
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    for k, v in state_dict.items():
        if "_image_encoder" in k:
            new_dict[k[22:]] = v
        elif "_text_encoder.module." in k:
            new_dict[k[21:]] = v
        elif "_logit_scale.module." in k:
            new_dict[k[20:]] = v
        elif "module." in k:
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v

    state_dict = {}
    for k, v in new_dict.items():
        if "visual." in k:
            k = k[7:]
            state_dict[k] = v
    # state_dict["proj"] = torch.cat([state_dict["proj"], state_dict["proj"]], dim=0)
    # state_dict["ln_post.weight"] = torch.cat([state_dict["ln_post.weight"], state_dict["ln_post.weight"]], dim=-1)
    # state_dict["ln_post.bias"] = torch.cat([state_dict["ln_post.bias"], state_dict["ln_post.bias"]], dim=-1)
    # last_layer = 5
    # for i in [0, 1]:
    # state_dict["transformer.lowblocks.{}.attn.in_proj_weight".format(i)] = state_dict[
    #     "transformer.resblocks.{}.attn.in_proj_weight".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.attn.in_proj_bias".format(i)] = state_dict[
    #     "transformer.resblocks.{}.attn.in_proj_bias".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.attn.out_proj.weight".format(i)] = state_dict[
    #     "transformer.resblocks.{}.attn.out_proj.weight".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.attn.out_proj.bias".format(i)] = state_dict[
    #     "transformer.resblocks.{}.attn.out_proj.bias".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.mlp.c_fc.weight".format(i)] = state_dict[
    #     "transformer.resblocks.{}.mlp.c_fc.weight".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.mlp.c_fc.bias".format(i)] = state_dict[
    #     "transformer.resblocks.{}.mlp.c_fc.bias".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.mlp.c_proj.weight".format(i)] = state_dict[
    #     "transformer.resblocks.{}.mlp.c_proj.weight".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.mlp.c_proj.bias".format(i)] = state_dict[
    #     "transformer.resblocks.{}.mlp.c_proj.bias".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.ln_1.weight".format(i)] = state_dict[
    #     "transformer.resblocks.{}.ln_1.weight".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.ln_1.bias".format(i)] = state_dict[
    #     "transformer.resblocks.{}.ln_1.bias".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.ln_2.weight".format(i)] = state_dict[
    #     "transformer.resblocks.{}.ln_2.weight".format(last_layer)]
    # state_dict["transformer.lowblocks.{}.ln_2.bias".format(i)] = state_dict[
    #     "transformer.resblocks.{}.ln_2.bias".format(last_layer)]

    return state_dict


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
    if pre_train:
        weights_map = trans_standard_static_dict(torch.load(config_path + opt["model"]["student"]["save_name"]))
        freeze_names = get_freeze_layers_names()
        for name, param in model.named_parameters():
            if name in freeze_names:
                param.requires_grad = False

        model.load_state_dict(weights_map, strict=False)

    return model, model_cfg["vision_cfg"]

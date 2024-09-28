#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 下午5:00
# @Author  : CaoQixuan
# @File    : utils.py.py
# @Description :
import argparse
import gc
import os
import random
import time

import numpy
import torch
import yaml
from torch import nn
from tqdm import tqdm


class LossMemory:
    def __init__(self, writer):
        self.writer = writer
        self.new_epoch = -1
        self.step = -1
        self.losses = {
        }

    def append(self, epoch, loss_dict):
        self.step += 1
        if epoch != self.new_epoch:
            self.new_epoch = epoch
            for k, v in loss_dict.items():
                self.losses[k] = []
        for k, v in loss_dict.items():
            if type(v) == torch.Tensor:
                v = v.cpu().detach().sum().item()
            self.losses[k].append(v)

    def clear(self):
        self.losses = {}
        self.new_epoch = -1
        self.step = -1

    def average(self):
        loss = {}
        for k, v in self.losses.items():
            loss[k] = sum(v) / len(v)
        return loss

    def to_string(self):
        info_string = ""
        for k, v in self.losses.items():
            info_string += "{}: {:<5.2f} ".format(k, sum(v) / len(v))
            self.writer.add_scalar("loss/%s" % k, sum(v) / len(v), self.new_epoch)
        return info_string


def get_devices(device_id=None):
    # learn from https://zh-v2.d2l.ai/index.html
    if device_id is None:
        return list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
    if device_id is []:
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device(f'cuda:{i}') for i in device_id]
    return devices if devices else [torch.device('cpu')]


def get_device(i=None):
    if i is None:
        i = int(get_devices()[0])
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def same_seeds(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        if not p.requires_grad:
            continue
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def get_options(model_name="base_clip"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default="models_data/" + model_name + "/config.yaml", type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()
    with open(opt.path_opt, 'r', encoding='utf-8') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)

    return options


def process(s: str):
    for i in [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']:
        s = s.strip().lower().replace(i, " " + i + " ")
    return s.strip()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
    return True


def generate_random_samples(options, percent=0.7):
    items = open(options["dataset"][options["dataset"]["type"]]["path"] + "dataset.txt",
                 encoding="utf-8").readlines()
    data_dict = {}
    for image_info in items:
        image_path, class_list = eval(image_info)
        image_class = image_path.split('/')[0]
        if image_class not in data_dict:
            data_dict[image_class] = [image_info]
        else:
            data_dict[image_class].append(image_info)
    train_samples = []
    test_samples = []
    for k, v in data_dict.items():
        random.shuffle(v)
        num_train = int(len(v) * percent)
        train_samples.extend(v[:num_train])
        test_samples.extend(v[num_train:])

    def split_write(data, data_type):
        with open(options["dataset"][options["dataset"]["type"]]["path"] + '{}.txt'.format(data_type), "w",
                  encoding="utf-8") as file:
            file.writelines("".join(data))

    split_write(train_samples, "train")
    split_write(test_samples, "test")
    print("Generate {} random samples to {} complete.train={} val={}".format(
        options["dataset"][options["dataset"]["type"]]["name"],
        options["dataset"][options["dataset"]["type"]]["path"],
        len(train_samples),
        len(test_samples)))


def save_log_txt(writer, log: str):
    print(log, end="")
    with open(writer.log_dir + "log.txt", mode="a+", encoding="utf-8") as f:
        f.write(str(log))


def save_params_config(writer, params: dict):
    with open(writer.log_dir + "config.yaml", mode="w", encoding="utf-8") as f:
        yaml.dump(params, f)


def mark_validate(opt, step: int, metric_pair: dict, best_metric_dict: dict, writer):
    mark = "{:*^100}\n".format(
        time.strftime('[%Y-%m-%d %H:%M:%S] ', time.localtime()) + "validate Epoch: [{:0>3d}/{:0>3d}]".format(
            step, opt["train"]["epoch"])
    )
    info = ("Now mAP: {:<5.4} recall: {:<5.4} precision: {:<5.4} | Best mAP: {:<5.4} recall: {:<5.4} precision: {"
            ":<5.4}\n").format(
        metric_pair["map"], metric_pair["recall"], metric_pair["precision"], best_metric_dict["map"],
        best_metric_dict["recall"], best_metric_dict["precision"])
    writer.add_scalar("map", scalar_value=metric_pair["map"], global_step=step)
    writer.add_scalar("recall", scalar_value=metric_pair["recall"], global_step=step)
    writer.add_scalar("precision", scalar_value=metric_pair["precision"], global_step=step)
    save_log_txt(writer, mark + info)


def calc_distance(matrix1: torch.Tensor, matrix2: torch.Tensor, dis_type='hamming', t=0):
    assert matrix1.shape[1] == matrix2.shape[1]
    if dis_type == 'hamming':
        # git from SCFR : https://github.com/xdplay17/SCFR
        q = matrix2.shape[1]
        distance = 0.5 * (q - torch.matmul(matrix1.sign(), matrix2.T.sign()))
    elif dis_type == 'dot':
        matrix1 = torch.nn.functional.normalize(matrix1, 2, -1)
        matrix2 = torch.nn.functional.normalize(matrix2, 2, -1)
        distance = torch.matmul(matrix1, matrix2.T)
    elif dis_type == 'batch_dot':
        matrix1 = torch.nn.functional.normalize(matrix1, 2, -1)
        matrix2 = torch.nn.functional.normalize(matrix2, 2, -1)
        distance = torch.bmm(matrix1, matrix2.transpose(-1, -2))
    elif dis_type == 'batch_contrastive':
        matrix1 = matrix1 / matrix1.norm(dim=-1, keepdim=True)
        matrix2 = matrix2 / matrix2.norm(dim=-1, keepdim=True)
        distance = torch.bmm(matrix1 / t, matrix2.transpose(-1, -2) / t)
    elif dis_type == 'l2':
        distance = (
                torch.sum(matrix1 ** 2, dim=1, keepdim=True) +
                torch.sum(matrix2 ** 2, dim=1) -
                2. * torch.matmul(matrix1, matrix2.t())
        )  # [N, M]
    else:
        raise TypeError
    return distance


@torch.no_grad()
def validate_on_tensor(sim: torch.Tensor, test_labels: torch.Tensor, train_labels: torch.Tensor, metric_dict,
                       descending=False):
    """
    :param sim:
    :param test_labels:
    :param train_labels:
    :param metric_dict:
    :param descending:
    :return:
    test code:
    test_features = torch.load("test_binary.pkl")
    test_labels = torch.argmax(torch.load("test_label.pkl"), dim=1)
    train_features = torch.load("database_binary.pkl")
    train_labels = torch.argmax(torch.load("database_label.pkl"), dim=1)
    sim = calc_distance(test_features, train_features, dis_type='hamming')
    print(sim.shape, test_labels.shape, train_labels.shape)
    result = validate_on_tensor(sim, test_labels, train_labels, metric_dict={"M": -1, "R": -1, "P": -1})
    print(result)
    """
    M, R, K = metric_dict["M"], metric_dict["R"], metric_dict["P"]
    test_len, train_len = sim.shape
    test_labels = test_labels.reshape(-1, 1).expand(test_len, train_len)
    train_labels = train_labels.reshape(1, -1).expand(test_len, train_len)
    pre_ids = torch.gather(train_labels, dim=1, index=torch.argsort(sim, dim=1, descending=descending))

    flags = (test_labels == pre_ids)
    if M == -1:
        map_flags = flags
    else:
        map_flags = flags[:, 0:M]
    if R == -1:
        recall_flags = flags
    else:
        recall_flags = flags[:, 0:R]

    def calc_map(mask: numpy.array):
        if numpy.sum(mask) == 0:
            return 0
        tIndex = numpy.asarray(numpy.where(mask == True)).flatten() + 1.0
        count = numpy.linspace(start=1, stop=len(tIndex), num=len(tIndex))
        return numpy.mean(count / tIndex)

    result_map = numpy.apply_along_axis(calc_map, axis=1, arr=map_flags.cpu().numpy())
    return {
        "map": numpy.sum(result_map) / len(numpy.where(result_map != 0)[0]),
        "recall": (torch.sum(recall_flags, dim=1) / torch.sum(flags, dim=1)).mean().detach().cpu().item(),
        "precision": -1.0 if K == -1 else (torch.sum(flags[:, 0:K], dim=1) / K).mean().detach().cpu().item(),
    }


@torch.no_grad()
def validate_with_no_cross(opt, model, train_loader, test_loader):
    """
    :param train_loader:
    :param test_loader:
    :param model:
    :param opt:
    :return:
    """
    model_device = get_device()  # 获取要使用的设备，如 'cuda:0' 或 'cpu'

    # 如果模型被封装为 DataParallel，提取其中的原始模型
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # 解封装 DataParallel

    # 设置模型到单个设备
    model.to(model_device)

    # 获取数据集长度和其他相关参数
    train_len = len(train_loader.dataset)
    test_len = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    data_type = torch.float16  # 使用的精度类型
    embedding_dim = opt["model"]["hash_bit"]  # 嵌入维度

    # 创建空的张量来存储相似度和标签
    sim = torch.empty((train_len, test_len), device=model_device, dtype=data_type)
    train_labels = torch.empty(train_len, device=model_device, dtype=torch.int16)
    test_labels = torch.empty(test_len, device=model_device, dtype=torch.int16)
    test_features = torch.empty((test_len, embedding_dim), device=model_device, dtype=data_type)

    # 处理测试数据
    for data_pair, i in zip(test_loader, tqdm(range(len(test_loader)))):
        # 直接调用模型的 encode 方法，因为模型已经被移动到单卡上
        features = model.encode(data_pair)

        # 获取批次长度
        batch_len_image = len(features)

        # 存储特征和标签
        test_features[batch_size * i:batch_size * i + batch_len_image] = features
        test_labels[batch_size * i:batch_size * i + batch_len_image] = data_pair["label"].to(model_device)

    # 处理训练数据
    for data_pair, i in zip(train_loader, tqdm(range(len(train_loader)))):
        torch.cuda.empty_cache()  # 清空缓存以减少显存占用
        gc.collect()  # 手动进行垃圾回收

        # 同样直接调用模型的 encode 方法
        features = model.encode(data_pair)
        batch_len_image = len(features)

        # 存储训练数据的标签和相似度计算结果
        train_labels[batch_size * i:batch_size * i + batch_len_image] = data_pair["label"].to(model_device)
        sim[batch_size * i:batch_size * i + batch_len_image, :] = calc_distance(
            features.to(data_type),
            test_features,
            dis_type=opt["model"]["dis_type"]
        )
    # 调用 validate_on_tensor 函数进行最终验证
    return validate_on_tensor(sim.T.cpu(), test_labels.cpu(), train_labels.cpu(),
                              metric_dict=opt["dataset"][opt["dataset"]["type"]]["metric"])
    # return validate_on_tensor(sim.T, test_labels, train_labels,
    #                           metric_dict=opt["dataset"][opt["dataset"]["type"]]["metric"])

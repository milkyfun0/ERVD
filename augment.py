import math
import random
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision import transforms


class GridMask(nn.Module):

    def __init__(self, max_epoch=300, d1=96, d2=224, rotate=360, ratio=0.6, mode=1, prob=0.1, crop_size=(224, 224)):
        """
        初始化 GridMask 类，定义网格掩码的参数。
        参数:
        d1 (int): 网格掩码间隔的最小值。掩码的方形网格大小会在 [d1, d2] 范围内随机选择。
        d2 (int): 网格掩码间隔的最大值。
        rotate (int): 允许的最大旋转角度（以度为单位）。每个掩码会在 0 到 rotate 之间随机旋转。
        ratio (float): 方格内空白的比例。用于控制网格的密度，范围是 [0, 1]。
        mode (int): 掩码模式。0 表示原始网格掩码，1 表示反转掩码（即将掩码区域变为非掩码区域，反之亦然）。
        prob (float): 掩码应用的概率，值为 0 到 1。当随机值大于此概率时，掩码将不被应用。
        crop_size (tuple): 输出图像的裁剪尺寸。
        参考文献：
        https://arxiv.org/abs/2001.04086
        """
        super(GridMask, self).__init__()
        self.device_flag = nn.Linear(1, 1)
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.max_epoch = max_epoch
        self.prob = prob
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @property
    def device(self):
        return self.device_flag.weight.data.device

    @property
    def dtype(self):
        return self.device_flag.weight.data.dtype

    def set_prob(self, epoch):
        """动态设置应用掩码的概率"""
        self.prob = self.st_prob * min(1, epoch / self.max_epoch)

    def forward(self, images, augment_images, epoch=None):
        target_device, target_dtype = augment_images.device, augment_images.dtype
        augment_images = augment_images.to(self.device).type(self.dtype)
        augment_images = self.transforms(augment_images)
        self.set_prob(epoch)
        n, c, h, w = augment_images.size()
        hh = math.ceil(math.sqrt(h * h + w * w))  # 计算掩码所需的最小方形尺寸
        d = torch.randint(self.d1, self.d2, (1,)).item()  # 随机生成网格大小
        l = math.ceil(d * self.ratio)  # 网格内的空白大小

        # 生成初始全为1的掩码
        mask = torch.ones((n, hh, hh), device=self.device)
        st_h = torch.randint(0, d, (1,)).item()  # 随机起始位置
        st_w = torch.randint(0, d, (1,)).item()

        # 创建垂直和水平的网格掩码
        for i in range(-1, hh // d + 1):
            s_h = d * i + st_h
            t_h = s_h + l
            s_h = max(min(s_h, hh), 0)
            t_h = max(min(t_h, hh), 0)
            mask[:, s_h:t_h, :] = 0

        for i in range(-1, hh // d + 1):
            s_w = d * i + st_w
            t_w = s_w + l
            s_w = max(min(s_w, hh), 0)
            t_w = max(min(t_w, hh), 0)
            mask[:, :, s_w:t_w] = 0

        if self.rotate > 0:
            angle = torch.randint(0, self.rotate, (n,)).float()  # 为每个样本生成一个旋转角度
            cos_vals = torch.cos(angle * math.pi / 180.0).unsqueeze(1)
            sin_vals = torch.sin(angle * math.pi / 180.0).unsqueeze(1)

            theta = torch.zeros((n, 2, 3), device=self.device)
            theta[:, 0, 0] = cos_vals.squeeze()
            theta[:, 0, 1] = -sin_vals.squeeze()
            theta[:, 1, 0] = sin_vals.squeeze()
            theta[:, 1, 1] = cos_vals.squeeze()

            grid = nn.functional.affine_grid(theta, size=(n, 1, hh, hh), align_corners=False)
            mask = nn.functional.grid_sample(mask.unsqueeze(1), grid, align_corners=False).squeeze(1)

        mask = mask[:, (hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.unsqueeze(1).expand_as(augment_images)  # [n, c, h, w] 扩展到输入的通道维度
        augment_images = augment_images * mask

        return augment_images.to(target_device).to(target_dtype)


class MixUp(nn.Module):
    def __init__(self, max_epoch=300, alpha=0.5):
        """
        MixUp 数据增强模块。
        该模块实现了 MixUp 方法，通过将不同图像按比例混合来增强模型的鲁棒性。
        参数：
            max_epoch (int): 最大训练轮数（目前未使用）。
            alpha (float): 控制混合比例的参数，越大表示混合程度越大。
        参考文献：
            "mixup: Beyond Empirical Risk Minimization"
            https://arxiv.org/abs/1710.09412
        """
        super(MixUp, self).__init__()
        self.alpha = alpha
        self.device_flag = nn.Linear(1, 1)
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @property
    def device(self):
        return self.device_flag.weight.data.device

    @property
    def dtype(self):
        return self.device_flag.weight.data.dtype

    def forward(self, images, augment_images, epoch=None):
        target_device, target_dtype = augment_images.device, augment_images.dtype
        augment_images = augment_images.to(self.device).type(self.dtype)
        images = images.to(self.device).type(self.dtype)
        augment_images = self.transform(augment_images)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        batch_size = augment_images.size(0)
        index = torch.randperm(batch_size).to(self.device)
        if lam < 0.5:
            lam = 1 - lam
        mixed_image = lam * augment_images + (1 - lam) * images[index, :]
        return mixed_image.to(target_device).to(target_dtype)


class PatchUpMode(Enum):
    SOFT = 'soft'
    HARD = 'hard'


class PatchUp(nn.Module):
    def __init__(self, block_size=7, gamma=0.9, patchup_type=PatchUpMode.SOFT):
        """
        PatchUp 模块，用于特征图增强。
        该模块对特征图应用软 PatchUp 或硬 PatchUp，以通过修改特定区域增强训练的鲁棒性。
        参数：
            block_size (int): 要改变的正方形块的大小，必须是奇数。
            gamma (float): [0, 1] 范围内的概率缩放因子，决定特征被修改的可能性。
                           更高的值会增加修改的可能性。
            patchup_type (PatchUpMode): 指定要应用的 PatchUp 操作类型。
                                         可以是软 PatchUp 或硬 PatchUp。
        参考文献：
            "PatchUp: A New Data Augmentation Method for Robustness in Neural Networks"
            https://arxiv.org/abs/2006.07794
        """
        super(PatchUp, self).__init__()
        self.device_flag = nn.Linear(1, 1)
        self.block_size = block_size
        self.gamma = gamma
        self.patchup_type = patchup_type
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @property
    def device(self):
        return self.device_flag.weight.data.device

    @property
    def dtype(self):
        return self.device_flag.weight.data.dtype

    def adjust_gamma(self, x):
        return self.gamma * x.shape[-1] ** 2 / \
            (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, images, augment_images, epoch=None):
        target_device, target_dtype = augment_images.device, augment_images.dtype
        augment_images = augment_images.to(self.device).type(self.dtype)
        images = images.to(self.device).type(self.dtype)
        augment_images = self.transform(augment_images)
        # images = self.transform(images)

        lam = torch.empty(1).uniform_(0, 1).item()  # 使用 PyTorch 生成随机数
        if lam < 0.5:
            lam = 1 - lam
        if self.gamma_adj is None:
            self.gamma_adj = self.adjust_gamma(augment_images)

        p = torch.ones_like(augment_images[0]) * self.gamma_adj
        m_i_j = torch.bernoulli(p)
        m_i_j = m_i_j.expand(augment_images.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))

        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)
        mask = 1 - holes
        unchanged = mask * augment_images
        indices = torch.randperm(images.size(0))  # 使用 PyTorch 生成随机排列
        if self.patchup_type == PatchUpMode.SOFT:
            patches = holes * images
            patches = holes * augment_images * lam + patches[indices] * (1 - lam)
        elif self.patchup_type == PatchUpMode.HARD:
            patches = holes * images
            patches = patches[indices]
        augment_images = unchanged + patches
        return augment_images.to(target_device).to(target_dtype)


class CutMix(nn.Module):
    def __init__(self, alpha=0.4):
        """
        CutMix 数据增强模块。
        该模块实现了 CutMix 方法，通过将不同图像的部分区域混合来增强模型的鲁棒性。
        参数：
            alpha (float): 控制随机裁剪区域大小的参数，越大表示裁剪区域越大。
        参考文献：
            "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
            https://arxiv.org/abs/1905.04899.pdf
        """
        super(CutMix, self).__init__()
        self.alpha = alpha
        self.device_flag = nn.Linear(1, 1)
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def device(self):
        return self.device_flag.weight.data.device

    @property
    def dtype(self):
        return self.device_flag.weight.data.dtype

    def rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix"""
        W = size[2]  # 图像宽度
        H = size[3]  # 图像高度

        # 计算剪裁区域的尺寸
        cut_rat = torch.sqrt(torch.tensor(1. - lam, device=self.device))  # 将 lam 转换为 Tensor，并计算平方根
        cut_w = (W * cut_rat).long()  # 剪裁宽度
        cut_h = (H * cut_rat).long()  # 剪裁高度

        # 随机生成中心点
        cx = torch.randint(0, W, (1,), device=self.device).item()
        cy = torch.randint(0, H, (1,), device=self.device).item()

        # 计算剪裁框的四个坐标，并确保不越界
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def forward(self, images, augment_images, epoch=None):
        target_device, target_dtype = augment_images.device, augment_images.dtype
        augment_images = augment_images.to(self.device).type(self.dtype)
        augment_images = self.transform(augment_images)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        if lam < 0.5:
            lam = 1 - lam
        rand_index = torch.randperm(augment_images.size(0)).to(self.device)  # 生成随机索引
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(augment_images.size(), lam)
        augment_images[:, :, bbx1:bbx2, bby1:bby2] = augment_images[rand_index, :, bbx1:bbx2, bby1:bby2]
        return augment_images.to(target_device).to(target_dtype)


class PatchMask(nn.Module):
    def __init__(self, max_epoch, image_size=224, patch_size=16, channels=3):
        """
         初始化 PatchMask 类，定义补丁掩码的参数。
         参数:
         max_epoch (int): 最大训练周期。
         image_size (int): 输入图像的大小（宽高相同）。
         patch_size (int): 每个补丁的大小。
         channels (int): 输入图像的通道数（例如，RGB图像为3）。
        """
        super(PatchMask, self).__init__()
        self.max_epoch = max_epoch
        self.alpha = None
        self.percent = None
        self.device_flag = nn.Linear(1, 1)
        self.img_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.patch_num = (self.img_size // self.patch_size) ** 2
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=[0.1, 2.0])], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.trans = transforms.Compose([
            transforms.Resize(224)
        ])

    @property
    def dtype(self):
        return self.device_flag.weight.data.dtype

    @property
    def device(self):
        return self.device_flag.weight.data.device

    def set_prob(self, epoch):
        if epoch is None:
            self.alpha = 0.3
            self.percent = 0.3
            return
        self.alpha = 0.5 * (1 - math.cos((0.5 * math.pi / self.max_epoch) * epoch))
        self.percent = 0.5 * (1 - math.cos((0.5 * math.pi / self.max_epoch) * epoch))

    def forward(self, images, augment_images, epoch=None):
        target_device, target_dtype = images.device, images.dtype
        self.set_prob(epoch)
        images = images.to(self.device).type(self.dtype)
        images_augment = self.transforms(augment_images)
        batch_size = images.size(0)
        mask_id = (torch.rand((batch_size, 1, self.patch_num), device=self.device) < self.percent).to(torch.int32)
        mask = torch.ones((batch_size, self.channels * self.patch_size * self.patch_size, 1),
                          device=self.device) * mask_id
        image_mask = F.fold(mask, kernel_size=self.patch_size, stride=self.patch_size, output_size=self.img_size)
        images_augment = torch.flip(images_augment, dims=[0])
        images_augment = images_augment + self.alpha * image_mask * (images - images_augment)
        images_augment = torch.flip(images_augment, dims=[0])
        return images_augment.to(target_device).to(target_dtype)


def load_image(image_paths):
    transform = transforms.Compose([transforms.ToTensor()])
    if isinstance(image_paths, list):
        # 如果是列表，则逐个处理每张图片
        images = []
        for path in image_paths:
            img = Image.open(path)  # 读取图片
            img = transform(img)  # 转换为张量
            images.append(img)
        return torch.stack(images)  # 返回张量列表
    else:
        # 如果是单个路径
        img = Image.open(image_paths)  # 读取图片
        return transform(img)


def show_image(image_tensor):
    """
    将张量形式的图片展示出来并沾满整个画布
    """

    def clip_image_data(image):
        """
        将输入图像裁剪到 imshow 所需的有效范围
        :param image: 输入图像（支持浮点数或整数）
        :return: 裁剪后的图像
        """
        if np.issubdtype(image.dtype, np.floating):  # 如果是浮点类型，范围 [0, 1]
            image = np.clip(image, 0.0, 1.0)
        elif np.issubdtype(image.dtype, np.integer):  # 如果是整数类型，范围 [0, 255]
            image = np.clip(image, 0, 255)
        return image

    image_np = image_tensor.permute(1, 2, 0).numpy()  # 将张量的通道维度转换为最后一个维度，并转为 numpy 数组
    image_np = clip_image_data(image_np)

    fig, ax = plt.subplots(figsize=(6, 6))  # 创建一个6x6的画布
    ax.imshow(image_np)
    ax.axis('off')  # 不显示坐标轴

    # 调整子图位置参数，确保图像沾满整个画布
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.show()


if __name__ == '__main__':
    # test_images = [f'dataset/AID/airport/airport_{i}.jpg' for i in range(1, 10)]
    test_images = [
        "dataset/NWPU_RESISC45/forest/forest_009.jpg",
        "dataset/NWPU_RESISC45/wetland/wetland_001.jpg",
        "dataset/NWPU_RESISC45/basketball_court/basketball_court_002.jpg",
        "dataset/NWPU_RESISC45/bridge/bridge_008.jpg",
        "dataset/NWPU_RESISC45/airplane/airplane_002.jpg",
        "dataset/NWPU_RESISC45/overpass/overpass_001.jpg",
        "dataset/NWPU_RESISC45/church/church_003.jpg",
    ]
    # 创建 GridMask 实例
    grid_mask = PatchMask(300)
    x = load_image(test_images)
    # 前向传播
    # show_image(x[-1])
    output = grid_mask(x, x, 200)
    for i in range(len(test_images)):
        show_image(x[i])
        show_image(output[i].cpu())

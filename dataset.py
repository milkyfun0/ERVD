#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 下午4:54
# @Author  : CaoQixuan
# @File    : dataset.py
# @Description :
import math

import numpy
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from augment import GridMask, MixUp, CutMix, PatchMask, PatchUp, RandomErasing

base_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

augment_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    # transforms.RandomApply(
    #     [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
    # ),
    # transforms.RandomGrayscale(p=0.2),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=[0.1, 2.0])], p=0.8),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
])


class BaseDataset(data.Dataset):
    def __init__(self, path, flag, transform):
        super(BaseDataset, self).__init__()
        self.flag = flag
        self.path = path
        self.images = open(self.path + "%s.txt" % self.flag).readlines()
        self.length = len(self.images)
        self.transform = transform

    def __getitem__(self, index):
        image_path, class_list = eval(self.images[index])
        class_id = numpy.argmax(numpy.array(class_list))

        img_pil = Image.open(self.path + image_path).convert('RGB')
        img = self.transform(img_pil)

        return {
            "image": img,
            "label": class_id,
            "image_pil": img_pil,
        }

    def __len__(self):
        return self.length


class NWEP_RESISC45(BaseDataset):
    def __init__(self, opt, flag):
        self.transform = base_transform

        super(NWEP_RESISC45, self).__init__(opt["path"], flag, self.transform)

    def __getitem__(self, index):
        data_pair = super(NWEP_RESISC45, self).__getitem__(index)
        data_pair.pop('image_pil')
        return data_pair


class NWEP_RESISC45_Aug(BaseDataset):
    def __init__(self, opt, flag):
        super(NWEP_RESISC45_Aug, self).__init__(opt["path"], flag, base_transform)
        self.augment_transform = augment_transform

    def __getitem__(self, index):
        data_pair = super(NWEP_RESISC45_Aug, self).__getitem__(index)
        image, label, image_pil = data_pair["image"], data_pair["image"], data_pair["image_pil"]
        augment_image = self.augment_transform(image_pil)
        data_pair.pop('image_pil')
        data_pair["augment_image"] = augment_image
        return data_pair


class AID(BaseDataset):
    def __init__(self, opt, flag):
        self.transform = base_transform
        super(AID, self).__init__(opt["path"], flag, self.transform)

    def __getitem__(self, index):
        data_pair = super(AID, self).__getitem__(index)
        data_pair.pop('image_pil')
        return data_pair


class AID_Aug(BaseDataset):
    def __init__(self, opt, flag):
        super(AID_Aug, self).__init__(opt["path"], flag, base_transform)
        self.augment_transform = augment_transform

    def __getitem__(self, index):
        data_pair = super(AID_Aug, self).__getitem__(index)
        image, label, image_pil = data_pair["image"], data_pair["image"], data_pair["image_pil"]
        augment_image = self.augment_transform(image_pil)
        data_pair.pop('image_pil')
        data_pair["augment_image"] = augment_image
        return data_pair


class UCMD(BaseDataset):
    def __init__(self, opt, flag):
        self.transform = base_transform
        super(UCMD, self).__init__(opt["path"], flag, self.transform)

    def __getitem__(self, index):
        data_pair = super(UCMD, self).__getitem__(index)
        data_pair.pop('image_pil')
        return data_pair


class UCMD_Aug(BaseDataset):
    def __init__(self, opt, flag):
        self.transform = base_transform
        super(UCMD_Aug, self).__init__(opt["path"], flag, self.transform)
        self.augment_transform = augment_transform

    def __getitem__(self, index):
        data_pair = super(UCMD_Aug, self).__getitem__(index)
        image, label, image_pil = data_pair["image"], data_pair["image"], data_pair["image_pil"]
        augment_image = self.augment_transform(image_pil)
        data_pair.pop('image_pil')
        data_pair["augment_image"] = augment_image
        return data_pair


class Augment(nn.Module):
    def __init__(self, opt, image_size=224, patch_size=16):
        super(Augment, self).__init__()
        max_epoch = opt["train"]["epoch"]
        if opt["dataset"]["aug_type"] == "gridmask":
            self.augment = GridMask(max_epoch=max_epoch)
        elif opt["dataset"]["aug_type"] == "mixup":
            self.augment = MixUp()
        elif opt["dataset"]["aug_type"] == "cutmix":
            self.augment = CutMix()
        elif opt["dataset"]["aug_type"] == "patchup":
            self.augment = PatchUp()
        elif opt["dataset"]["aug_type"] == "patchmask":
            self.augment = PatchMask(max_epoch=max_epoch, image_size=image_size, patch_size=patch_size)
        else:
            self.augment = None

    @property
    def dtype(self):
        return self.augment.dtype

    @property
    def device(self):
        return self.augment.device

    def forward(self, data_pair, epoch=None):
        images, labels = data_pair["image"].to(self.augment.device), data_pair["label"].to(self.augment.device)
        if not self.training:
            return images, labels
        if "augment_image" not in data_pair:
            return images, labels

        augment_images = data_pair["augment_image"].to(self.device)
        images_augment = self.augment(images, augment_images, epoch=epoch)
        images = torch.cat((images, images_augment), dim=0)
        labels = torch.cat((labels, labels), dim=0)

        return images, labels


def get_loader(opt):
    modules = __import__("dataset")
    config = opt["dataset"][opt["dataset"]["type"]]
    config["name"] = config["name"] if not opt["dataset"]["augment"] else config["name"] + "_Aug"
    train_loader = torch.utils.data.DataLoader(
        dataset=getattr(modules, config["name"])(
            opt=config,
            flag="train"
        ),
        batch_size=opt["train"]["batch_size"],
        num_workers=opt["train"]["num_works"],
        shuffle=True,
        # pin_memory=True,
        drop_last=True,
        # prefetch_factor=2

    )
    test_loader = torch.utils.data.DataLoader(
        dataset=getattr(modules, config["name"])(
            opt=config,
            flag="test"
        ),
        batch_size=opt["train"]["batch_size"] * 2,
        num_workers=opt["train"]["num_works"],
        shuffle=False,
        # pin_memory=True
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset=getattr(modules, config["name"])(
            opt=config,
            flag="dataset"
        ),
        batch_size=opt["train"]["batch_size"] * 2,
        num_workers=opt["train"]["num_works"],
        shuffle=False,
        # pin_memory=True,
        # prefetch_factor=2
    )
    return train_loader, test_loader, dataset_loader


if __name__ == '__main__':
    data = open("dataset/UCMD/database.txt").read().splitlines()
    imgs = [(val.split()[0], numpy.array([int(la) for la in val.split()[1:]])) for val in data]
    with open("dataset/UCMD/database.txt", "w") as f:
        for img in imgs:
            f.write("\"%s\", %s\n" % (img[0], img[1].tolist()))

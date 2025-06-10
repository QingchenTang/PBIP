import os
import os.path

import torch.distributed as dist

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.bcss import BCSSTestDataset, BCSSTrainingDataset, BCSSWSSSDataset


def get_wsss_dataset(cfg):
    MEAN, STD = get_mean_std(cfg.dataset.name)

    transform = {
        "train": A.Compose([
            A.Normalize(MEAN, STD),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True),
        ]),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ])
    }
    train_dataset = BCSSWSSSDataset(cfg.dataset.train_root, mask_name=cfg.dataset.mask_root,transform=transform["train"])
    val_dataset = BCSSTestDataset(cfg.dataset.val_root, split="valid", transform=transform["val"])

    return train_dataset, val_dataset


def get_cls_dataset(cfg, split="valid", p=0.5, enable_rotation=True):
    MEAN, STD = get_mean_std(cfg.dataset.name)
    
    # 构建训练时的变换列表
    train_transforms = [
        A.Normalize(MEAN, STD),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
    ]
    
    # 根据参数决定是否添加旋转
    if enable_rotation:
        train_transforms.append(A.RandomRotate90())
    
    train_transforms.append(ToTensorV2(transpose_mask=True))
    
    transform = {
        "train": A.Compose(train_transforms),
        "val": A.Compose([
            A.Normalize(MEAN, STD),
            ToTensorV2(transpose_mask=True),
        ]),
    }


    train_dataset = BCSSTrainingDataset(cfg.dataset.train_root, transform=transform["train"])
    val_dataset = BCSSTestDataset(cfg.dataset.val_root, split, transform=transform["val"])

    return train_dataset, val_dataset


def get_mean_std(dataset):
    norm = [[0.66791496, 0.47791372, 0.70623304], [0.1736589,  0.22564577, 0.19820057]]
    return norm[0], norm[1]


def all_reduced(x, n_gpus):
    dist.all_reduce(x)
    x /= n_gpus


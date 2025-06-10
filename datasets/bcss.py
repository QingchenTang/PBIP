import numpy as np
from numpy.lib.utils import deprecate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import re


class BCSSTrainingDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/train", transform=None):
        super(BCSSTrainingDataset, self).__init__()
        self.get_images_and_labels(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cls_label = self.cls_labels[index]
        assert os.path.exists(img_path), "img_path: {} does not exists".format(img_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return os.path.basename(img_path), img, cls_label, 0

    def get_images_and_labels(self, img_root=None):
        self.img_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "*.png"))
        for img_path in self.img_paths:
            term_split = re.split("\[|\]", img_path)
            cls_label = np.array([int(x) for x in term_split[1]])
            self.cls_labels.append(cls_label)


class BCSSTestDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/", split="test", transform=None):
        assert split in ["test", "valid"], "split must be one of [test, valid]"
        super(BCSSTestDataset, self).__init__()
        self.get_images_and_labels(img_root, split)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        assert os.path.exists(img_path), "img_path: {} does not exists".format(img_path)
        assert os.path.exists(mask_path), "mask_path: {} does not exists".format(mask_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask = np.array(Image.open(mask_path))
        cls_label = np.array([0, 0, 0, 0])
        x = np.unique(mask) if np.unique(mask)[-1] != 4 else np.unique(mask)[:-1]
        cls_label[x] = 1

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self, img_root=None, split=None):
        self.img_paths = []
        self.mask_paths = []

        self.mask_paths = glob.glob(os.path.join(img_root, split, "mask", "*.png"))

        for mask_path in self.mask_paths:
            img_name = os.path.basename(mask_path)
            self.img_paths.append(os.path.join(img_root, split, "img", img_name))


class BCSSWSSSDataset(Dataset):
    CLASSES = ["TUM", "STR", "LYM", "NEC", "BACK"]
    def __init__(self, img_root="../data/BCSS-WSSS/train", mask_name="pseudo_label", transform=None):
        super(BCSSWSSSDataset, self).__init__()
        self.get_images_and_labels(img_root, mask_name)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        cls_label = self.cls_labels[index]
        assert os.path.exists(img_path), "image path: {}, does not exist".format(img_path)
        assert os.path.exists(mask_path), "mask path: {}, does not exist".format(mask_path)

        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        mask = np.array(Image.open(mask_path))
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return os.path.basename(img_path), img, cls_label, mask

    def get_images_and_labels(self, img_root, mask_name):
        self.img_paths = []
        self.mask_paths = []
        self.cls_labels = []

        self.img_paths = glob.glob(os.path.join(img_root, "img", "*.png"))

        for img_path in self.img_paths:
            img_name = os.path.basename(img_path)
            self.mask_paths.append(os.path.join(img_root, mask_name, img_name))
            term_split = re.split("\[|\]", img_path)
            cls_label = np.array([int(x) for x in term_split[1]])
            self.cls_labels.append(cls_label)

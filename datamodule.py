import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2

from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Subset

from dataset import (
    XrayFindingDataset,
    XrayDetectionDataset,
    XrayTestDataset,
    XrayDetectionNmsDataset,
    XrayDetectionWbfDataset,
)


class XrayFindingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir="dataset-jpg",
        fold_splits=10,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size

    def setup(self, stage=None):
        self.train_dataset = XrayFindingDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_dataset = XrayFindingDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_df = self.train_dataset.train_df
        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return valid_loader

    def get_train_transform(self):
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=0.5),
                        A.RGBShift(p=0.5),
                        A.HueSaturationValue(p=0.5),
                        A.ToGray(p=0.5),
                        A.ChannelDropout(p=0.5),
                        A.ChannelShuffle(p=0.5),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Blur(p=0.5),
                        A.GaussNoise(p=0.5),
                        A.IAASharpen(p=0.5),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        # A.Rotate(limit=20, p=0.5),
                        A.HorizontalFlip(p=0.5),
                        # A.VerticalFlip(p=0.5),
                    ],
                    p=0.5,
                ),
                A.RandomResizedCrop(
                    height=self.resize_height,
                    width=self.resize_width,
                    scale=(0.1, 1.0),
                    p=1.0,
                ),
                # A.Resize(height=self.resize_height, width=self.resize_width),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def get_valid_transform(self):
        return A.Compose(
            [
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def make_fold_index(self, n_splits=10, fold_index=0):
        print(f"Fold splits: {n_splits}")
        print(f"Fold index: {fold_index}")
        skf = StratifiedKFold(n_splits=n_splits)
        train_fold = []
        valid_fold = []
        for train_index, valid_index in skf.split(
            self.train_df.image_id, self.train_df.label
        ):
            train_fold.append(train_index)
            valid_fold.append(valid_index)

        return train_fold[fold_index], valid_fold[fold_index]


class XrayDetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir="dataset-jpg",
        fold_splits=10,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size

    def setup(self, stage=None):
        self.train_dataset = XrayDetectionDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_dataset = XrayDetectionDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.image_ids = self.train_dataset.image_ids
        self.most_class_ids = self.train_dataset.most_class_ids
        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.make_batch,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.make_batch,
        )
        return valid_loader

    def get_train_transform(self):
        return A.Compose(
            [
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
                A.PadIfNeeded(
                    min_height=self.resize_height,
                    min_width=self.resize_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0,
                ),
                A.RandomCrop(height=self.resize_height, width=self.resize_width, p=1.0),
                A.RandomBrightnessContrast(p=0.8),
                A.ChannelDropout(p=0.5),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.5),
                        A.MedianBlur(p=0.5),
                        A.GaussianBlur(p=0.5),
                        A.GaussNoise(p=0.5),
                    ],
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0,
                min_visibility=0,
                label_fields=["labels"],
            ),
        )

    def get_valid_transform(self):
        return A.Compose(
            [
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0,
                min_visibility=0,
                label_fields=["labels"],
            ),
        )

    def make_fold_index(self, n_splits=10, fold_index=0):
        print(f"Fold splits: {n_splits}")
        print(f"Fold index: {fold_index}")
        skf = StratifiedKFold(n_splits=n_splits)
        train_fold = []
        valid_fold = []
        for train_index, valid_index in skf.split(self.image_ids, self.most_class_ids):
            train_fold.append(train_index)
            valid_fold.append(valid_index)

        return train_fold[fold_index], valid_fold[fold_index]

    def make_batch(self, samples):
        image = torch.stack([sample["image"] for sample in samples])
        bboxes = [sample["bboxes"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        image_id = [sample["image_id"] for sample in samples]

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "image_id": image_id,
        }


class XrayDetectionNmsDataModule(XrayDetectionDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionNmsDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_dataset = XrayDetectionNmsDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.image_ids = self.train_dataset.image_ids
        self.most_class_ids = self.train_dataset.most_class_ids
        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class XrayDetectionWbfDataModule(XrayDetectionDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionWbfDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_dataset = XrayDetectionWbfDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.image_ids = self.train_dataset.image_ids
        self.most_class_ids = self.train_dataset.most_class_ids
        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class XrayTestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir="dataset-jpg",
        image_ids=None,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size

    def setup(self, stage=None):
        self.test_dataset = XrayTestDataset(
            dataset_dir=self.dataset_dir,
            image_ids=self.image_ids,
            transform=self.get_test_transform(),
        )

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader

    def get_test_transform(self):
        return A.Compose(
            [
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
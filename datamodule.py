import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from dataset import (
    XrayFindingDataset,
    NIHFindingDataset,
    XrayDetectionDataset,
    XrayTestDataset,
    XrayTestEnsembleDataset,
    XrayDetectionNmsDataset,
    XrayDetectionNmsDataset_V2,
    XrayDetectionWbfDataset,
    XrayDetectionAllDataset,
    XrayDetectionAllNmsDataset,
    XrayDetectionAllNmsDataset_V2,
    XrayDetectionAllWbfDataset,
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


class XrayFindingConcatDataModule_V1(XrayFindingDataModule):
    def __init__(
        self,
        dataset_dir="dataset-jpg",
        dataset_nih_dir="dataset-nih",
        fold_splits=10,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            fold_splits=fold_splits,
            fold_index=fold_index,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )
        self.dataset_nih_dir = dataset_nih_dir
        print("Train on VBD & NIH Chest X-Rays Concat Dataset")

    def setup(self, stage=None):
        self.train_vbd_dataset = XrayFindingDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.train_nih_dataset = NIHFindingDataset(
            self.dataset_nih_dir, transform=self.get_train_transform()
        )

        self.valid_vbd_dataset = XrayFindingDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )
        self.valid_nih_dataset = NIHFindingDataset(
            self.dataset_nih_dir, transform=self.get_valid_transform()
        )

        self.train_dataset = ConcatDataset(
            [self.train_vbd_dataset, self.train_nih_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_vbd_dataset, self.valid_nih_dataset]
        )

        self.train_df = pd.concat(
            [self.train_vbd_dataset.train_df, self.train_nih_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class XrayFindingConcatDataModule(XrayFindingDataModule):
    def __init__(
        self,
        dataset_dir="dataset-jpg",
        dataset_nih_dir="dataset-nih",
        fold_splits=10,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            fold_splits=fold_splits,
            fold_index=fold_index,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
        )
        self.dataset_nih_dir = dataset_nih_dir
        print("Train on VBD & NIH Chest X-Rays Concat Dataset")

    def setup(self, stage=None):
        self.train_vbd_dataset = XrayFindingDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_vbd_dataset = XrayFindingDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )
        self.train_df = self.train_vbd_dataset.train_df
        self.train_vbd_index, self.valid_vbd_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_vbd_dataset = Subset(self.train_vbd_dataset, self.train_vbd_index)
        self.valid_vbd_dataset = Subset(self.valid_vbd_dataset, self.valid_vbd_index)

        self.train_nih_dataset = NIHFindingDataset(
            self.dataset_nih_dir, transform=self.get_train_transform()
        )

        self.train_dataset = ConcatDataset(
            [self.train_vbd_dataset, self.train_nih_dataset]
        )
        self.valid_dataset = self.valid_vbd_dataset


class XrayDetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir="dataset-jpg",
        fold_splits=10,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
        valid_filter=False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size
        self.valid_filter = valid_filter

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

        if self.valid_filter:
            print("Apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionNmsDataset(
                self.dataset_dir, transform=self.get_valid_transform()
            )
        else:
            print("Not apply bbox filter on valid_dataset")
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


class XrayDetectionNmsDataModule_V2(XrayDetectionDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionNmsDataset_V2(
            self.dataset_dir, transform=self.get_train_transform()
        )

        if self.valid_filter:
            print("Apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionNmsDataset_V2(
                self.dataset_dir, transform=self.get_valid_transform()
            )
        else:
            print("Not apply bbox filter on valid_dataset")
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


class XrayDetectionWbfDataModule(XrayDetectionDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionWbfDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )

        if self.valid_filter:
            print("Apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionWbfDataset(
                self.dataset_dir, transform=self.get_valid_transform()
            )
        else:
            print("Not apply bbox filter on valid_dataset")
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


class XrayDetectionAllDataModule(XrayDetectionDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionAllDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_dataset = XrayDetectionAllDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.image_ids = self.train_dataset.image_ids
        self.most_class_ids = self.train_dataset.most_class_ids
        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class XrayDetectionAllNmsDataModule(XrayDetectionAllDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionAllNmsDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )

        if self.valid_filter:
            print("Apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionAllNmsDataset(
                self.dataset_dir, transform=self.get_valid_transform()
            )
        else:
            print("Not apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionAllDataset(
                self.dataset_dir, transform=self.get_valid_transform()
            )

        self.image_ids = self.train_dataset.image_ids
        self.most_class_ids = self.train_dataset.most_class_ids
        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class XrayDetectionAllNmsDataModule_V2(XrayDetectionAllDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionAllNmsDataset_V2(
            self.dataset_dir, transform=self.get_train_transform()
        )

        if self.valid_filter:
            print("Apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionAllNmsDataset_V2(
                self.dataset_dir, transform=self.get_valid_transform()
            )
        else:
            print("Not apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionAllDataset(
                self.dataset_dir, transform=self.get_valid_transform()
            )

        self.image_ids = self.train_dataset.image_ids
        self.most_class_ids = self.train_dataset.most_class_ids
        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class XrayDetectionAllWbfDataModule(XrayDetectionAllDataModule):
    def setup(self, stage=None):
        self.train_dataset = XrayDetectionAllWbfDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )

        if self.valid_filter:
            print("Apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionAllWbfDataset(
                self.dataset_dir, transform=self.get_valid_transform()
            )
        else:
            print("Not apply bbox filter on valid_dataset")
            self.valid_dataset = XrayDetectionAllDataset(
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


class XrayTestEnsembleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir="dataset-jpg",
        image_ids=None,
        batch_size=32,
        num_workers=2,
        image_size_list=[1024],
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size_list = image_size_list

    def setup(self, stage=None):
        self.test_dataset = XrayTestEnsembleDataset(
            dataset_dir=self.dataset_dir,
            image_ids=self.image_ids,
            image_size_list=self.image_size_list,
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
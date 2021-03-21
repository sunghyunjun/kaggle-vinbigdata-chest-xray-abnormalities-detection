import os
from typing import Tuple, Dict, Union

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.ops.boxes import batched_nms

from tqdm import tqdm

from ensemble_boxes import *

from prepare_data import read_xray


class XrayFindingDataset(Dataset):
    def __init__(self, dataset_dir="dataset-jpg", transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.csv_path = os.path.join(self.dataset_dir, "train.csv")
        self.load_train_csv()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id, label = self.train_df.iloc[index, :][["image_id", "label"]]
        image_path = os.path.join(self.dataset_dir, "train", image_id + ".jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = A.Compose([A.Normalize(), ToTensorV2()])(image=image)["image"]

        label = torch.as_tensor(label)

        return image, label

    def __len__(self) -> int:
        return len(self.train_df)

    def load_train_csv(self):
        self.train_df = pd.read_csv(self.csv_path, usecols=["image_id", "class_name"])

        # Label 0: No finding, Label 1: Finding
        self.train_df["label"] = np.where(
            self.train_df["class_name"] == "No finding", 0, 1
        )
        del self.train_df["class_name"]
        self.train_df.drop_duplicates(inplace=True)


class NIHFindingDataset(Dataset):
    def __init__(self, dataset_dir="dataset-nih", transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.csv_path = os.path.join(self.dataset_dir, "Data_Entry_2017_jpg.csv")
        self.load_train_csv()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id, label = self.train_df.iloc[index, :][["image_id", "label"]]
        image_path = os.path.join(self.dataset_dir, "images", image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = A.Compose([A.Normalize(), ToTensorV2()])(image=image)["image"]

        label = torch.as_tensor(label)

        return image, label

    def __len__(self) -> int:
        return len(self.train_df)

    def load_train_csv(self):
        self.train_df = pd.read_csv(
            self.csv_path, usecols=["Image Index", "Finding Labels"]
        )
        self.train_df.rename(columns={"Image Index": "image_id"}, inplace=True)

        # Label 0: No finding, Label 1: Finding
        self.train_df["label"] = np.where(
            self.train_df["Finding Labels"] == "No Finding", 0, 1
        )
        del self.train_df["Finding Labels"]
        # self.train_df.drop_duplicates(inplace=True)


class XrayDetectionDataset(Dataset):
    def __init__(self, dataset_dir="dataset-jpg", transform=None, bboxes_yxyx=True):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.csv_path = os.path.join(self.dataset_dir, "train.csv")
        self.bboxes_yxyx = bboxes_yxyx
        self.load_train_csv()

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        image, bboxes, labels = self.load_image_boxes_labels(index)
        image_id = self.image_ids[index]

        data = {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "image_id": image_id,
        }

        if self.transform:
            sample = self.transform(
                image=data["image"],
                bboxes=data["bboxes"],
                labels=data["labels"],
                image_id=data["image_id"],
            )

            while len(sample["bboxes"]) < 1:
                # print("re-transform sample")
                sample = self.transform(
                    image=data["image"],
                    bboxes=data["bboxes"],
                    labels=data["labels"],
                    image_id=data["image_id"],
                )
            sample["bboxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            sample["labels"] = torch.as_tensor(sample["labels"], dtype=torch.int64)
        else:
            sample = A.Compose([A.Normalize(), ToTensorV2()])(
                image=data["image"],
                bboxes=data["bboxes"],
                labels=data["labels"],
                image_id=data["image_id"],
            )

            sample["bboxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            sample["labels"] = torch.as_tensor(sample["labels"], dtype=torch.int64)

        if self.bboxes_yxyx:
            # yxyx: for efficientdet training
            sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]

        return sample

    def __len__(self) -> int:
        return len(self.image_ids)

    def load_image_boxes_labels(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_id = self.image_ids[index]

        image_path = os.path.join(self.dataset_dir, "train", image_id + ".jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        records = self.train_data[image_id]

        # xyxy: x_min, y_min, x_max, y_max
        bboxes = records[:, 1:]

        # change class label 0-index 0~13 to non-zero, 1-index 1~14
        labels = records[:, 0] + 1

        return image, bboxes, labels

    def load_train_csv(self):
        self.preprocess_df()
        self.filter_extreme_bbox()

        self.image_ids = self.train_df.image_id.unique()

        self.train_data = {}
        max_index = len(self.image_ids)
        pbar = tqdm(range(max_index))

        for index in pbar:
            pbar.set_description("Processing train_labels")
            image_id = self.image_ids[index]
            records = self.train_df.loc[self.train_df.image_id == image_id].copy()

            self.train_data[image_id] = records[
                ["class_id", "x_min", "y_min", "x_max", "y_max"]
            ].values

        self.most_class_ids = []
        for image_id in self.image_ids:
            class_ids = self.train_data[image_id][:, 0].astype(np.int64)
            class_ids_counts = np.bincount(class_ids)
            self.most_class_ids.append(np.argmax(class_ids_counts))

    def preprocess_df(self):
        """Load csv data and preprocess."""
        self.train_df = pd.read_csv(
            self.csv_path,
            usecols=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
        )

        self.train_df.drop(
            self.train_df[self.train_df.class_id == 14].index, inplace=True
        )
        self.train_df.reset_index(drop=True, inplace=True)
        self.train_df = self.train_df.astype(
            {
                "class_id": "float32",
                "x_min": "float32",
                "y_min": "float32",
                "x_max": "float32",
                "y_max": "float32",
            }
        )

    def filter_extreme_bbox(self):
        """Filter out extreme large bbox data."""
        self.train_df["bbox_area"] = (self.train_df.x_max - self.train_df.x_min) * (
            self.train_df.y_max - self.train_df.y_min
        )
        self.train_df = self.train_df[self.train_df.bbox_area < 500_000]


class XrayDetectionNmsDataset(XrayDetectionDataset):
    """Dataset with duplicate bbox removed using torchvision's batched_nms"""

    def load_train_csv(self):
        self.preprocess_df()
        self.filter_extreme_bbox()

        self.image_ids = self.train_df.image_id.unique()

        # bbox NMS
        df_nms = pd.DataFrame(
            columns=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
            dtype=np.float32,
        )
        pbar = tqdm(self.image_ids)
        for image_id in pbar:
            pbar.set_description("Processing bbox nms")
            boxes, class_ids = self.get_bbox_nms(self.train_df, image_id)
            for box, class_id in zip(boxes, class_ids):
                df_nms = df_nms.append(
                    [
                        {
                            "image_id": image_id,
                            "class_id": class_id.numpy(),
                            "x_min": box[0].numpy(),
                            "y_min": box[1].numpy(),
                            "x_max": box[2].numpy(),
                            "y_max": box[3].numpy(),
                        }
                    ],
                    ignore_index=True,
                )
                df_nms[["class_id", "x_min", "y_min", "x_max", "y_max"]] = df_nms[
                    ["class_id", "x_min", "y_min", "x_max", "y_max"]
                ].astype(np.float32)

        self.train_df = df_nms

        self.train_data = {}
        max_index = len(self.image_ids)
        pbar = tqdm(range(max_index))

        for index in pbar:
            pbar.set_description("Processing train_labels")
            image_id = self.image_ids[index]
            records = self.train_df.loc[self.train_df.image_id == image_id].copy()

            self.train_data[image_id] = records[
                ["class_id", "x_min", "y_min", "x_max", "y_max"]
            ].values

        self.most_class_ids = []
        for image_id in self.image_ids:
            class_ids = self.train_data[image_id][:, 0].astype(np.int64)
            class_ids_counts = np.bincount(class_ids)
            self.most_class_ids.append(np.argmax(class_ids_counts))

    def get_bbox_nms(self, df, image_id, iou_threshold=0.4):
        boxes = torch.as_tensor(
            df[["x_min", "y_min", "x_max", "y_max"]][
                df.image_id == image_id
            ].to_numpy(),
            dtype=torch.float32,
        )
        idx = torch.as_tensor(df["class_id"][df.image_id == image_id].to_numpy())
        scores = torch.ones(idx.size(), dtype=torch.float32)
        keep = batched_nms(boxes, scores, idx, iou_threshold=iou_threshold)
        return boxes[keep], idx[keep]


class XrayDetectionNmsDataset_V2(XrayDetectionDataset):
    def load_train_csv(self):
        self.preprocess_df()
        self.filter_extreme_bbox()

        self.image_ids = self.train_df.image_id.unique()

        # bbox NMS
        df_nms = pd.DataFrame(
            columns=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
            dtype=np.float32,
        )
        pbar = tqdm(self.image_ids)
        for image_id in pbar:
            pbar.set_description("Processing bbox nms_v2")
            boxes, class_ids = self.get_bbox_nms(self.train_df, image_id)
            for box, class_id in zip(boxes, class_ids):
                df_nms = df_nms.append(
                    [
                        {
                            "image_id": image_id,
                            "class_id": class_id,
                            "x_min": box[0],
                            "y_min": box[1],
                            "x_max": box[2],
                            "y_max": box[3],
                        }
                    ],
                    ignore_index=True,
                )
                df_nms[["class_id", "x_min", "y_min", "x_max", "y_max"]] = df_nms[
                    ["class_id", "x_min", "y_min", "x_max", "y_max"]
                ].astype(np.float32)

        self.train_df = df_nms

        self.train_data = {}
        max_index = len(self.image_ids)
        pbar = tqdm(range(max_index))

        for index in pbar:
            pbar.set_description("Processing train_labels")
            image_id = self.image_ids[index]
            records = self.train_df.loc[self.train_df.image_id == image_id].copy()

            self.train_data[image_id] = records[
                ["class_id", "x_min", "y_min", "x_max", "y_max"]
            ].values

        self.most_class_ids = []
        for image_id in self.image_ids:
            class_ids = self.train_data[image_id][:, 0].astype(np.int64)
            class_ids_counts = np.bincount(class_ids)
            self.most_class_ids.append(np.argmax(class_ids_counts))

    def get_bbox_nms(self, df, image_id, iou_threshold=0.4, skip_box_thr=0.0):
        boxes = df[["x_min", "y_min", "x_max", "y_max"]][
            df.image_id == image_id
        ].to_numpy()
        idx = df["class_id"][df.image_id == image_id].to_numpy()
        scores = np.ones(len(idx), dtype=np.float32)

        boxes_max = np.max(boxes)
        boxes_normalized = boxes / boxes_max

        boxes_nms, _, labels_nms = non_maximum_weighted(
            [boxes_normalized],
            [scores],
            [idx],
            weights=None,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_thr,
        )
        boxes_nms *= boxes_max
        return boxes_nms, labels_nms


class XrayDetectionWbfDataset(XrayDetectionDataset):
    def load_train_csv(self):
        self.preprocess_df()
        self.filter_extreme_bbox()

        self.image_ids = self.train_df.image_id.unique()

        # bbox WBF
        df_wbf = pd.DataFrame(
            columns=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
            dtype=np.float32,
        )
        pbar = tqdm(self.image_ids)
        for image_id in pbar:
            pbar.set_description("Processing bbox wbf")
            boxes, class_ids = self.get_bbox_wbf(self.train_df, image_id)
            for box, class_id in zip(boxes, class_ids):
                df_wbf = df_wbf.append(
                    [
                        {
                            "image_id": image_id,
                            "class_id": class_id,
                            "x_min": box[0],
                            "y_min": box[1],
                            "x_max": box[2],
                            "y_max": box[3],
                        }
                    ],
                    ignore_index=True,
                )
                df_wbf[["class_id", "x_min", "y_min", "x_max", "y_max"]] = df_wbf[
                    ["class_id", "x_min", "y_min", "x_max", "y_max"]
                ].astype(np.float32)

        self.train_df = df_wbf

        self.train_data = {}
        max_index = len(self.image_ids)
        pbar = tqdm(range(max_index))

        for index in pbar:
            pbar.set_description("Processing train_labels")
            image_id = self.image_ids[index]
            records = self.train_df.loc[self.train_df.image_id == image_id].copy()

            self.train_data[image_id] = records[
                ["class_id", "x_min", "y_min", "x_max", "y_max"]
            ].values

        self.most_class_ids = []
        for image_id in self.image_ids:
            class_ids = self.train_data[image_id][:, 0].astype(np.int64)
            class_ids_counts = np.bincount(class_ids)
            self.most_class_ids.append(np.argmax(class_ids_counts))

    def get_bbox_wbf(self, df, image_id, iou_threshold=0.6, skip_box_thr=0.0):
        boxes = df[["x_min", "y_min", "x_max", "y_max"]][
            df.image_id == image_id
        ].to_numpy()
        idx = df["class_id"][df.image_id == image_id].to_numpy()
        scores = np.ones(len(idx), dtype=np.float32)

        boxes_max = np.max(boxes)
        boxes_normalized = boxes / boxes_max

        boxes_wbf, _, labels_wbf = weighted_boxes_fusion(
            [boxes_normalized],
            [scores],
            [idx],
            weights=None,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_thr,
        )
        boxes_wbf *= boxes_max
        return boxes_wbf, labels_wbf


class XrayDetectionAllDataset(XrayDetectionDataset):
    def preprocess_df(self):
        """Load csv data and preprocess."""
        self.train_df = pd.read_csv(
            self.csv_path,
            usecols=[
                "image_id",
                "class_id",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "raw_width",
                "raw_height",
            ],
        )
        self.train_df.drop_duplicates(inplace=True)
        self.train_df["x_min"].fillna(0, inplace=True)
        self.train_df["y_min"].fillna(0, inplace=True)

        # 1000 to prevent rounding error of albumentations
        self.train_df["x_max"] = self.train_df.apply(
            lambda x: 1000 if x["class_id"] == 14 else x["x_max"],
            axis=1,
        )
        self.train_df["y_max"] = self.train_df.apply(
            lambda x: 1000 if x["class_id"] == 14 else x["y_max"],
            axis=1,
        )

        self.train_df.drop(["raw_width", "raw_height"], axis=1, inplace=True)

        self.train_df = self.train_df.astype(
            {
                "class_id": "float32",
                "x_min": "float32",
                "y_min": "float32",
                "x_max": "float32",
                "y_max": "float32",
            }
        )

    def filter_extreme_bbox(self):
        """Filter out extreme large bbox data."""
        self.train_df["bbox_area"] = (self.train_df.x_max - self.train_df.x_min) * (
            self.train_df.y_max - self.train_df.y_min
        )

        self.train_df = self.train_df[
            (self.train_df.bbox_area < 500_000) | (self.train_df.class_id == 14)
        ]
        self.train_df.reset_index(drop=True, inplace=True)


class XrayDetectionAllNmsDataset(XrayDetectionAllDataset):
    """Dataset with duplicate bbox removed using torchvision's batched_nms"""

    def load_train_csv(self):
        self.preprocess_df()
        self.filter_extreme_bbox()

        self.image_ids = self.train_df.image_id.unique()

        # bbox NMS
        df_nms = pd.DataFrame(
            columns=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
            dtype=np.float32,
        )
        pbar = tqdm(self.image_ids)
        for image_id in pbar:
            pbar.set_description("Processing bbox nms")
            boxes, class_ids = self.get_bbox_nms(self.train_df, image_id)
            for box, class_id in zip(boxes, class_ids):
                df_nms = df_nms.append(
                    [
                        {
                            "image_id": image_id,
                            "class_id": class_id.numpy(),
                            "x_min": box[0].numpy(),
                            "y_min": box[1].numpy(),
                            "x_max": box[2].numpy(),
                            "y_max": box[3].numpy(),
                        }
                    ],
                    ignore_index=True,
                )
                df_nms[["class_id", "x_min", "y_min", "x_max", "y_max"]] = df_nms[
                    ["class_id", "x_min", "y_min", "x_max", "y_max"]
                ].astype(np.float32)

        self.train_df = df_nms

        self.train_data = {}
        max_index = len(self.image_ids)
        pbar = tqdm(range(max_index))

        for index in pbar:
            pbar.set_description("Processing train_labels")
            image_id = self.image_ids[index]
            records = self.train_df.loc[self.train_df.image_id == image_id].copy()

            self.train_data[image_id] = records[
                ["class_id", "x_min", "y_min", "x_max", "y_max"]
            ].values

        self.most_class_ids = []
        for image_id in self.image_ids:
            class_ids = self.train_data[image_id][:, 0].astype(np.int64)
            class_ids_counts = np.bincount(class_ids)
            self.most_class_ids.append(np.argmax(class_ids_counts))

    def get_bbox_nms(self, df, image_id, iou_threshold=0.4):
        boxes = torch.as_tensor(
            df[["x_min", "y_min", "x_max", "y_max"]][
                df.image_id == image_id
            ].to_numpy(),
            dtype=torch.float32,
        )
        idx = torch.as_tensor(df["class_id"][df.image_id == image_id].to_numpy())
        scores = torch.ones(idx.size(), dtype=torch.float32)
        keep = batched_nms(boxes, scores, idx, iou_threshold=iou_threshold)
        return boxes[keep], idx[keep]


class XrayDetectionAllNmsDataset_V2(XrayDetectionAllDataset):
    def load_train_csv(self):
        self.preprocess_df()
        self.filter_extreme_bbox()

        self.image_ids = self.train_df.image_id.unique()

        # bbox NMS
        df_nms = pd.DataFrame(
            columns=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
            dtype=np.float32,
        )
        pbar = tqdm(self.image_ids)
        for image_id in pbar:
            pbar.set_description("Processing bbox nms_v2")
            boxes, class_ids = self.get_bbox_nms(self.train_df, image_id)
            for box, class_id in zip(boxes, class_ids):
                df_nms = df_nms.append(
                    [
                        {
                            "image_id": image_id,
                            "class_id": class_id,
                            "x_min": box[0],
                            "y_min": box[1],
                            "x_max": box[2],
                            "y_max": box[3],
                        }
                    ],
                    ignore_index=True,
                )
                df_nms[["class_id", "x_min", "y_min", "x_max", "y_max"]] = df_nms[
                    ["class_id", "x_min", "y_min", "x_max", "y_max"]
                ].astype(np.float32)

        self.train_df = df_nms

        self.train_data = {}
        max_index = len(self.image_ids)
        pbar = tqdm(range(max_index))

        for index in pbar:
            pbar.set_description("Processing train_labels")
            image_id = self.image_ids[index]
            records = self.train_df.loc[self.train_df.image_id == image_id].copy()

            self.train_data[image_id] = records[
                ["class_id", "x_min", "y_min", "x_max", "y_max"]
            ].values

        self.most_class_ids = []
        for image_id in self.image_ids:
            class_ids = self.train_data[image_id][:, 0].astype(np.int64)
            class_ids_counts = np.bincount(class_ids)
            self.most_class_ids.append(np.argmax(class_ids_counts))

    def get_bbox_nms(self, df, image_id, iou_threshold=0.4, skip_box_thr=0.0):
        boxes = df[["x_min", "y_min", "x_max", "y_max"]][
            df.image_id == image_id
        ].to_numpy()
        idx = df["class_id"][df.image_id == image_id].to_numpy()
        scores = np.ones(len(idx), dtype=np.float32)

        boxes_max = np.max(boxes)
        boxes_normalized = boxes / boxes_max

        boxes_nms, _, labels_nms = non_maximum_weighted(
            [boxes_normalized],
            [scores],
            [idx],
            weights=None,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_thr,
        )
        boxes_nms *= boxes_max
        return boxes_nms, labels_nms


class XrayDetectionAllWbfDataset(XrayDetectionAllDataset):
    def load_train_csv(self):
        self.preprocess_df()
        self.filter_extreme_bbox()

        self.image_ids = self.train_df.image_id.unique()

        # bbox WBF
        df_wbf = pd.DataFrame(
            columns=["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"],
            dtype=np.float32,
        )
        pbar = tqdm(self.image_ids)
        for image_id in pbar:
            pbar.set_description("Processing bbox wbf")
            boxes, class_ids = self.get_bbox_wbf(self.train_df, image_id)
            for box, class_id in zip(boxes, class_ids):
                df_wbf = df_wbf.append(
                    [
                        {
                            "image_id": image_id,
                            "class_id": class_id,
                            "x_min": box[0],
                            "y_min": box[1],
                            "x_max": box[2],
                            "y_max": box[3],
                        }
                    ],
                    ignore_index=True,
                )
                df_wbf[["class_id", "x_min", "y_min", "x_max", "y_max"]] = df_wbf[
                    ["class_id", "x_min", "y_min", "x_max", "y_max"]
                ].astype(np.float32)

        self.train_df = df_wbf

        self.train_data = {}
        max_index = len(self.image_ids)
        pbar = tqdm(range(max_index))

        for index in pbar:
            pbar.set_description("Processing train_labels")
            image_id = self.image_ids[index]
            records = self.train_df.loc[self.train_df.image_id == image_id].copy()

            self.train_data[image_id] = records[
                ["class_id", "x_min", "y_min", "x_max", "y_max"]
            ].values

        self.most_class_ids = []
        for image_id in self.image_ids:
            class_ids = self.train_data[image_id][:, 0].astype(np.int64)
            class_ids_counts = np.bincount(class_ids)
            self.most_class_ids.append(np.argmax(class_ids_counts))

    def get_bbox_wbf(self, df, image_id, iou_threshold=0.6, skip_box_thr=0.0):
        boxes = df[["x_min", "y_min", "x_max", "y_max"]][
            df.image_id == image_id
        ].to_numpy()
        idx = df["class_id"][df.image_id == image_id].to_numpy()
        scores = np.ones(len(idx), dtype=np.float32)

        boxes_max = np.max(boxes)
        boxes_normalized = boxes / boxes_max

        boxes_wbf, _, labels_wbf = weighted_boxes_fusion(
            [boxes_normalized],
            [scores],
            [idx],
            weights=None,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_thr,
        )
        boxes_wbf *= boxes_max
        return boxes_wbf, labels_wbf


class XrayTestDataset(Dataset):
    def __init__(self, dataset_dir="dataset", image_ids=None, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.testset_dir = os.path.join(self.dataset_dir, "test")
        self.transform = transform
        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = os.listdir(self.testset_dir)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, int, int]:
        image_id = self.image_ids[index]
        image_path = os.path.join(self.testset_dir, image_id)

        image = read_xray(image_path, downscale_factor=1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        height, width, _ = image.shape

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = A.Compose([A.Normalize(), ToTensorV2()])(image=image)["image"]

        return image, image_id, height, width

    def __len__(self) -> int:
        return len(self.image_ids)


class XrayTestEnsembleDataset(Dataset):
    def __init__(self, dataset_dir="dataset", image_ids=None, image_size_list=[1024]):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.testset_dir = os.path.join(self.dataset_dir, "test")
        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = os.listdir(self.testset_dir)
        self.image_size_list = image_size_list

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], str, int, int]:
        image_id = self.image_ids[index]
        image_path = os.path.join(self.testset_dir, image_id)

        image = read_xray(image_path, downscale_factor=1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        height, width, _ = image.shape

        image_dict = {}
        image_size_set = set(self.image_size_list)
        for image_size in image_size_set:
            image_aug = self.transform(image_size=image_size)(image=image)["image"]
            key = "image_" + str(image_size)
            image_dict.update({key: image_aug})

        return image_dict, image_id, height, width

    def __len__(self) -> int:
        return len(self.image_ids)

    def transform(self, image_size=1024):
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
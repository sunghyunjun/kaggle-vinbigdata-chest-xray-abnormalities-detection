import os
from typing import Tuple, Dict, Union

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tqdm import tqdm


class XrayFindingDataset(Dataset):
    def __init__(self, dataset_dir="dataset-jpg", transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.csv_path = os.path.join(self.dataset_dir, "train_3x_downsampled.csv")
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


class XrayDetectionDataset(Dataset):
    def __init__(self, dataset_dir="dataset-jpg", transform=None, bboxes_yxyx=True):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.csv_path = os.path.join(self.dataset_dir, "train_3x_downsampled.csv")
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

        labels = records[:, 0]

        return image, bboxes, labels

    def load_train_csv(self):
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

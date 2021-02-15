from argparse import ArgumentParser
import os
import warnings

import cv2
import numpy as np
import pandas as pd

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

warnings.filterwarnings(action="ignore", category=UserWarning)


def read_xray(path, voi_lut=True, fix_monochrome=True, downscale_factor=1):
    # Read dicom image.
    # Original from:
    # https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    # https://www.kaggle.com/raddar/vinbigdata-competition-jpg-data-3x-downsampled
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    if downscale_factor != 1:
        new_shape = tuple([int(x / downscale_factor) for x in data.shape])
        data = cv2.resize(data, (new_shape[1], new_shape[0]))

    return data


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    raw_data_dir = os.path.join(args.dataset_dir)
    jpg_data_dir = os.path.join("dataset-jpg")

    os.makedirs(jpg_data_dir, exist_ok=True)
    os.makedirs(os.path.join(jpg_data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(jpg_data_dir, "test"), exist_ok=True)

    train_images = os.listdir(os.path.join(raw_data_dir, "train"))
    test_images = os.listdir(os.path.join(raw_data_dir, "test"))

    df = pd.read_csv(os.path.join(raw_data_dir, "train.csv"))

    IMAGE_SIZE = 1024

    print(f"Making train images - {IMAGE_SIZE} px jpg")
    if args.debug:
        pbar = tqdm(train_images[:10])
    else:
        pbar = tqdm(train_images)

    new_df = pd.DataFrame(
        columns=[
            "image_id",
            "class_name",
            "class_id",
            "rad_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
    )

    for raw_image in pbar:
        img = read_xray(
            os.path.join(raw_data_dir, "train", raw_image), downscale_factor=1
        )

        scale_x = IMAGE_SIZE / img.shape[1]
        scale_y = IMAGE_SIZE / img.shape[0]

        image_id = raw_image.split(".")[0]

        temp_df = df[df.image_id == image_id].copy()

        temp_df["raw_x_min"] = temp_df["x_min"]
        temp_df["raw_x_max"] = temp_df["x_max"]
        temp_df["raw_y_min"] = temp_df["y_min"]
        temp_df["raw_y_max"] = temp_df["y_max"]

        temp_df["raw_width"] = img.shape[1]
        temp_df["raw_height"] = img.shape[0]

        temp_df["scale_x"] = scale_x
        temp_df["scale_y"] = scale_y

        temp_df[["x_min", "x_max"]] = temp_df[["x_min", "x_max"]].mul(scale_x).round(0)
        temp_df[["y_min", "y_max"]] = temp_df[["y_min", "y_max"]].mul(scale_y).round(0)

        new_df = new_df.append(temp_df, ignore_index=True)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        cv2.imwrite(
            os.path.join(jpg_data_dir, "train", raw_image.replace(".dicom", ".jpg")),
            img,
        )

    new_df.to_csv(os.path.join(jpg_data_dir, "train.csv"))

    print(f"Making test images - {IMAGE_SIZE} px jpg")
    if args.debug:
        pbar = tqdm(test_images[:10])
    else:
        pbar = tqdm(test_images)

    for raw_image in pbar:
        img = read_xray(
            os.path.join(raw_data_dir, "test", raw_image), downscale_factor=1
        )

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        cv2.imwrite(
            os.path.join(jpg_data_dir, "test", raw_image.replace(".dicom", ".jpg")), img
        )


if __name__ == "__main__":
    main()
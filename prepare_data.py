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


def read_xray(path, voi_lut=True, fix_monochrome=True, downscale_factor=3):
    # Read dicom image and downscale 3x by default.
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

    train_files = os.listdir(os.path.join(raw_data_dir, "train"))
    test_files = os.listdir(os.path.join(raw_data_dir, "test"))

    df = pd.read_csv(os.path.join(raw_data_dir, "train.csv"))

    print("Make 3x downsampled images and csv file to 'dataset-jpg'")
    print("Making train_3x_downsampled.csv")

    df[["x_min", "y_min", "x_max", "y_max"]] = df[
        ["x_min", "y_min", "x_max", "y_max"]
    ].floordiv(3)
    df.to_csv(os.path.join(jpg_data_dir, "train_3x_downsampled.csv"))

    print("Making train images - 3x downsampled jpg")
    if args.debug:
        pbar = tqdm(train_files[:10])
    else:
        pbar = tqdm(train_files)

    for file in pbar:
        img = read_xray(os.path.join(raw_data_dir, "train", file))
        cv2.imwrite(
            os.path.join(jpg_data_dir, "train", file.replace(".dicom", ".jpg")), img
        )

    print("Making test images - 3x downsampled jpg")
    if args.debug:
        pbar = tqdm(test_files[:10])
    else:
        pbar = tqdm(test_files)

    for file in pbar:
        img = read_xray(os.path.join(raw_data_dir, "test", file))
        cv2.imwrite(
            os.path.join(jpg_data_dir, "test", file.replace(".dicom", ".jpg")), img
        )


if __name__ == "__main__":
    main()
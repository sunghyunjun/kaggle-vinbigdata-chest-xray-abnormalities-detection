from argparse import ArgumentParser

import os

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from tqdm import tqdm

from models import XrayClassifier, XrayDetector
from datamodule import XrayTestDataModule


def make_classification(model, test_loader, device, debug=False):
    image_ids = []
    predictions = []
    for index, (image, image_id, height, width) in enumerate(tqdm(test_loader)):
        if debug and index > 10:
            break
        prediction = model(image.to(device))
        image_ids.extend(image_id)
        predictions.extend(prediction.detach().cpu().numpy().squeeze())
    return image_ids, predictions


def make_classification_df(model, test_loader, device, debug=False):
    image_ids, predictions = make_classification(model, test_loader, device, debug)
    df = pd.DataFrame(data=list(zip(image_ids, predictions)), columns=["ids", "preds"])
    return df


def make_detection(model, test_loader, device, debug=False):
    resize_width = 512
    resize_height = 512
    downscale_factor = 3

    image_ids = []
    boxes_pred = []
    scores_pred = []
    labels_pred = []

    for index, (image, image_id, height, width) in enumerate(tqdm(test_loader)):
        if debug and index > 10:
            break
        prediction = model(image.to(device))

        boxes = prediction[:, :, :4].detach().cpu().numpy()
        scores = prediction[:, :, 4].detach().cpu().numpy()
        labels = prediction[:, :, 5].detach().cpu().numpy().astype(np.int32)

        height = height.detach().cpu().numpy()
        height = np.expand_dims(height, axis=1)

        width = width.detach().cpu().numpy()
        width = np.expand_dims(width, axis=1)

        boxes[:, :, 0] = boxes[:, :, 0] * width / resize_width
        boxes[:, :, 1] = boxes[:, :, 1] * height / resize_height
        boxes[:, :, 2] = boxes[:, :, 2] * width / resize_width
        boxes[:, :, 3] = boxes[:, :, 3] * height / resize_height

        boxes = boxes.astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0].clip(min=0, max=height - 1)
        boxes[:, :, 1] = boxes[:, :, 1].clip(min=0, max=width - 1)
        boxes[:, :, 2] = boxes[:, :, 2].clip(min=0, max=height - 1)
        boxes[:, :, 3] = boxes[:, :, 3].clip(min=0, max=width - 1)

        boxes *= downscale_factor

        image_ids.extend(image_id)
        boxes_pred.extend(boxes)
        scores_pred.extend(scores)
        labels_pred.extend(labels)

    return image_ids, boxes_pred, scores_pred, labels_pred


def format_pred(labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")
    return " ".join(pred_strings)


def make_detection_df(model, test_loader, device, debug=False):
    image_ids, boxes_pred, scores_pred, labels_pred = make_detection(
        model, test_loader, device, debug
    )

    ids = []
    prediction_strings = []

    # class, confidence, xmin, ymin, xmax, ymax
    for image_id, box, score, label in zip(
        image_ids, boxes_pred, scores_pred, labels_pred
    ):
        image_id = image_id.split(".")[0]
        ids.append(image_id)
        pred_string = format_pred(label, box, score)
        prediction_strings.append(pred_string)

    df = pd.DataFrame(
        data=(zip(ids, prediction_strings)), columns=["image_id", "PredictionString"]
    )

    return df


def make_normal_df(image_ids):
    ids = []
    prediction_strings = []

    for image_id in image_ids:
        image_id = image_id.split(".")[0]
        ids.append(image_id)
        pred_string = "14 1 0 0 1 1"
        prediction_strings.append(pred_string)

    df = pd.DataFrame(
        data=(zip(ids, prediction_strings)), columns=["image_id", "PredictionString"]
    )
    return df


def main():
    # ----------
    # seed
    # ----------
    pl.seed_everything(0)

    # ----------
    # args
    # ----------
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--classifier_checkpoint", default="checkpoint/b5-0.5947.ckpt")
    parser.add_argument("--detector_checkpoint", default="checkpoint/d0-0.7787.ckpt")

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)

    args = parser.parse_args()

    # ----------
    # for debug
    # ----------
    args.debug = True
    if args.debug:
        print("DEBUG Mode")

    # ----------
    # device
    # ----------
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device {device}")

    torch.set_grad_enabled(False)

    # -------------------------
    # Stage 1: Classification
    # -------------------------
    print("Stage 1: Classification - finding vs no-finding(normal)")
    dm_clf = XrayTestDataModule(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm_clf.prepare_data()
    dm_clf.setup()

    Classifier = XrayClassifier.load_from_checkpoint(
        args.classifier_checkpoint, pretrained=False
    )

    Classifier.to(device)
    Classifier.eval()

    clf_df = make_classification_df(
        Classifier, dm_clf.test_dataloader(), device, debug=args.debug
    )

    finding_df = clf_df[clf_df.preds > 0.5]
    no_finding_df = clf_df[clf_df.preds <= 0.5]

    # --------------------
    # Stage 2: Detection
    # --------------------
    print("Stage 2: Detection")
    dm_det = XrayTestDataModule(
        dataset_dir=args.dataset_dir,
        image_ids=finding_df["ids"].tolist(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm_det.prepare_data()
    dm_det.setup()

    Detector = XrayDetector.load_from_checkpoint(
        args.detector_checkpoint, pretrained=False, pretrained_backbone=False
    )

    Detector.to(device)
    Detector.eval()

    det_df = make_detection_df(
        Detector, dm_det.test_dataloader(), device, debug=args.debug
    )

    normal_df = make_normal_df(no_finding_df.ids.tolist())
    submission_df = pd.concat([det_df, normal_df])
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()

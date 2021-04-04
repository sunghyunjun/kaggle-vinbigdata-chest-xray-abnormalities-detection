from argparse import ArgumentParser

import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from timm.models.efficientnet import default_cfgs

from efficientnet_pytorch import EfficientNet

from datamodule import (
    XrayFindingDataModule,
    XrayFindingConcatDataModule,
    XrayDetectionDataModule,
    XrayDetectionNmsDataModule,
    XrayDetectionNmsDataModule_V2,
    XrayDetectionWbfDataModule,
    XrayDetectionAllDataModule,
    XrayDetectionAllNmsDataModule,
    XrayDetectionAllNmsDataModule_V2,
    XrayDetectionAllWbfDataModule,
)
from models import XrayClassifier, XrayDetector
from evaluator import XrayEvaluator, ZFTurboEvaluator


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
    parser.add_argument(
        "--mode",
        choices=["classification", "detection", "detection_all"],
    )

    parser.add_argument("--clf_dataset", default="vbd", choices=["vbd", "concat"])
    parser.add_argument(
        "--detector_bbox_filter", default="nms", choices=["raw", "nms", "nms_v2", "wbf"]
    )
    parser.add_argument("--detector_valid_bbox_filter", action="store_true")
    parser.add_argument("--freeze_batch_norm", action="store_true")
    parser.add_argument("--group_norm", action="store_true")
    parser.add_argument("--evaluator_alt", action="store_true")
    parser.add_argument("--pretrained_backbone_checkpoint", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dataset_dir", default="dataset-jpg")
    parser.add_argument("--dataset_nih_dir", default="dataset-nih")
    parser.add_argument("--default_root_dir", default=os.getcwd())
    parser.add_argument("--lr_finder", action="store_true")
    parser.add_argument("--neptune_logger", action="store_true")
    parser.add_argument("--neptune_project", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--model_name")

    # b0: 224, b1: 240, b2: 260, b3: 300
    # b4: 380, b5: 456, b6: 528, b7: 600, b8: 672
    parser.add_argument("--clf_image_size", default=456, type=int)

    # d0: 512, d1: 640, d2: 768, d3: 896
    # d4: 1024, d5: 1280, d6: 1280, d7: 1536
    parser.add_argument(
        "--detector_image_size",
        default=512,
        type=int,
        choices=[x * 128 for x in range(4, 13)],
    )

    parser.add_argument("--downconv", action="store_true")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--fold_splits", default=10, type=int)
    parser.add_argument("--fold_index", default=0, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--anchor_scale", default=4, type=int)
    parser.add_argument("--aspect_ratios_expand", action="store_true")
    parser.add_argument("--max_det_per_image", default=100, type=int)
    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)

    args = parser.parse_args()

    if args.mode is None:
        raise ValueError(
            "--mode should be one of ['classification', 'detection', 'detection_all']"
        )

    # ----------
    # for debug
    # ----------
    if args.debug:
        args.max_epochs = 1
        args.limit_train_batches = 3
        args.limit_val_batches = 3

    # ----------
    # data
    # ----------
    if args.mode == "classification":
        # resolution
        # b0: 224, b1: 240, b2: 260, b3: 300
        # b4: 380, b5: 456, b6: 528, b7: 600, b8: 672
        if args.clf_dataset == "concat":
            dm = XrayFindingConcatDataModule(
                dataset_dir=args.dataset_dir,
                dataset_nih_dir=args.dataset_nih_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=args.clf_image_size,
            )
        else:
            dm = XrayFindingDataModule(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=args.clf_image_size,
            )
    elif args.mode == "detection":
        # d0: 512, d1: 640, d2: 768, d3: 896
        # d4: 1024, d5: 1280, d6: 1280, d7: 1536
        det_image_size = args.detector_image_size
        if args.downconv:
            det_image_size *= 2
            print(
                f"downconv used\n"
                f"dataset  image size: {det_image_size}\n"
                f"detector image size: {args.detector_image_size}\n"
            )

        if args.detector_bbox_filter == "nms":
            print("Detector's bbox filter: NMS")
            dm = XrayDetectionNmsDataModule(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
                valid_filter=args.detector_valid_bbox_filter,
            )
        elif args.detector_bbox_filter == "nms_v2":
            print("Detector's bbox filter: NMS_V2")
            dm = XrayDetectionNmsDataModule_V2(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
                valid_filter=args.detector_valid_bbox_filter,
            )
        elif args.detector_bbox_filter == "wbf":
            print("Detector's bbox filter: WBF")
            dm = XrayDetectionWbfDataModule(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
                valid_filter=args.detector_valid_bbox_filter,
            )
        else:
            print("Detector's bbox filter: None, Raw")
            dm = XrayDetectionDataModule(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
            )
    elif args.mode == "detection_all":
        # d0: 512, d1: 640, d2: 768, d3: 896
        # d4: 1024, d5: 1280, d6: 1280, d7: 1536
        det_image_size = args.detector_image_size
        if args.downconv:
            det_image_size *= 2

        if args.detector_bbox_filter == "nms":
            print("Detector's bbox filter: NMS")
            dm = XrayDetectionAllNmsDataModule(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
                valid_filter=args.detector_valid_bbox_filter,
            )
        elif args.detector_bbox_filter == "nms_v2":
            print("Detector's bbox filter: NMS_V2")
            dm = XrayDetectionAllNmsDataModule_V2(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
                valid_filter=args.detector_valid_bbox_filter,
            )
        elif args.detector_bbox_filter == "wbf":
            print("Detector's bbox filter: WBF")
            dm = XrayDetectionAllWbfDataModule(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
                valid_filter=args.detector_valid_bbox_filter,
            )
        else:
            print("Detector's bbox filter: None, Raw")
            dm = XrayDetectionAllDataModule(
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                fold_splits=args.fold_splits,
                fold_index=args.fold_index,
                image_size=det_image_size,
            )

    # ----------
    # model
    # ----------
    if args.mode == "classification":
        dm.setup()

        model = XrayClassifier(
            model_name=args.model_name,
            init_lr=args.init_lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            group_norm=args.group_norm,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="xray-classifier-{epoch:03d}-{val_loss:.4f}",
            save_last=True,
            save_top_k=3,
            mode="min",
        )
    elif args.mode == "detection" or args.mode == "detection_all":
        dm.setup()

        include_nofinding = False if args.mode == "detection" else True
        evaluator = XrayEvaluator(
            dataset=dm.valid_dataset,
            include_nofinding=include_nofinding,
            downconv=args.downconv,
        )

        evaluator_alt = (
            ZFTurboEvaluator(
                image_size=args.detector_image_size, include_nofinding=include_nofinding
            )
            if args.evaluator_alt
            else None
        )

        num_classes = 14 if args.mode == "detection" else 15
        print(f"num_classes: {num_classes}")

        model = XrayDetector(
            model_name=args.model_name,
            init_lr=args.init_lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            num_classes=num_classes,
            anchor_scale=args.anchor_scale,
            aspect_ratios_expand=args.aspect_ratios_expand,
            evaluator=evaluator,
            evaluator_alt=evaluator_alt,
            image_size=args.detector_image_size,
            freeze_batch_norm=args.freeze_batch_norm,
            group_norm=args.group_norm,
            pretrained_backbone_checkpoint=args.pretrained_backbone_checkpoint,
            max_det_per_image=args.max_det_per_image,
            downconv=args.downconv,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="xray-detector-{epoch:03d}-{val_loss:.4f}",
            save_last=True,
            save_top_k=3,
            mode="min",
        )

    # ----------
    # logger
    # ----------
    if args.neptune_logger:
        logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project_name=args.neptune_project,
            experiment_name=args.experiment_name,
            params=args.__dict__,
            tags=["pytorch-lightning"],
        )
    else:
        # default logger: TenserBoardLogger
        logger = True

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ----------
    # training
    # ----------
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, lr_monitor], logger=logger
    )

    if args.lr_finder:
        lr_finder = trainer.tuner.lr_find(model=model, datamodule=dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("./suggested_lr.png")
        suggested_lr = lr_finder.suggestion()
        print(suggested_lr)
    else:
        trainer.fit(model, dm)

    # ----------
    # cli example
    # ----------
    # python train.py --debug --mode=classification --model_name="tf_efficientnet_b0"
    # python train.py --debug --mode=classification --model_name="tf_efficientnet_b0" --clf_dataset=concat
    # python train.py --debug --mode=classification --model_name="tf_efficientnet_b0" --group_norm
    # python train.py --debug --mode=detection --model_name="tf_efficientdet_d0"
    # python train.py --debug --mode=detection --model_name="tf_efficientdet_d0" --detector_bbox_filter=raw
    # python train.py --debug --mode=detection --model_name="tf_efficientdet_d0" --detector_bbox_filter=nms
    # python train.py --debug --mode=detection --model_name="tf_efficientdet_d0" --detector_bbox_filter=nms_v2
    # python train.py --debug --mode=detection --model_name="tf_efficientdet_d0" --detector_bbox_filter=raw --downconv
    # python train.py --debug --mode=detection_all --model_name="tf_efficientdet_d0" --detector_bbox_filter=raw


if __name__ == "__main__":
    main()
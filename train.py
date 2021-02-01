from argparse import ArgumentParser

import os

# import random

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from efficientnet_pytorch import EfficientNet

from datamodule import XrayFindingDataModule, XrayDetectionDataModule
from models import XrayClassifier, XrayDetector
from evaluator import XrayEvaluator


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
    parser.add_argument("--mode", choices=["classification", "detection"])
    parser.add_argument("--dataset_dir", default="dataset-jpg")
    parser.add_argument("--default_root_dir", default=os.getcwd())
    parser.add_argument("--lr_finder", action="store_true")

    parser.add_argument("--neptune_logger", action="store_true")
    parser.add_argument("--neptune_project", default=None)
    parser.add_argument("--experiment_name", default=None)

    parser.add_argument("--gpus", default=None, type=int)
    parser.add_argument("--precision", default=32, type=int)

    parser.add_argument("--model_name")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--fold_index", default=0, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--anchor_scale", default=4, type=int)

    parser.add_argument("--init_lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)

    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)

    args = parser.parse_args()

    if args.mode is None:
        raise ValueError("--mode should be one of ['classification', 'detection']")

    # ----------
    # for debug
    # ----------
    if args.debug:
        args.max_epochs = 1
        args.limit_train_batches = 10
        args.limit_val_batches = 10

    # ----------
    # data
    # ----------
    if args.mode == "classification":
        # resolution
        # b0: 224, b1: 240, b2: 260, b3: 300
        # b4: 380, b5: 456, b6: 528, b7: 600, b8: 672
        image_size = EfficientNet.get_image_size(args.model_name)

        dm = XrayFindingDataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fold_index=args.fold_index,
            image_size=image_size,
        )
    elif args.mode == "detection":
        image_size = 512
        dm = XrayDetectionDataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fold_index=args.fold_index,
            image_size=image_size,
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
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="xray-classifier-{epoch:03d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
        )
    elif args.mode == "detection":
        dm.setup()
        evaluator = XrayEvaluator(dm.valid_dataset)

        model = XrayDetector(
            model_name=args.model_name,
            init_lr=args.init_lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            anchor_scale=args.anchor_scale,
            evaluator=evaluator,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="xray-detector-{epoch:03d}-{val_loss:.4f}",
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
    # python train.py --debug --mode=classification --model_name="efficientnet-b0"
    # python train.py --debug --mode=detection --model_name="tf_efficientdet_d0"


if __name__ == "__main__":
    main()
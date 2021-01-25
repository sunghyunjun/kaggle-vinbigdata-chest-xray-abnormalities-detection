from argparse import ArgumentParser

# import os
# import random

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import XrayFindingDataModule, XrayDetectionDataModule
from models import XrayClassifier, XrayDetector


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

    parser.add_argument("--model_name")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--fold_index", default=0, type=int)

    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser = pl.Trainer.add_argparse_args(parser)
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
    # model
    # ----------
    if args.mode == "classification":
        model = XrayClassifier(
            model_name=args.model_name,
            init_lr=args.init_lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="xray-classifier-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )
    elif args.mode == "detection":
        model = XrayDetector(
            model_name=args.model_name,
            init_lr=args.init_lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="xray-detector-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )

    # ----------
    # data
    # ----------
    if args.mode == "classification":
        dm = XrayFindingDataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fold_index=args.fold_index,
            image_size=model.image_size,
        )
    elif args.mode == "detection":
        dm = XrayDetectionDataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fold_index=args.fold_index,
            image_size=model.image_size,
        )

    # ----------
    # logger
    # ----------
    if args.debug:
        pass
    else:
        lr_monitor = LearningRateMonitor(logging_interval="step")

    # ----------
    # training
    # ----------
    if args.debug:
        trainer = pl.Trainer.from_argparse_args(args)
    else:
        trainer = pl.Trainer.from_argparse_args(
            args, callbacks=[checkpoint_callback, lr_monitor]
        )

    trainer.fit(model, dm)

    # ----------
    # cli example
    # ----------
    # python train.py --debug --mode=classification --model_name="efficientnet-b0"


if __name__ == "__main__":
    main()
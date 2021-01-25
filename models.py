import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from efficientnet_pytorch import EfficientNet
from effdet import create_model, create_model_from_config
from effdet.config import get_efficientdet_config


class XrayClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name="efficientnet-b0",
        pretrained=True,
        init_lr=1e-4,
        weight_decay=1e-5,
        max_epochs=10,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        # resolution
        # b0: 224, b1: 240, b2: 260, b3: 300, b4: 380, b5: 456, b6: 528, b7: 600, b8: 672
        self.image_size = EfficientNet.get_image_size(self.model_name)
        self.num_classes = 1
        self.model = self.get_model(self.model_name, self.pretrained)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.train_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
        self.valid_confmat = pl.metrics.ConfusionMatrix(num_classes=2)

        self.save_hyperparameters()

    def forward(self, x):
        target = F.sigmoid(self.model(x))
        return target

    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        pred = F.sigmoid(output)
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.log("train_loss", loss)
        self.log("train_acc_step", self.train_acc(pred, target))
        self.log("train_confmat_step", self.train_confmat(pred, target))
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("train_acc_epoch", self.train_acc.compute())
        self.log("train_confmat_epoch", self.train_confmat.compute())

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        pred = F.sigmoid(output)
        loss = F.binary_cross_entropy_with_logits(output, target)
        self.log("val_loss", loss)
        self.log("val_acc_step", self.valid_acc(pred, target))
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val_acc_epoch", self.valid_acc.compute())
        self.log("val_confmat_epoch", self.valid_confmat.compute())

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        print(f"CosineAnnealingLR T_max epochs = {self.max_epochs}")
        return [optimizer], [scheduler]

    def get_model(self, model_name="efficientnet-b0", pretrained=True):
        if pretrained:
            model = EfficientNet.from_pretrained(model_name)
        else:
            model = EfficientNet.from_pretrained(model_name)

        num_in_features = model._fc.in_features
        model.fc = nn.Linear(num_in_features, self.num_classes)

        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--init_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parser


class XrayDetector(pl.LightningModule):
    def __init__(
        self,
        model_name="tf_efficientdet_d0",
        pretrained=True,
        init_lr=1e-4,
        weight_decay=1e-5,
        max_epochs=10,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.model = self.get_model(self.model_name, self.pretrained)
        self.image_size = 512

        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        boxes = batch["bboxes"]
        labels = batch["labels"]
        output = self.model(image, {"bbox": boxes, "cls": labels})
        loss = output["loss"]
        class_loss = output["class_loss"]
        box_loss = output["box_loss"]

        self.log("tr_loss", loss)
        self.log("tr_cls_loss", class_loss)
        self.log("tr_box_loss", box_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        boxes = batch["bboxes"]
        labels = batch["labels"]
        output = self.model(
            image, {"bbox": boxes, "cls": labels, "img_scale": None, "img_size": None}
        )
        loss = output["loss"]
        class_loss = output["class_loss"]
        box_loss = output["box_loss"]

        self.log("val_loss", loss)
        self.log("val_cls_loss", class_loss)
        self.log("val_box_loss", box_loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        print(f"CosineAnnealingLR T_max epochs = {self.max_epochs}")
        return [optimizer], [scheduler]

    def get_model(self, model_name="tf_efficientdet_d0", pretrained=True):
        config = get_efficientdet_config(model_name)
        config.image_size = (512, 512)
        num_classes = 13

        model = create_model_from_config(
            config=config,
            banch_task="train",
            num_classes=num_classes,
            pretrained=pretrained,
            checkpoint_path="",
            checkpoint_ema=False,
            bench_labeler=True,
        )

        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--init_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parser
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from efficientnet_pytorch import EfficientNet
from effdet import create_model, create_model_from_config
from effdet.config import get_efficientdet_config
from effdet.bench import _post_process, _batch_detection

from map_boxes import mean_average_precision_for_boxes

from evaluator import XrayEvaluator, ZFTurboEvaluator


def set_bn_eval(m):
    classname = m.__class__.__name__
    if "BatchNorm2d" in classname:
        m.affine = False
        m.weight.requires_grad = False
        m.bias.requires_grad = False
        m.eval()


def freeze_bn(model):
    model.apply(set_bn_eval)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    # return setattr(rgetattr(obj, pre) if pre else obj, post, val)
    return setattr(operator.attrgetter(pre)(obj) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def set_gn(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn = operator.attrgetter(name)(model)
            gn = torch.nn.GroupNorm(8, bn.num_features)
            rsetattr(model, name, gn)


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
        self.num_classes = 1
        self.model = self.get_model(self.model_name, self.pretrained)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

        self.valid_precision = pl.metrics.Precision()
        self.valid_recall = pl.metrics.Recall()
        self.valid_roc = pl.metrics.ROC()

        self.save_hyperparameters()

    def forward(self, x):
        target = torch.sigmoid(self.model(x))
        return target

    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        pred = torch.sigmoid(output)

        pred = torch.squeeze(pred)
        target = target.double()

        loss = F.binary_cross_entropy_with_logits(pred, target)
        self.log("train_loss", loss)
        self.log("train_acc_step", self.train_acc(pred, target))
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("train_acc_epoch", self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        pred = torch.sigmoid(output)

        pred = torch.squeeze(pred)
        target = target.float()

        loss = F.binary_cross_entropy_with_logits(pred, target)
        self.log("val_loss", loss)
        self.log("val_acc_step", self.valid_acc(pred, target))
        # return loss
        return {"pred": pred, "target": target}

    def validation_epoch_end(self, validation_step_outputs):
        preds = None
        targets = None
        for out in validation_step_outputs:
            if preds is None:
                preds = out["pred"]
            else:
                preds = torch.cat([preds, out["pred"]], dim=0)

            if targets is None:
                targets = out["target"]
            else:
                targets = torch.cat([targets, out["target"]], dim=0)

        precision = self.valid_precision(preds, targets)
        recall = self.valid_recall(preds, targets)

        try:
            fpr, tpr, thresholds = self.valid_roc(preds, targets)
            auc = pl.metrics.functional.classification.auc(fpr, tpr)
            self.log("val_auc", auc)
        except ValueError:
            pass

        self.log("val_acc_epoch", self.valid_acc.compute())
        self.log("val_precision", precision)
        self.log("val_recall", recall)

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
            model = EfficientNet.from_name(model_name)

        num_in_features = model._fc.in_features
        model._fc = nn.Linear(num_in_features, self.num_classes)

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
        anchor_scale=4,
        aspect_ratios_expand=False,
        evaluator: XrayEvaluator = None,
        evaluator_alt: ZFTurboEvaluator = None,
        pretrained_backbone=True,
        image_size=512,
        freeze_batch_norm=False,
        group_norm=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.pretrained_backbone = pretrained_backbone
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.anchor_scale = anchor_scale
        self.aspect_ratios_expand = aspect_ratios_expand
        self.image_size = image_size
        self.freeze_batch_norm = freeze_batch_norm
        self.group_norm = group_norm
        self.model = self.get_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            anchor_scale=self.anchor_scale,
            aspect_ratios_expand=self.aspect_ratios_expand,
            pretrained_backbone=self.pretrained_backbone,
            image_size=self.image_size,
            freeze_batch_norm=self.freeze_batch_norm,
            group_norm=self.group_norm,
        )
        self.evaluator = evaluator
        self.evaluator_alt = evaluator_alt

        self.save_hyperparameters()

    def forward(self, x):
        # output = self.model.forward(x)
        # return output["detections"]
        class_out, box_out = self.model.model(x)
        class_out, box_out, indices, classes = _post_process(
            class_out,
            box_out,
            num_levels=self.model.num_levels,
            num_classes=self.model.num_classes,
            max_detection_points=self.model.max_detection_points,
        )

        return _batch_detection(
            x.shape[0],
            class_out,
            box_out,
            self.model.anchors.boxes,
            indices,
            classes,
            max_det_per_image=self.model.max_det_per_image,
            soft_nms=self.model.soft_nms,
        )

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
        detections = output["detections"]

        if batch_idx == 0:
            self.initial_batch_size = len(image)

        current_batch_size = len(image)

        img_idx = (
            torch.arange(0, current_batch_size, dtype=torch.int64)
            + self.initial_batch_size * batch_idx
        )

        targets = {}
        targets["img_idx"] = img_idx
        targets["bbox"] = boxes
        targets["cls"] = labels

        if self.evaluator:
            self.evaluator.add_predictions(detections, targets)

        if self.evaluator_alt:
            self.evaluator_alt.add_predictions(detections, targets)

        self.log("val_loss", loss)
        self.log("val_cls_loss", class_loss)
        self.log("val_box_loss", box_loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        if self.evaluator:
            metrics = self.evaluator.evaluate()
            for key, value in metrics.items():
                self.log(key, value)

        if self.evaluator_alt:
            metrics = self.evaluator_alt.evaluate()
            for key, value in metrics.items():
                self.log(key, value)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        print(f"CosineAnnealingLR T_max epochs = {self.max_epochs}")
        return [optimizer], [scheduler]

    def get_model(
        self,
        model_name="tf_efficientdet_d0",
        pretrained=True,
        anchor_scale=4,
        pretrained_backbone=True,
        image_size=512,
        aspect_ratios_expand=False,
        freeze_batch_norm=False,
        group_norm=False,
    ):
        config = get_efficientdet_config(model_name)
        config.image_size = (image_size, image_size)
        num_classes = 14
        config.anchor_scale = anchor_scale
        if aspect_ratios_expand:
            config.aspect_ratios = [
                (1.0, 1.0),
                (1.4, 0.7),
                (0.7, 1.4),
                (1.8, 0.6),
                (0.6, 1.8),
            ]
            pretrained = False

        model = create_model_from_config(
            config=config,
            bench_task="train",
            num_classes=num_classes,
            pretrained=pretrained,
            checkpoint_path="",
            checkpoint_ema=False,
            bench_labeler=True,
            pretrained_backbone=pretrained_backbone,
        )

        if freeze_batch_norm:
            freeze_bn(model)

        if group_norm:
            print("BatchNorm changed to GroupNorm")
            set_gn(model)

        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--init_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parser
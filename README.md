# kaggle-vinbigdata-chest-xray-abnormalities-detection

Code for 91th place solution in [Kaggle VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection).

- [Code Overview](https://github.com/sunghyunjun/kaggle-vinbigdata-chest-xray-abnormalities-detection/blob/main/README.md#code-overview)

  - [Prepare dataset](https://github.com/sunghyunjun/kaggle-vinbigdata-chest-xray-abnormalities-detection/blob/main/README.md#prepare-dataset)

  - [Train](https://github.com/sunghyunjun/kaggle-vinbigdata-chest-xray-abnormalities-detection/blob/main/README.md#train)

  - [Submission Notebook](https://github.com/sunghyunjun/kaggle-vinbigdata-chest-xray-abnormalities-detection/blob/main/README.md#submission-notebook)

- [Solution Summary](https://github.com/sunghyunjun/kaggle-vinbigdata-chest-xray-abnormalities-detection/blob/main/README.md#solution-summary)

- *Read Solution Summary in other languages: [English](https://github.com/sunghyunjun/kaggle-vinbigdata-chest-xray-abnormalities-detection/blob/main/README.md#solution-summary), [한국어](https://github.com/sunghyunjun/kaggle-vinbigdata-chest-xray-abnormalities-detection/blob/main/README.ko.md)*

## Code Overview

---

## Prepare dataset

My solution use 1024px resized dataset. And you have two options.

- Download original dataset (~192 GB) and create resized dataset.
- Or download kaggle public dataset already made (3.59 GB).

### Option 1. - Download original dataset and create 1024px resized dataset

Download Original Competition data [on the Kaggle competition page](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data). And extract zip file.

```bash
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection
unzip vinbigdata-chest-xray-abnormalities-detection.zip -d dataset
```

Create 1024px resized dataset. It will be saved to "dataset-jpg" folder. It takes some hours.

```bash
python prepare_data.py --dataset_dir=dataset
```

### Option 2. - Download kaggle public dataset already made

Alternatively, you can use kaggle public dataset - [VinBigData 1024 JPG Dataset](https://www.kaggle.com/sunghyunjun/vinbigdata-1024-jpg-dataset)

```bash
kaggle datasets download -d sunghyunjun/vinbigdata-1024-jpg-dataset
unzip vinbigdata-1024-jpg-dataset.zip -d dataset-jpg
```

## Train

### Train classifier

```bash
python train.py \
--mode=classification \
--model_name="tf_efficientnet_b5" \
--default_root_dir=./checkpoint_path \
--gpus=1 \
--clf_image_size=600 \
--batch_size=12 \
--num_workers=4 \
--fold_splits=5 \
--fold_index=0 \
--max_epochs=50 \
--init_lr=0.75e-4 \
--weight_decay=1e-4 \
--precision=16 \
```

### Train detector for 14-class

```bash
python train.py \
--mode=detection \
--model_name="tf_efficientdet_d5" \
--default_root_dir=./checkpoint_path \
--gpus=1 \
--detector_image_size=896 \
--batch_size=3 \
--num_workers=2 \
--fold_splits=5 \
--fold_index=0 \
--max_epochs=50 \
--init_lr=3e-4 \
--weight_decay=1e-3 \
--detector_bbox_filter=nms_v2 \
--detector_valid_bbox_filter \
--progress_bar_refresh_rate=30 \
--precision=16 \
```

### Train detector for 15-class

```bash
python train.py \
--mode=detection_all \
--model_name="tf_efficientdet_d4" \
--default_root_dir=./checkpoint_path \
--gpus=1 \
--detector_image_size=896 \
--batch_size=4 \
--num_workers=2 \
--fold_splits=5 \
--fold_index=0 \
--max_epochs=50 \
--init_lr=4e-4 \
--weight_decay=1e-3 \
--detector_bbox_filter=nms_v2 \
--detector_valid_bbox_filter \
--precision=16 \
```

You can use Pytorch-Lightning Trainer's flags.

```bash
--resume_from_checkpoint=./checkpoint_path/saved_checkpoint.ckpt
```

Also you can use neptune logger.

```bash
--neptune_logger \
--neptune_project="your_project_name" \
--experiment_name="your_exp_name" \
```

And there are some flags for experiments.

`--debug` : Debug mode. run script for short iteration.

`--aspect_ratios_expand` : Set config.aspect_ratios of efficientdet as follows.

```python
config.aspect_ratios = [
    (1.0, 1.0),
    (1.4, 0.7),
    (0.7, 1.4),
    (1.8, 0.6),
    (0.6, 1.8),
]
```

`--clf_dataset` : Select dataset for classifier, "vbd" or "concat". If you choose "concat" you have to set --dataset_nih_dir flag.

`--dataset_nih_dir` : Path of NIH dataset. You can download as follows. [NIH Chest X-rays 600 JPG Dataset](https://www.kaggle.com/sunghyunjun/nih-chest-xrays-600-jpg-dataset)

```bash
kaggle datasets download -d sunghyunjun/nih-chest-xrays-600-jpg-dataset
unzip -q nih-chest-xrays-600-jpg-dataset.zip -d dataset-nih
```

`--evaluator_alt` : Use addtional evaluator of ZFTurbo's mAP. [ZFTurbo/Mean-Average-Precision-for-Boxes](https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes)

`--freeze_batch_norm`

`--group_norm`

`--accumulate_grad_batches`

`--downconv` : Add downconv module. Using downconv, efficientdet's input image size is equal to the half of --detector_image_size.

If you need to use full-size jpg dataset, you may check kaggle's public dataset. [VinBigData Original Image Dataset](https://www.kaggle.com/awsaf49/vinbigdata-original-image-dataset)

If you experiment downconv, run as follows.

```bash
kaggle datasets download -d awsaf49/vinbigdata-original-image-dataset
unzip -q vinbigdata-original-image-dataset.zip -d dataset-jpg

mv dataset-jpg/vinbigdata/* dataset-jpg
rmdir dataset-jpg/vinbigdata

python train.py \
--mode=detection \
--model_name="tf_efficientdet_d1" \
--default_root_dir=./checkpoint_path \
--gpus=1 \
--detector_image_size=1280 \
--downconv \
--batch_size=4 \
--num_workers=2 \
--fold_splits=5 \
--fold_index=0 \
--max_epochs=50 \
--init_lr=4e-4 \
--weight_decay=1e-3 \
--detector_bbox_filter=nms_v2 \
--detector_valid_bbox_filter \
--precision=16 \
```

## Submission Notebook

[91th place VBD inference CLF DET and DET ALL](https://www.kaggle.com/sunghyunjun/91th-place-vbd-inference-clf-det-and-det-all)

---

## Solution Summary

The ensembled results of the two-stage approach and the one-stage approach. The two-stage approach consists of 2-class classifier and 14-class detector. And the one-stage approach is 15-class detector.

## Tools

- Colab Pro, Tesla V100 16GB single GPU
- GCS
- Pytorch Lightning
- Neptune
- Kaggle API

## Validation

StratifiedKFold was used and the data set was composed of 5 folds.

The classifier of the two-stage approach was trained using all normal and abnormal images, and only abnormal images were used for the detector.

The one-stage detector was trained using all images of normal and abnormal.

## Fusing BBoxes

The overlapped bboxes in both train set and validation set were fused using nms. I tested it using batched_nms of torchvision, nms and wbf of ZFTurbo.

14-class Efficientdet d4 896px 30 epochs without classifier, local cv on positive image only
||cv(mAP@iou=0.4)|public LB|private LB|
|-|-|-|-|
|torchvision batched_nms|0.4317|0.155|0.168|
|ZFTurbo nms|0.4419|0.164|0.181|
|ZFTurbo wbf|0.4158|0.157|0.185|

I thought direct comparison of local cv was not possible, and LB was also somewhat difficult to use as a criterion for judgement.

After considering the labeling method of the test set, I thought that the nms method was more similar to the labeling method, so I decided to use nms.

[[Discussion] nms > weighted box fusion?](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/214906#1199596)

### torchvision's nms vs ZFTurbo's nms

Torchvision'nms has some problems in the order of results when the bbox scores are the same.

[[Stackoverflow] Pytorch argsort ordered, with duplicate elements in the tensor](https://stackoverflow.com/questions/56176439/pytorch-argsort-ordered-with-duplicate-elements-in-the-tensor)

[torchvision.ops.nms](https://pytorch.org/vision/stable/ops.html#torchvision.ops.nms)

> If multiple boxes have the exact same score and satisfy the IoU criterion with respect to a reference box, the selected box is not guaranteed to be the same between CPU and GPU. This is similar to the behavior of argsort in PyTorch when repeated values are present.

It doesn't matter when you convert the label only once at first and then save and load it as csv, pickle, etc., but if you convert the label and use it in a new environment, the consistency of the bbox may not be maintained.

I decided to use ZFTurbo's nms which uses numpy.argsort, as this part would interfere with experiment flexibility and consistency during training.

[[GitHub] ZFTurbo/Weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)

## Model training

All models were trained on Colab Pro's V100 16GB single GPU.

- AdamW
- CosineAnnealingLR
- epochs = 50
- checkpoint selection : max mAP among top-3 min val_loss

### Two-stage approach

- 2-class classifier : EfficientNet, Resnet200d, total 15 models

|model|image size(px)|folds|batch size|init lr|weight decay|val_acc|auc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|b5|600|5 of 5|12|7.5e-5|1.0e-4|0.9601|0.9930|
|b6|528|5 of 5|12|7.5e-5|1.0e-3|0.9553|0.9927|
|resnet200d|600|3 of 5|12|7.5e-5|1.0e-4|0.9541|0.9934|
|b5|456|single|16|1.0e-4|1.0e-4|0.9557|0.9927|
|b5|1024|single|4|2.5e-5|1.0e-4|0.9577|0.9936|

- 14-class detector : EfficientDet with 2-class classifier, total 18 models, local cv on positive image only

|model|image size(px)|folds|batch size|init lr|weight decay|cv(mAP@iou=0.4)|public LB|private LB|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|d3|1024|single|3|3e-4|1e-3|0.4545|0.209|0.250|
|d4|896|5 of 5|4|4e-4|1e-4|0.4541|0.218|0.250|
|d4|896|single|4|4e-4|1e-3|0.4606|0.257|0.247|
|d4|1024|single|3|3e-4|1e-3|0.4545|0.228|0.249|
|d5|768|5 of 5|4|4e-4|1e-3|0.4472|0.225|0.253|
|d5|896|4 of 5|3|3e-4|1e-3|0.4522|0.214|0.250|
|d5|1024|single|2|2e-4|1e-3|0.4462|0.214|0.232|

### One-stage approach

- 15-class detector, total 2 models, local cv on positive image only

|model|image size(px)|folds|batch size|init lr|weight decay|cv(mAP@iou=0.4)|public LB|private LB|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|d4|896|2 of 5|4|4e-4|1e-3|0.4546|0.230|0.246|

At batch size < 4, the mAP result was poor. The larger the image size, the better the mAP, but no further training was possible.

I tried Freeze BatchNorm, accumulate grad batches, and GroupNorm but I didn't get any better results.

I also tested Downconv, but the AP of small objects like Calcification increased, but the AP of ILD and large objects decreased, so the overall mAP was not improved.

[[RANZCR CLiP] 11th Place Solution - Utilizing High resolution, Annotations, and Unlabeled data](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226557)

## Augmentation

Resize, scale, and crop were configured by referring to the method in the paper of EfficientDet.

CLAHE, equalize, invertimg, huesaturationvalue, randomgamma, shiftscalerotate did not work.

```python
A.Compose(
[
    A.Resize(height=self.resize_height, width=self.resize_width),
    A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
    A.PadIfNeeded(
        min_height=self.resize_height,
        min_width=self.resize_width,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0,
    ),
    A.RandomCrop(height=self.resize_height, width=self.resize_width, p=1.0),
    A.RandomBrightnessContrast(p=0.8),
    A.ChannelDropout(p=0.5),
    A.OneOf(
        [
            A.MotionBlur(p=0.5),
            A.MedianBlur(p=0.5),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),
        ],
        p=0.5,
    ),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
],
```

## Post processing

In the case of the two-stage approach, it was difficult to find an appropriate threshold for the classifier. The threshold was determined to be a 60~70% normal case. And if it was normal, all detections of the detector were excluded.

In the case of the one-stage approach, similarly, detections were excluded if it was normal based on the threshold.

## Blending

The results of the two-stage approach and the one-stage approach were blended using nms, and the final result is as follows.

### two-stage approach, 15 classifier + 18 detector

|normal thr|public LB|private LB||
|-|-|-|-|
|0.70|0.219|**0.246**|**final submission**|
|0.65|0.217|0.255|
|0.60|0.215|0.256|

### one-stage approach, 2 detector

|normal thr|public LB|private LB|
|-|-|-|
|0.10|0.198|0.245|
|0.00|0.198|0.245|

### two-stage + one-stage

|thr|public LB|private LB||
|-|-|-|-|
|0.70-0.30|0.224|**0.253**|**final submission**|
|0.65-0.30|0.222|0.259|
|0.60-0.30|0.220|0.258|

## What did not work

Freeze BatchNorm

accumulate grad batches

GroupNorm, group per channel = 8

GroupNorm, pretrained backbone with GroupNorm

Downconv

NIH dataset concat classifier

change aspect ratio

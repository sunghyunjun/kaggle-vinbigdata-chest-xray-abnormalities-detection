# kaggle-vinbigdata-chest-xray-abnormalities-detection

Code for 91th place solution in [Kaggle VinBigData Chest X-ray Abnormalities Detection][kaggle_link].

[kaggle_link]: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

## Summary

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

14-class Efficientdet d4 896px 30 epochs without classifier
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

- 14-class detector : EfficientDet, total 18 models

|model|image size(px)|folds|batch size|init lr|weight decay|mAP|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|d3|1024|single|3|3e-4|1e-3|0.4545|
|d4|896|5 of 5|4|4e-4|1e-4|0.4541|
|d4|896|single|4|4e-4|1e-3|0.4606|
|d4|1024|single|3|3e-4|1e-3|0.4545|
|d5|768|5 of 5|4|4e-4|1e-3|0.4472|
|d5|896|4 of 5|3|3e-4|1e-3|0.4522|
|d5|1024|single|2|2e-4|1e-3|0.4462|

### One-state approach

- 15-class detector, total 2 models

|model|image size(px)|folds|batch size|init lr|weight decay|mAP|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|d4|896|2 of 5|4|4e-4|1e-3|0.4546|

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

## What did not work

Freeze BatchNorm

accumulate grad batches

GroupNorm, group per channel = 8

GroupNorm, pretrained backbone with GroupNorm

Downconv

NIH dataset concat classifier

change aspect ratio

---

## 요약

2단계 방식과 1단계 방식의 결과를 앙상블 하였습니다. 2단계 방식은 2-class classifier와 14-class detector로 구성하였고, 1단계 방식은 15-class detector를 사용하였습니다.

## 사용도구

- Colab Pro, Tesla V100 16GB single GPU
- GCS
- Pytorch Lightning
- Neptune
- Kaggle API

## 데이터 검증

StratifiedKFold를 사용하였으며 5 fold 로 데이터 세트를 구성하였습니다.

2단계 방식의 classifier는 normal, abnormal 전체 이미지를 사용하여 훈련하였으며, detector는 abnormal 이미지만 사용하였습니다. 1단계 방식의 detector는 normal, abnormal 전체 이미지를 사용하여 훈련하였습니다.

## 경계박스 정제

train set과 validation set 모두 중첩된 bboxes를 nms를 이용하여 정제하였습니다. torchvision의 batched_nms, ZFTurbo의 nms, wbf 를 사용하여 테스트를 하였습니다.

14-class Efficientdet d4 896px 30 epochs without classifier
||cv(mAP@iou=0.4)|public LB|private LB|
|-|-|-|-|
|torchvision batched_nms|0.4317|0.155|0.168|
|ZFTurbo nms|0.4419|0.164|0.181|
|ZFTurbo wbf|0.4158|0.157|0.185|

local cv는 직접비교가 불가능하고 LB 또한 판단기준으로는 삼기가 어려워 어떤 방법을 사용할 지 고민하였습니다.

test set 의 labeling 방법을 고려했을때 nms 방식이 좀 더 유사하다고 생각하여 nms를 사용하기로 결정하였습니다.

[[Discussion] nms > weighted box fusion?](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/214906#1199596)

### torchvision's nms, ZFTurbo's nms

torchvision의 nms는 bbox의 score가 동일할 경우 결과값 순서에 있어 torch.argsort에서 발생하는 아래와 같은 현상과 유사한 현상이 나타납니다.

[[Stackoverflow] Pytorch argsort ordered, with duplicate elements in the tensor](https://stackoverflow.com/questions/56176439/pytorch-argsort-ordered-with-duplicate-elements-in-the-tensor)

[torchvision.ops.nms](https://pytorch.org/vision/stable/ops.html#torchvision.ops.nms)

> If multiple boxes have the exact same score and satisfy the IoU criterion with respect to a reference box, the selected box is not guaranteed to be the same between CPU and GPU. This is similar to the behavior of argsort in PyTorch when repeated values are present.

최초에 라벨을 한번만 변환시키고 이를 csv, pickle 등으로 저장, 불러와 사용할 때는 문제가 되지 않지만, 새로운 환경에서 새로이 라벨을 변환 시켜 사용할 경우에는 bbox의 일관성이 유지가 되지 않을 수 있습니다.

이런 부분은 실험의 유연성이나 훈련시 일관성에 방해가 된다고 판단하여, numpy.argsort를 사용하는 ZFTurbo의 nms를 사용하기로 결정하였습니다.

[[GitHub] ZFTurbo/Weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)

## 모델 훈련

모든 모델은 Colab Pro의 V100 16GB Single GPU로 훈련하였습니다.

- AdamW
- CosineAnnealingLR
- epochs = 50
- checkpoint selection : max mAP among top-3 min val_loss

### 2단계 방식

- 2-class classifier : EfficientNet, Resnet200d, total 15 models

|model|image size(px)|folds|batch size|init lr|weight decay|val_acc|auc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|b5|600|5 of 5|12|7.5e-5|1.0e-4|0.9601|0.9930|
|b6|528|5 of 5|12|7.5e-5|1.0e-3|0.9553|0.9927|
|resnet200d|600|3 of 5|12|7.5e-5|1.0e-4|0.9541|0.9934|
|b5|456|single|16|1.0e-4|1.0e-4|0.9557|0.9927|
|b5|1024|single|4|2.5e-5|1.0e-4|0.9577|0.9936|

- 14-class detector : EfficientDet, total 18 models

|model|image size(px)|folds|batch size|init lr|weight decay|mAP|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|d3|1024|single|3|3e-4|1e-3|0.4545|
|d4|896|5 of 5|4|4e-4|1e-4|0.4541|
|d4|896|single|4|4e-4|1e-3|0.4606|
|d4|1024|single|3|3e-4|1e-3|0.4545|
|d5|768|5 of 5|4|4e-4|1e-3|0.4472|
|d5|896|4 of 5|3|3e-4|1e-3|0.4522|
|d5|1024|single|2|2e-4|1e-3|0.4462|

### 1단계 방식

- 15-class detector, total 2 models

|model|image size(px)|folds|batch size|init lr|weight decay|mAP|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|d4|896|2 of 5|4|4e-4|1e-3|0.4546|

batch size < 4 에서는 mAP 결과가 좋지 않았습니다. 이미지 크기가 클수록 mAP 값이 좋아지는 경향이 있었으나 그 이상의 훈련을 하지 못하였습니다.

Freeze BatchNorm, accumulate grad batches, GroupNorm 을 시도해보았으나 더 좋은 결과를 얻지는 못하였습니다.

Downconv를 테스트 해보았으나, Calcification와 같이 small object의 AP는 높아지지만 ILD 및 large object의 AP가 줄어들어 전체 mAP는 개선되지 않았습니다.

[[RANZCR CLiP] 11th Place Solution - Utilizing High resolution, Annotations, and Unlabeled data](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226557)

## 이미지 증강

EfficientDet 논문에 있는 방식을 참고하여 resize, scale, crop을 구성하였습니다.

CLAHE, equalize, invertimg, huesaturationvalue, randomgamma, shiftscalerotate는 효과가 없었습니다.

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

## 데이터 사후처리

2단계 방식의  경우 classifier의 적절한 threshold를 찾는 것이 어려웠습니다. normal 케이스가 60~70% 가 되도록 threshold를 결정하여, normal에 해당될 경우 detector의 detection을 모두 제외하였습니다.

1단계 방식의 경우도 유사하게 threshold를 기준으로 normal에 해당될 경우 detection을 제외하였습니다.

## 결과값 블렌딩

2단계 방식과 1단계 방식의 결과를 nms를 이용하여 블렌딩 하였고 최종 결과는 다음과 같습니다.

2단계 방식, 15 classifier + 18 detector
|normal thr|public LB|private LB||
|-|-|-|-|
|0.70|0.219|**0.246**|**final submission**|
|0.65|0.217|0.255|
|0.60|0.215|0.256|

1단계 방식, 2 detector
|normal thr|public LB|private LB|
|-|-|-|
|0.10|0.198|0.245|
|0.00|0.198|0.245|

2단계 방식 + 1단계 방식
|thr|public LB|private LB||
|-|-|-|-|
|0.70-0.30|0.224|**0.253**|**final submission**|
|0.65-0.30|0.222|0.259|

## 효과적이지 않았던 실험들

Freeze BatchNorm

accumulate grad batches

GroupNorm, group per channel = 8

GroupNorm, pretrained backbone with GroupNorm

Downconv

NIH dataset concat classifier

change aspect ratio

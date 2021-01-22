# kaggle-vinbigdata-chest-xray-abnormalities-detection

Code for [Kaggle VinBigData Chest X-ray Abnormalities Detection][kaggle_link].

[kaggle_link]: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection

***

## Overview

***

## Prepare dataset

My solution use 3x downsampled dataset. And you have two options.

- Download original dataset (~192 GB) and create 3x-downsampled dataset.
- Or download kaggle public dataset already made (3.4 GB).

### Option 1. - Download original dataset and create 3x-downsampled dataset

Download Original Competition data [on the Kaggle competition page][kaggle_dataset_link]. And extract zip file.

[kaggle_dataset_link]: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data

```bash
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection
unzip vinbigdata-chest-xray-abnormalities-detection.zip -d dataset
```

Create 3x-downsampled dataset. It will be saved to "dataset-jpg" folder. It takes some hours.

```bash
python prepare_data.py --dataset_dir=dataset
```

### Option 2. - Download kaggle public dataset already made

Alternatively, you can use kaggle public dataset - [VinBigData Competition 3x downsampled jpg][kaggle_downsampled_dataset_link]

[kaggle_downsampled_dataset_link]: https://www.kaggle.com/sunghyunjun/vinbigdata-competition-3x-downsampled-jpg

```bash
kaggle datasets download -d vinbigdata-competition-3x-downsampled-jpg
unzip vinbigdata-competition-3x-downsampled-jpg.zip -d dataset-jpg
```

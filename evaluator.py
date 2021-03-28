import numpy as np
import torch

from effdet.evaluator import Evaluator, TfmEvaluator
import effdet.evaluation.detection_evaluator as tfm_eval
from map_boxes import mean_average_precision_for_boxes


class XrayEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        distributed=False,
        pred_yxyx=False,
        evaluator_cls=tfm_eval.PascalDetectionEvaluator,
        include_nofinding=False,
        downconv=False,
    ):
        super().__init__(distributed=distributed, pred_yxyx=pred_yxyx)
        self.include_nofinding = include_nofinding
        self._evaluator = evaluator_cls(
            categories=self.get_categories(), matching_iou_threshold=0.4
        )
        self._eval_metric_name = self._evaluator._metric_names[0]
        self._dataset = dataset
        self.downconv = downconv

    def reset(self):
        self._evaluator.clear()
        self.img_indices = []
        self.predictions = []

    def evaluate(self):
        for img_idx, img_dets in zip(self.img_indices, self.predictions):
            sample = self._dataset.__getitem__(img_idx)

            if self.downconv:
                sample_bboxes = sample["bboxes"].numpy() / 2
            else:
                sample_bboxes = sample["bboxes"].numpy()

            sample_labels = sample["labels"].numpy()

            gt = {
                # "bbox": sample["bboxes"].numpy(),
                # "cls": sample["labels"].numpy(),
                "bbox": sample_bboxes,
                "cls": sample_labels,
            }
            self._evaluator.add_single_ground_truth_image_info(img_idx, gt)

            bbox = img_dets[:, 0:4] if self.pred_yxyx else img_dets[:, [1, 0, 3, 2]]
            det = dict(bbox=bbox, score=img_dets[:, 4], cls=img_dets[:, 5])
            self._evaluator.add_single_detected_image_info(img_idx, det)

        metrics = self._evaluator.evaluate()
        self.reset()
        return metrics

    def get_categories(self):
        original_categories = [
            {"id": 0, "name": "Aortic enlargement"},
            {"id": 1, "name": "Atelectasis"},
            {"id": 2, "name": "Calcification"},
            {"id": 3, "name": "Cardiomegaly"},
            {"id": 4, "name": "Consolidation"},
            {"id": 5, "name": "ILD"},
            {"id": 6, "name": "Infiltration"},
            {"id": 7, "name": "Lung Opacity"},
            {"id": 8, "name": "Nodule/Mass"},
            {"id": 9, "name": "Other lesion"},
            {"id": 10, "name": "Pleural effusion"},
            {"id": 11, "name": "Pleural thickening"},
            {"id": 12, "name": "Pneumothorax"},
            {"id": 13, "name": "Pulmonary fibrosis"},
        ]

        if self.include_nofinding:
            original_categories.append(
                {"id": 14, "name": "No finding"},
            )

        # change class label 0-index 0~13 to non-zero, 1-index 1~14
        categories = [
            {"id": cat["id"] + 1, "name": cat["name"]} for cat in original_categories
        ]

        return categories


class ZFTurboEvaluator(object):
    def __init__(self, image_size, iou_thr=0.4, include_nofinding=False):
        self.image_size = image_size
        self.iou_thr = iou_thr
        self.include_nofinding = include_nofinding
        self.categories = self.get_categories()
        self.det_ids: List[object] = []
        self.det_list: List[object] = []
        self.ann_boxes: List[object] = []
        self.ann_labels: List[object] = []

    def add_predictions(self, detections, target):
        img_idx = str(target["img_idx"])
        img_idx = target["img_idx"]
        boxes = target["bbox"]
        labels = target["cls"]

        self.det_ids.append(img_idx)
        self.det_list.append(detections)
        self.ann_boxes.append(boxes)
        self.ann_labels.append(labels)

    def reset(self):
        self.det_ids = []
        self.det_list = []
        self.ann_boxes = []
        self.ann_labels = []

    def evaluate(self):
        """
        ZFTurbo's mAP calculation
        https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes

        mean_ap, average_precisions = mean_average_precision_for_boxes(anns, dets)

        anns: numpy arrays of shapes (N, 6)
        ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']

        dets: numpy arrays of shapes (N, 7)
        ['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']

        batch_ids: batch_size x 1
        batch_labels: batch_size x num_boxes x 1
        batch_boxes: batch_size x num_boxes x 4

        batch_detections: batch_size x max_det_per_image x 6
        detections: [x_min, y_min, x_max_, y_max, score, class]

        """
        anns = []
        dets = []

        for batch_ids, batch_labels, batch_boxes in zip(
            self.det_ids, self.ann_labels, self.ann_boxes
        ):
            for i, (labels, boxes) in enumerate(
                zip(batch_labels, batch_boxes), start=0
            ):
                labels = labels.detach().cpu().numpy()
                labels = np.expand_dims(labels, axis=1)
                boxes = boxes.detach().cpu().numpy()
                # yxyx to xxyy
                boxes = boxes[:, [1, 3, 0, 2]]
                boxes /= self.image_size
                row = np.column_stack((labels, boxes))
                for item in row:
                    ids = np.array(str(batch_ids[i]), dtype=np.object_, ndmin=1)
                    ann = np.concatenate((ids, item), axis=0)
                    ann = np.expand_dims(ann, axis=0)
                    anns.append(ann)

        anns = np.concatenate(anns, axis=0)

        for batch_ids, batch_detections in zip(self.det_ids, self.det_list):
            for i, detections in enumerate(batch_detections, start=0):
                detections = detections.detach().cpu().numpy()
                for item in detections:
                    # xyxy to xxyy
                    item = item[[5, 4, 0, 2, 1, 3]]
                    item[2:] /= self.image_size
                    ids = np.array(str(batch_ids[i]), dtype=np.object_, ndmin=1)
                    det = np.concatenate((ids, item), axis=0)
                    det = np.expand_dims(det, axis=0)
                    dets.append(det)

        dets = np.concatenate(dets, axis=0)

        ### average_precisions: (average_precision, num_annotations)
        mean_ap, average_precisions = mean_average_precision_for_boxes(
            anns, dets, iou_threshold=self.iou_thr, verbose=False
        )

        categories = self.get_categories()
        category_index = {}
        for cat in categories:
            category_index[cat["id"]] = cat

        metrics = {"ZFTurbo_mAP@0.4IOU": mean_ap}
        metrics_prefix = f"ZFTurbo_ByCategory/AP@{self.iou_thr}IOU/"

        ### metrics contains average_precision value only
        metrics.update(
            {
                metrics_prefix + category_index[float(key)]["name"]: value[0]
                for key, value in average_precisions.items()
            }
        )
        self.reset()

        return metrics

    def get_categories(self):
        original_categories = [
            {"id": 0, "name": "Aortic enlargement"},
            {"id": 1, "name": "Atelectasis"},
            {"id": 2, "name": "Calcification"},
            {"id": 3, "name": "Cardiomegaly"},
            {"id": 4, "name": "Consolidation"},
            {"id": 5, "name": "ILD"},
            {"id": 6, "name": "Infiltration"},
            {"id": 7, "name": "Lung Opacity"},
            {"id": 8, "name": "Nodule/Mass"},
            {"id": 9, "name": "Other lesion"},
            {"id": 10, "name": "Pleural effusion"},
            {"id": 11, "name": "Pleural thickening"},
            {"id": 12, "name": "Pneumothorax"},
            {"id": 13, "name": "Pulmonary fibrosis"},
        ]

        if self.include_nofinding:
            original_categories.append(
                {"id": 14, "name": "No finding"},
            )

        # change class label 0-index 0~13 to non-zero, 1-index 1~14
        categories = [
            {"id": cat["id"] + 1, "name": cat["name"]} for cat in original_categories
        ]

        return categories
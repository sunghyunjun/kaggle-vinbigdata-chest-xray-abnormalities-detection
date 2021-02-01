from effdet.evaluator import Evaluator, TfmEvaluator
import effdet.evaluation.detection_evaluator as tfm_eval


class XrayEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        distributed=False,
        pred_yxyx=False,
        evaluator_cls=tfm_eval.PascalDetectionEvaluator,
    ):
        super().__init__(distributed=distributed, pred_yxyx=pred_yxyx)
        self._evaluator = evaluator_cls(
            categories=self.get_categories(), matching_iou_threshold=0.4
        )
        self._eval_metric_name = self._evaluator._metric_names[0]
        self._dataset = dataset

    def reset(self):
        self._evaluator.clear()
        self.img_indices = []
        self.predictions = []

    def evaluate(self):
        for img_idx, img_dets in zip(self.img_indices, self.predictions):
            sample = self._dataset.__getitem__(img_idx)
            gt = {
                "bbox": sample["bboxes"].numpy(),
                "cls": sample["labels"].numpy(),
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

        # change class label 0-index 0~13 to non-zero, 1-index 1~14
        categories = [
            {"id": cat["id"] + 1, "name": cat["name"]} for cat in original_categories
        ]

        return categories
import numpy as np
from typing import Dict, Any, List, Tuple
from .base import Evaluator
from core.model_spec import Model_Spec
from core.inference_result import InferenceResult
from decoders.object_detection import DETECTIONS_KEY, RawYoloDetectionDecoder

class ObjectDetectionEvaluator(Evaluator):
    """
    객체 탐지 성능 평가 모듈.
    decoder가 만든 canonical detection 행을 받아 mAP/FPS 지표를 계산합니다.

    스트리밍 평가를 지원합니다.
    add_batch()에서 통과한 박스 좌표(경량)만 누산하고 원본 출력은 즉시 폐기합니다.
    mAP는 모든 배치 완료 후 compute()에서 한 번에 산출합니다.
    """
    def __init__(self, **eval_options):
        self.conf_threshold = eval_options.get("conf_threshold", 0.25)
        self.iou_threshold  = eval_options.get("iou_threshold",  0.45)
        self.image_size     = eval_options.get("image_size",      640)
        self._legacy_decoder = RawYoloDetectionDecoder(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
        self._reset()

    # ------------------------------------------------------------------
    # 내부 상태 초기화
    # ------------------------------------------------------------------

    def _reset(self):
        """누산 상태를 초기화합니다."""
        self._all_preds: List[List[float]] = []  # [img_global_idx, class_id, conf, x1,y1,x2,y2]
        self._all_gts:   List[List[float]] = []  # [img_global_idx, class_id, 1.0, x1,y1,x2,y2]
        self._total_detected: int = 0
        self._total_samples:  int = 0
        self._img_idx_offset: int = 0            # 배치 간 이미지 인덱스 연속성 보장
        self._timing_records: List[float] = []

    # ------------------------------------------------------------------
    # 스트리밍 인터페이스
    # ------------------------------------------------------------------

    def add_batch(self, outputs: Dict[str, np.ndarray], labels: Any, timing_ms: float) -> None:
        """
        배치의 canonical detection 리스트만 누산하고 원본을 폐기합니다.
        배치당 저장량: detection 수 × 7 float — dense 텐서 대비 수십 배 절약.
        """
        flat_labels = self._flatten_labels(labels)

        gts = self._process_ground_truths(flat_labels, self._img_idx_offset)
        batch_preds, detected = self._process_predictions(outputs, self._img_idx_offset)
        sample_count = (
            len(flat_labels)
            if isinstance(flat_labels, list)
            else self._infer_batch_size(outputs)
        )

        self._all_gts.extend(gts)
        self._all_preds.extend(batch_preds)
        self._total_detected += detected
        self._total_samples  += sample_count
        self._img_idx_offset += sample_count
        self._timing_records.append(timing_ms)

    def compute(self) -> Dict[str, Any]:
        """누산된 경량 박스 리스트로 mAP 및 레이턴시 메트릭을 계산합니다."""
        mean_ap = self._calculate_mean_ap(self._all_preds, self._all_gts)
        avg_det = float(self._total_detected / max(1, self._total_samples))

        latency_metrics = self._calculate_latency_metrics(self._timing_records)

        return {
            "mAP@0.5":            mean_ap,
            "Average Detections": avg_det,
            "Total Samples":      self._total_samples,
            **latency_metrics,
        }

    # ------------------------------------------------------------------
    # 배치 호환 인터페이스 (단위 테스트 및 레거시 지원)
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> Dict[str, Any]:
        """추론 결과를 받아 헬퍼 메서드로 각 연산을 위임하여 채점함."""
        self._reset()

        flat_labels = self._flatten_labels(result.labels)

        gts = self._process_ground_truths(flat_labels, 0)
        batch_preds, detected = self._process_predictions(result.outputs, 0)

        self._all_gts.extend(gts)
        self._all_preds.extend(batch_preds)
        self._total_detected = detected
        self._total_samples  = (
            len(flat_labels)
            if isinstance(flat_labels, list)
            else self._infer_batch_size(result.outputs)
        )
        self._timing_records = list(result.timing_records)

        metrics = {"Total Samples": self._total_samples}
        metrics.update(self._calculate_detection_metrics_from_state())
        metrics.update(self._calculate_latency_metrics(self._timing_records))
        return metrics

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _flatten_labels(self, raw_labels: Any) -> List:
        """배치 레이블을 1D 리스트로 평탄화합니다."""
        labels = []
        if isinstance(raw_labels, list):
            for item in raw_labels:
                if isinstance(item, list):
                    labels.extend(item)
                else:
                    labels.append(item)
        else:
            labels = raw_labels
        return labels

    def _split_label_item(self, item: Any) -> Tuple[Any, Dict[str, Any] | None]:
        """Return YOLO label rows plus optional preprocessing context."""
        if isinstance(item, dict) and "label" in item:
            context = item.get("preprocess_context")
            return item["label"], context if isinstance(context, dict) else None
        return item, None

    def _process_ground_truths(self, labels: Any, img_idx_offset: int) -> List[List[float]]:
        """정답지를 파싱하여 [img_global_idx, class_id, 1.0, x1, y1, x2, y2] 형태로 평탄화."""
        all_gts = []
        if labels is None:
            return all_gts

        for local_idx in range(len(labels)):
            label_rows, context = self._split_label_item(labels[local_idx])
            if len(label_rows) == 0:
                continue
            img_gts = np.array(label_rows)
            cls_ids = img_gts[:, 0]
            if context and {
                "original_width",
                "original_height",
                "scale",
                "pad_x",
                "pad_y",
            }.issubset(context):
                orig_w = float(context["original_width"])
                orig_h = float(context["original_height"])
                scale = float(context["scale"])
                pad_x = float(context["pad_x"])
                pad_y = float(context["pad_y"])
                cx, cy, w, h = (
                    img_gts[:, 1] * orig_w,
                    img_gts[:, 2] * orig_h,
                    img_gts[:, 3] * orig_w,
                    img_gts[:, 4] * orig_h,
                )
                x1 = (cx - w / 2) * scale + pad_x
                y1 = (cy - h / 2) * scale + pad_y
                x2 = (cx + w / 2) * scale + pad_x
                y2 = (cy + h / 2) * scale + pad_y
            else:
                size = self.image_size
                cx, cy, w, h = (
                    img_gts[:, 1] * size, img_gts[:, 2] * size,
                    img_gts[:, 3] * size, img_gts[:, 4] * size
                )
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            global_idx = img_idx_offset + local_idx
            for j in range(len(img_gts)):
                all_gts.append([
                    global_idx, cls_ids[j], 1.0,
                    x1[j], y1[j], x2[j], y2[j]
                ])
        return all_gts

    def _process_predictions(
        self, outputs: Dict[str, np.ndarray], img_idx_offset: int
    ) -> Tuple[List[List[float]], int]:
        """Canonical detection rows -> global-indexed prediction rows."""
        if DETECTIONS_KEY not in outputs:
            outputs = self._legacy_decoder.decode(outputs)

        return self._process_canonical_detections(
            outputs[DETECTIONS_KEY],
            img_idx_offset,
        )

    def _process_canonical_detections(
        self, detections: np.ndarray, img_idx_offset: int
    ) -> Tuple[List[List[float]], int]:
        all_preds = []
        det_arr = np.asarray(detections, dtype=np.float32)
        if det_arr.size == 0:
            return all_preds, 0
        if det_arr.ndim == 1 and det_arr.shape[0] == 7:
            det_arr = det_arr.reshape(1, 7)
        if det_arr.ndim != 2 or det_arr.shape[1] != 7:
            raise ValueError(
                "ObjectDetectionEvaluator expects canonical detections with "
                f"shape (N, 7), got {det_arr.shape}"
            )

        finite = np.isfinite(det_arr).all(axis=1)
        positive = (det_arr[:, 2] > 0) & ((det_arr[:, 5] - det_arr[:, 3]) > 0) & (
            (det_arr[:, 6] - det_arr[:, 4]) > 0
        )
        for row in det_arr[finite & positive]:
            local_idx, class_id, conf, x1, y1, x2, y2 = row.tolist()
            all_preds.append(
                [
                    int(img_idx_offset + int(local_idx)),
                    int(class_id),
                    float(conf),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ]
            )
        return all_preds, len(all_preds)

    def _infer_batch_size(self, outputs: Dict[str, np.ndarray]) -> int:
        if DETECTIONS_KEY in outputs:
            det_arr = np.asarray(outputs[DETECTIONS_KEY])
            if det_arr.size == 0:
                return 0
            det_arr = det_arr.reshape(-1, 7)
            return int(np.max(det_arr[:, 0])) + 1
        pred = np.asarray(next(iter(outputs.values())))
        return int(pred.shape[0]) if pred.ndim > 0 else 1

    def _calculate_ap_per_class(self, c_preds: np.ndarray, c_gts: np.ndarray) -> float:
        """단일 클래스의 예측 상자들과 정답 박스 간의 IoU를 측정하여 AP@0.5를 도출합니다."""
        c_preds = c_preds[c_preds[:, 2].argsort()[::-1]]
        nd, nt = len(c_preds), len(c_gts)
        tp, fp = np.zeros(nd), np.zeros(nd)

        gt_matched = {i: np.zeros(len(c_gts[c_gts[:, 0] == i])) for i in np.unique(c_gts[:, 0])}

        for d_idx in range(nd):
            img_idx  = c_preds[d_idx, 0]
            pred_box = c_preds[d_idx, 3:7]

            img_gts = c_gts[c_gts[:, 0] == img_idx]
            if len(img_gts) == 0:
                fp[d_idx] = 1
                continue

            ixmin = np.maximum(img_gts[:, 3], pred_box[0])
            iymin = np.maximum(img_gts[:, 4], pred_box[1])
            ixmax = np.minimum(img_gts[:, 5], pred_box[2])
            iymax = np.minimum(img_gts[:, 6], pred_box[3])
            iw, ih = np.maximum(ixmax - ixmin, 0.), np.maximum(iymax - iymin, 0.)
            inters = iw * ih

            area_gt   = (img_gts[:, 5] - img_gts[:, 3]) * (img_gts[:, 6] - img_gts[:, 4])
            area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            ious = inters / np.maximum(area_gt + area_pred - inters, 1e-6)

            jmax = np.argmax(ious)
            if np.max(ious) >= 0.5 and not gt_matched[img_idx][jmax]:
                tp[d_idx] = 1
                gt_matched[img_idx][jmax] = 1
            else:
                fp[d_idx] = 1

        fpc, tpc = np.cumsum(fp), np.cumsum(tp)
        recalls    = tpc / np.maximum(nt, 1e-6)
        precisions = tpc / np.maximum(fpc + tpc, 1e-6)

        # 표준 VOC AP 계산: sentinel 경계값 추가 후 precision envelope 적용
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        idx_res = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx_res + 1] - mrec[idx_res]) * mpre[idx_res + 1])
        return float(ap)

    def _calculate_mean_ap(
        self, all_preds: List[List[float]], all_gts: List[List[float]]
    ) -> float:
        """모든 예측/정답값을 클래스별로 쪼갠 뒤 AP를 구하고 전체 평균(mAP)을 산출함."""
        if len(all_preds) == 0 or len(all_gts) == 0:
            return 0.0

        pred_arr = np.array(all_preds)
        gt_arr   = np.array(all_gts)
        unique_classes = np.unique(gt_arr[:, 1])
        ap_list = []

        for c in unique_classes:
            c_preds = pred_arr[pred_arr[:, 1] == c]
            c_gts   = gt_arr[gt_arr[:, 1] == c]
            if len(c_preds) == 0:
                ap_list.append(0.0)
                continue
            if len(c_gts) == 0:
                continue
            ap_list.append(self._calculate_ap_per_class(c_preds, c_gts))

        return float(np.mean(ap_list)) if ap_list else 0.0

    def _calculate_detection_metrics_from_state(self) -> Dict[str, float]:
        """누산된 상태에서 mAP 및 평균 탐지 수를 계산합니다."""
        mean_ap = self._calculate_mean_ap(self._all_preds, self._all_gts)
        avg_det = float(self._total_detected / max(1, self._total_samples))
        return {
            "mAP@0.5":            mean_ap,
            "Average Detections": avg_det,
        }

    def _calculate_latency_metrics(self, timing_records: List[float]) -> Dict[str, float]:
        """레이턴시 및 FPS를 계산합니다."""
        if not timing_records:
            return {}

        avg_latency = float(np.mean(timing_records))
        p99_latency = float(np.percentile(timing_records, 99))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        return {
            "Average Latency (ms)": avg_latency,
            "P99 Latency (ms)":     p99_latency,
            "FPS":                  fps
        }

    def is_applicable(self, device_spec: Dict[str, Any], model_spec: Model_Spec) -> bool:
        task_name = str(getattr(model_spec, "task", ""))
        return "OBJECT_DETECTION" in task_name

    def get_metric_names(self) -> List[str]:
        return [
            "mAP@0.5", "Average Detections",
            "Average Latency (ms)", "P99 Latency (ms)", "FPS", "Total Samples"
        ]

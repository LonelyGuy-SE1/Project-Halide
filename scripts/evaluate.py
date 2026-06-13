"""Evaluate defect JSON predictions against Halide annotations.

This script does not run model inference. It evaluates saved predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import TRAINING_JSONL, load_jsonl
from data.schemas import ALLOWED_LABELS, bbox_iou, clean_defects, label_counts


@dataclass
class ClassMetrics:
    label: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    mean_iou: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "precision": round(self.precision, 6),
                "recall": round(self.recall, 6),
                "f1": round(self.f1, 6),
                "mean_iou": round(self.mean_iou, 6),
            }
        )
        return data


def _prediction_map(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        image = str(row.get("image", ""))
        if not image:
            continue
        raw = row.get("predictions", row.get("defects", []))
        cleaned, _ = clean_defects(raw)
        out[image] = cleaned
    return out


def _ground_truth_map(path: str | Path = TRAINING_JSONL) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        image = str(row.get("image", ""))
        cleaned, _ = clean_defects(row.get("annotations", []))
        out[image] = cleaned
    return out


def match_image(
    predictions: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> tuple[dict[str, ClassMetrics], dict[str, int]]:
    metrics = {label: ClassMetrics(label=label) for label in sorted(ALLOWED_LABELS)}
    used_truth: set[int] = set()
    matched_ious: dict[str, list[float]] = {label: [] for label in ALLOWED_LABELS}

    for pred in predictions:
        label = pred.get("label")
        if label not in metrics:
            continue
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(truth):
            if idx in used_truth or gt.get("label") != label:
                continue
            iou = bbox_iou(pred.get("bbox"), gt.get("bbox"))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            metrics[label].true_positive += 1
            used_truth.add(best_idx)
            matched_ious[label].append(best_iou)
        else:
            metrics[label].false_positive += 1

    for idx, gt in enumerate(truth):
        label = gt.get("label")
        if idx not in used_truth and label in metrics:
            metrics[label].false_negative += 1

    for label, ious in matched_ious.items():
        if ious:
            metrics[label].mean_iou = sum(ious) / len(ious)

    return metrics, {
        "prediction_count": len(predictions),
        "truth_count": len(truth),
    }


def evaluate_predictions(
    predictions_path: str | Path,
    *,
    ground_truth_path: str | Path = TRAINING_JSONL,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    preds = _prediction_map(predictions_path)
    truth = _ground_truth_map(ground_truth_path)
    aggregate = {label: ClassMetrics(label=label) for label in sorted(ALLOWED_LABELS)}
    image_reports: dict[str, Any] = {}
    negative_images = 0
    negative_images_with_predictions = 0
    positive_images = 0
    positive_images_with_predictions = 0

    for image, gt_defects in truth.items():
        pred_defects = preds.get(image, [])
        if gt_defects:
            positive_images += 1
            if pred_defects:
                positive_images_with_predictions += 1
        else:
            negative_images += 1
            if pred_defects:
                negative_images_with_predictions += 1
        metrics, counts = match_image(
            pred_defects,
            gt_defects,
            iou_threshold=iou_threshold,
        )
        image_reports[image] = {
            "counts": counts,
            "prediction_labels": label_counts(pred_defects),
            "truth_labels": label_counts(gt_defects),
            "classes": {label: m.to_json() for label, m in metrics.items()},
        }
        for label, metric in metrics.items():
            previous_tp = aggregate[label].true_positive
            aggregate[label].true_positive += metric.true_positive
            aggregate[label].false_positive += metric.false_positive
            aggregate[label].false_negative += metric.false_negative
            if metric.true_positive:
                previous_sum = aggregate[label].mean_iou * previous_tp
                current_sum = metric.mean_iou * metric.true_positive
                aggregate[label].mean_iou = (
                    (previous_sum + current_sum)
                    / aggregate[label].true_positive
                )

    totals = {
        "true_positive": sum(m.true_positive for m in aggregate.values()),
        "false_positive": sum(m.false_positive for m in aggregate.values()),
        "false_negative": sum(m.false_negative for m in aggregate.values()),
    }
    precision_den = totals["true_positive"] + totals["false_positive"]
    recall_den = totals["true_positive"] + totals["false_negative"]
    precision = totals["true_positive"] / precision_den if precision_den else 0.0
    recall = totals["true_positive"] / recall_den if recall_den else 0.0
    f1_den = precision + recall
    negative_rate = (
        negative_images_with_predictions / negative_images if negative_images else 0.0
    )
    positive_rate = (
        positive_images_with_predictions / positive_images if positive_images else 0.0
    )

    return {
        "iou_threshold": iou_threshold,
        "images_evaluated": len(truth),
        "image_level": {
            "negative_images": negative_images,
            "negative_images_with_predictions": negative_images_with_predictions,
            "negative_detection_rate": round(negative_rate, 6),
            "positive_images": positive_images,
            "positive_images_with_predictions": positive_images_with_predictions,
            "positive_detection_rate": round(positive_rate, 6),
        },
        "micro": {
            **totals,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(2 * precision * recall / f1_den, 6) if f1_den else 0.0,
        },
        "classes": {label: metric.to_json() for label, metric in aggregate.items()},
        "images": image_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="JSONL with image and predictions/defects")
    parser.add_argument("--ground-truth", default=str(TRAINING_JSONL))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = evaluate_predictions(
        args.predictions,
        ground_truth_path=args.ground_truth,
        iou_threshold=args.iou,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

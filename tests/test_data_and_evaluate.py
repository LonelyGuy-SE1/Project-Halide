from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from data.augmentation import OverlayDefect, augment_image
from data.datasets import dataset_summary, load_jsonl, validate_entries
from scripts.evaluate import evaluate_predictions
from scripts.generate_v5_negative_curriculum import (
    add_chemical_stain,
    add_crack_network,
    make_example,
    row,
)
from scripts.summarize_training_log import parse_metrics, summarize_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dataset_summary_and_validation(tmp_path) -> None:
    rows = [
        {
            "image": "scan.jpg",
            "source": "unit",
            "annotations": [
                {"label": "dust", "bbox": [0.1, 0.1, 0.2, 0.2]},
                {"label": "bad", "bbox": [0.1, 0.1, 0.2, 0.2]},
            ],
        }
    ]
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, rows)
    loaded = load_jsonl(path)
    assert dataset_summary(loaded)["label_counts"] == {"dust": 1}
    issues = validate_entries(loaded)
    assert issues[0].message == "1 invalid annotations"


def test_evaluate_predictions(tmp_path) -> None:
    gt = tmp_path / "gt.jsonl"
    preds = tmp_path / "preds.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "image": "scan.jpg",
                "annotations": [
                    {"label": "scratch", "bbox": [0.1, 0.1, 0.4, 0.4]},
                    {"label": "dust", "bbox": [0.7, 0.7, 0.8, 0.8]},
                ],
            }
        ],
    )
    _write_jsonl(
        preds,
        [
            {
                "image": "scan.jpg",
                "predictions": [
                    {"label": "scratch", "bbox": [0.1, 0.1, 0.4, 0.4]},
                    {"label": "dust", "bbox": [0.0, 0.0, 0.1, 0.1]},
                ],
            }
        ],
    )
    result = evaluate_predictions(preds, ground_truth_path=gt, iou_threshold=0.5)
    assert result["micro"]["true_positive"] == 1
    assert result["micro"]["false_positive"] == 1
    assert result["micro"]["false_negative"] == 1
    assert result["classes"]["scratch"]["precision"] == 1.0


def test_evaluate_predictions_averages_iou_across_matches(tmp_path) -> None:
    gt = tmp_path / "gt.jsonl"
    preds = tmp_path / "preds.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "image": "scan-a.jpg",
                "annotations": [{"label": "scratch", "bbox": [0.0, 0.0, 0.4, 0.4]}],
            },
            {
                "image": "scan-b.jpg",
                "annotations": [{"label": "scratch", "bbox": [0.0, 0.0, 0.4, 0.4]}],
            },
        ],
    )
    _write_jsonl(
        preds,
        [
            {
                "image": "scan-a.jpg",
                "predictions": [{"label": "scratch", "bbox": [0.0, 0.0, 0.4, 0.4]}],
            },
            {
                "image": "scan-b.jpg",
                "predictions": [{"label": "scratch", "bbox": [0.0, 0.0, 0.2, 0.4]}],
            },
        ],
    )

    result = evaluate_predictions(preds, ground_truth_path=gt, iou_threshold=0.1)

    assert result["classes"]["scratch"]["true_positive"] == 2
    assert result["classes"]["scratch"]["mean_iou"] == 0.75


def test_evaluate_predictions_reports_negative_detection_rate(tmp_path) -> None:
    gt = tmp_path / "gt.jsonl"
    preds = tmp_path / "preds.jsonl"
    _write_jsonl(
        gt,
        [
            {"image": "clean-a.jpg", "annotations": []},
            {"image": "clean-b.jpg", "annotations": []},
            {
                "image": "defect.jpg",
                "annotations": [{"label": "dust", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            },
        ],
    )
    _write_jsonl(
        preds,
        [
            {
                "image": "clean-a.jpg",
                "predictions": [{"label": "dust", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            },
            {"image": "clean-b.jpg", "predictions": []},
            {
                "image": "defect.jpg",
                "predictions": [{"label": "dust", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            },
        ],
    )

    result = evaluate_predictions(preds, ground_truth_path=gt)

    assert result["image_level"]["negative_images"] == 2
    assert result["image_level"]["negative_images_with_predictions"] == 1
    assert result["image_level"]["negative_detection_rate"] == 0.5
    assert result["image_level"]["positive_detection_rate"] == 1.0


def test_augment_image_generates_valid_annotations(tmp_path) -> None:
    overlay_path = tmp_path / "scratch.png"
    overlay = Image.new("RGBA", (10, 4), (0, 0, 0, 0))
    for x in range(10):
        overlay.putpixel((x, 2), (255, 255, 255, 255))
    overlay.save(overlay_path)

    image, annotations = augment_image(
        Image.new("RGB", (100, 100), "black"),
        [OverlayDefect(overlay_path, "scratch")],
        seed=1,
        defects_per_image=(1, 1),
    )
    assert image.size == (100, 100)
    assert annotations[0]["label"] == "scratch"
    assert all(0.0 <= v <= 1.0 for v in annotations[0]["bbox"])


def test_v5_negative_curriculum_generates_clean_and_stain_examples() -> None:
    rng = random.Random(123)
    clean_image, clean_annotations = make_example(0, rng, max_side=240, clean=True)
    assert clean_image.size[0] <= 240
    assert clean_annotations == []

    stained, annotation = add_chemical_stain(
        Image.new("RGB", (240, 160), "gray"),
        random.Random(456),
    )
    assert stained.size == (240, 160)
    assert annotation["label"] == "chemical_stain"
    assert all(0.0 <= value <= 1.0 for value in annotation["bbox"])

    training_row = row(
        "augmented/v5_negative_curriculum/images/example.jpg",
        [annotation],
    )
    assert training_row["images"] == [
        "augmented/v5_negative_curriculum/images/example.jpg"
    ]
    assert "chemical_stain" in training_row["conversations"][1]["value"]


def test_v5_negative_curriculum_generates_crack_focus_examples() -> None:
    image, annotations = make_example(
        1,
        random.Random(789),
        max_side=320,
        clean=False,
        crack_focus=True,
    )
    labels = {annotation["label"] for annotation in annotations}

    assert image.size[0] <= 320
    assert "scratch" in labels
    assert "emulsion_damage" in labels

    cracked, crack_annotations = add_crack_network(
        Image.new("RGB", (320, 240), "gray"),
        random.Random(987),
    )
    assert cracked.size == (320, 240)
    assert any(annotation["label"] == "scratch" for annotation in crack_annotations)
    assert any(annotation["label"] == "emulsion_damage" for annotation in crack_annotations)
    assert all(
        0.0 <= value <= 1.0
        for annotation in crack_annotations
        for value in annotation["bbox"]
    )


def test_training_log_summary() -> None:
    metrics = parse_metrics(
        "{'loss': '1.0', 'epoch': '1'}\n"
        "{'eval_loss': '0.8', 'epoch': '1'}\n"
        "{'eval_loss': '0.7', 'epoch': '2'}\n"
    )
    summary = summarize_metrics(metrics)
    assert summary["train_points"] == 1
    assert summary["eval_points"] == 2
    assert summary["best_eval"]["eval_loss"] == "0.7"

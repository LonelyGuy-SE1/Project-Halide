"""
Defect JSON and annotation schemas.
All bounding boxes are normalized [0.0-1.0] (x_min, y_min, x_max, y_max).
"""
from dataclasses import dataclass, field
from typing import List, Optional


# Defect classes from FilmDamageSimulator
DEFECT_CLASSES = {
    "dust": 0,
    "dirt": 1,
    "scratch": 2,
    "long_hair": 3,
    "short_hair": 4,
    "light_leak": 5,
    "chemical_stain": 6,
    "emulsion_damage": 7,
}

CLASS_NAMES = {v: k for k, v in DEFECT_CLASSES.items()}


@dataclass
class BBox:
    """Normalized bounding box [0.0-1.0]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def validate(self):
        assert 0.0 <= self.x_min <= 1.0, f"x_min {self.x_min} out of range"
        assert 0.0 <= self.y_min <= 1.0, f"y_min {self.y_min} out of range"
        assert 0.0 <= self.x_max <= 1.0, f"x_max {self.x_max} out of range"
        assert 0.0 <= self.y_max <= 1.0, f"y_max {self.y_max} out of range"
        assert self.x_min < self.x_max, "x_min must be < x_max"
        assert self.y_min < self.y_max, "y_min must be < y_max"

    def to_list(self) -> List[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BBox":
        return cls(x_min=coords[0], y_min=coords[1], x_max=coords[2], y_max=coords[3])


@dataclass
class DefectAnnotation:
    """Single defect annotation."""
    label: str
    bbox: BBox
    confidence: float = 1.0

    def to_dict(self):
        return {
            "label": self.label,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DefectAnnotation":
        return cls(
            label=d["label"],
            bbox=BBox.from_list(d["bbox"]),
            confidence=d.get("confidence", 1.0),
        )


@dataclass
class ScanAnnotation:
    """All defects for a single scan."""
    image_path: str
    width: int
    height: int
    source: str  # "film_damage_simulator" or "blueneg"
    defects: List[DefectAnnotation] = field(default_factory=list)

    def to_dict(self):
        return {
            "image": self.image_path,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "annotations": [d.to_dict() for d in self.defects],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanAnnotation":
        return cls(
            image_path=d["image"],
            width=d["width"],
            height=d["height"],
            source=d["source"],
            defects=[DefectAnnotation.from_dict(a) for a in d.get("annotations", [])],
        )


@dataclass
class DiagnosticResult:
    """Reasoning model output."""
    defect_type: str
    severity: str  # mild, moderate, severe
    location: str  # edge, center, distributed
    root_cause: str
    recommended_fix: str

    def to_dict(self):
        return {
            "defect_type": self.defect_type,
            "severity": self.severity,
            "location": self.location,
            "root_cause": self.root_cause,
            "recommended_fix": self.recommended_fix,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


def polygon_to_bbox_norm(points: list, width: int, height: int) -> BBox:
    """Convert pixel polygon points to normalized bounding box."""
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return BBox(
        x_min=min(xs) / width,
        y_min=min(ys) / height,
        x_max=max(xs) / width,
        y_max=max(ys) / height,
    )


def convert_film_damage_annotation(
    json_path: str, image_width: int, image_height: int
) -> ScanAnnotation:
    """Convert FilmDamageSimulator JSON annotation to our format."""
    with open(json_path) as f:
        raw = json.load(f)

    defects = []
    for key, item in raw.items():
        if "points" in item and "label" in item:
            label = item["label"]["name"].lower().replace(" ", "_")
            bbox = polygon_to_bbox_norm(item["points"], image_width, image_height)
            defects.append(DefectAnnotation(label=label, bbox=bbox))

    return ScanAnnotation(
        image_path="",
        width=image_width,
        height=image_height,
        source="film_damage_simulator",
        defects=defects,
    )


import json

"""
Defect class registry for Project Halide.

The 5 ALLOWED_LABELS are the only classes the vision model emits and the
only ones present in the training data. The 3 additional classes
(light_leak, chemical_stain, emulsion_damage) are defined in
FilmDamageSimulator but are not in the current training set; they are
listed in `DEFECT_CLASSES_ALL` for reference but not used at runtime.

Bounding boxes are normalized [0.0-1.0] (x_min, y_min, x_max, y_max).
"""

ALLOWED_LABELS = frozenset({
    "dust",
    "dirt",
    "scratch",
    "long_hair",
    "short_hair",
})

DEFECT_CLASSES_ALL = {
    "dust": 0,
    "dirt": 1,
    "scratch": 2,
    "long_hair": 3,
    "short_hair": 4,
    "light_leak": 5,
    "chemical_stain": 6,
    "emulsion_damage": 7,
}

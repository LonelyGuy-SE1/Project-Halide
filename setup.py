"""Package setup for Project Halide."""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="project-halide",
    version="0.1.0",
    description="Edge-native diagnostic engine for analog film scans.",
    packages=find_packages(
        include=[
            "data",
            "data.*",
            "models",
            "models.*",
            "pipeline",
            "pipeline.*",
            "storage",
            "storage.*",
            "ui",
            "ui.*",
            "scripts",
            "scripts.*",
        ]
    ),
    python_requires=">=3.11",
)

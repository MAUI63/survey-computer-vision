from dataclasses import dataclass
from typing import List

from click import Path


@dataclass
class Annotation:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str


@dataclass
class Detection(Annotation):
    score: float


@dataclass
class InferredImage:
    img_path: Path
    detections: List[Detection]
    annotations: List[Annotation]

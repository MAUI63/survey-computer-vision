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

    def to_dict(self):
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "label": self.label,
        }


@dataclass
class Detection(Annotation):
    score: float

    def to_dict(self):
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "label": self.label,
            "score": self.score,
        }


@dataclass
class InferredImage:
    img_path: Path
    detections: List[Detection]
    annotations: List[Annotation]

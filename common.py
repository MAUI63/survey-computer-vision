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

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def iou(self, other: "Annotation") -> float:
        # Compute intersection over union with another annotation
        ix0 = max(self.x0, other.x0)
        ix1 = min(self.x1, other.x1)
        if ix1 <= ix0:
            return 0.0
        iy0 = max(self.y0, other.y0)
        iy1 = min(self.y1, other.y1)
        if iy1 <= iy0:
            return 0.0
        intersection_area = (ix1 - ix0) * (iy1 - iy0)
        if intersection_area == 0:
            return 0.0
        union = self.area() + other.area() - intersection_area
        if union == 0:
            return 0.0

        return intersection_area / union


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

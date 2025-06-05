# OBSS SAHI Tool
# Code written by AnNT, 2023.

import logging
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
from yolov9.detect import DetectMultiBackend
from yolov9.utils.general import non_max_suppression

from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


class Yolov9DetectionModel(Yolov8DetectionModel):
    def __init__(self, iou_threshold: float = 0.5, *args, **kwargs):
        self.iou_threshold = iou_threshold
        super().__init__(*args, **kwargs)

    def check_dependencies(self) -> None:
        check_requirements(["ultralytics"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        # from yolov9.models.yolo import DetectionModel

        logger.info(f"Loading YOLOv9 model from {self.model_path}")
        self.model = DetectMultiBackend(self.model_path, device=self.device, dnn=False, data=None, fp16=False)
        self.batch_size = self.model.batch_size if hasattr(self.model, "batch_size") else 247
        logger.info(f"Model batch size: {self.batch_size}")
        # self.model.warmup(imgsz=(self.batch_size, 3, 640, 640))
        self.category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}

    def perform_inference(self, images: List[np.ndarray]):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # t0 = time.monotonic()
        if not isinstance(images, list):
            images = [images]

        # Old way:
        im = []
        for img in images:
            im.append(img.transpose((2, 0, 1))[::-1])  # HWC to CHW, BGR to RGB
        im = np.array(im)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)

        # # New way:
        # im = torch.stack(tuple(torch.from_numpy(i).to(self.device) for i in images))
        # im = im[:, :, :, [2, 1, 0]]  # bgr -> rgb
        # im = im.permute(0, 3, 1, 2)  # nhwc -> nchw

        # Commont
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            conf_thres=self.confidence_threshold,
            iou_thres=self.iou_threshold,
            classes=None,
            agnostic=False,
            max_det=100,
        )
        self._original_predictions = [p[p[:, 4] >= self.confidence_threshold] for p in pred]

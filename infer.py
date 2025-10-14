import json
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import LOW_MODEL_CONFIDENCE, get_sliced_prediction
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from yolov9.models.yolo import Model  # Need this here to not break imports elsewhere ... weird

from common import Annotation, Detection, InferredImage


class AnnotatedDataset(Dataset):
    def __init__(self, paths: List[Tuple[Path, Path]], annotations: Optional[List[Any]] = None):
        if annotations is None:
            annotations = [None] * len(paths)
        assert len(paths) == len(annotations)
        self.tasks = list(zip(paths, annotations))

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # print("getting", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        (img_path, vis_path), annotations = self.tasks[idx]
        img = Image.open(str(img_path))
        img.load()
        img = img.convert("RGB")
        return dict(pil_img=img, img_path=img_path, vis_path=vis_path, annotations=annotations)


class ThreadedVisualizer:
    def __init__(
        self,
        title_prefix,
        visual_with_gt_dir,
        output_height=720,
        max_colum_detection_width=200,
        crop_size=100,
        max_detections=20,
    ):
        self.title_prefix = title_prefix
        self.visual_with_gt_dir = visual_with_gt_dir
        self.output_height = output_height
        self.max_column_detection_width = max_colum_detection_width
        self.crop_size = crop_size
        self.max_detections = max_detections
        self._buffer = queue.Queue()
        self._processing = False
        self._t = threading.Thread(target=self._run)
        self._t.daemon = True
        self._t.start()

    def visualize(self, image_as_pil, img_path, out_path, detections, annotations):
        self._buffer.put((image_as_pil, img_path, out_path, detections, annotations))

    def draw_crops(self, crops, sceneh, add_score=False):
        # Sort most important first and get maxn:
        crops = sorted(crops, key=lambda x: -x[1])
        if len(crops) > self.max_detections:
            crops = crops[: self.max_detections]

        # Split into bunchs of crops such that each bunch has a height <= sceneh:
        bunches = []
        current_bunch = []
        for crop, score in crops:
            h, w = crop.shape[:2]
            if h > sceneh:
                logger.info("This is a really tall detection - ignoring")
                continue
            else:
                if sum(i[0].shape[0] for i in current_bunch) + h > sceneh:
                    # New bunch
                    bunches.append(current_bunch)
                    current_bunch = []
                else:
                    current_bunch.append((crop, score))
        if current_bunch:
            bunches.append(current_bunch)
        del current_bunch

        # Now draw each:
        columns = []
        for bunch in bunches:
            column_w = max(i[0].shape[1] for i in bunch)
            column = np.zeros((sceneh, column_w, 3), np.uint8)
            y0 = 0
            for crop, score in bunch:
                h, w = crop.shape[:2]
                # Ok, copy it over
                column_y1 = y0 + h
                assert column_y1 <= sceneh
                column[y0:column_y1, :w, :] = crop
                if add_score:
                    # Make top few pixels the score
                    column[y0 : min(sceneh, y0 + 5), : int(column_w * score), 1] = 255
                y0 = column_y1
            columns.append(column)
        if not columns:
            return None
        return np.hstack(columns)

    def _run(self):
        while True:
            self._processing = False
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 0.5
            fontthickness = 1
            _, fonth = cv2.getTextSize("hectors", font, fontScale=fontscale, thickness=fontthickness)[0]
            try:
                annotations: List[Annotation]
                detections: List[Detection]
                image_as_pil, img_path, out_path, detections, annotations = self._buffer.get()
                self._processing = True
                # resize:
                bgr = cv2.cvtColor(np.array(image_as_pil), cv2.COLOR_RGB2BGR)
                ih, iw = bgr.shape[:2]
                sceneh = self.output_height
                scenew = int(sceneh / ih * iw)
                scene = cv2.resize(bgr, (scenew, sceneh))
                xscale = scenew / iw
                yscale = sceneh / ih

                pred_crops = []
                gt_crops = []

                # Add annotations:
                if annotations is not None:
                    color = (0, 0, 255)
                    for a in annotations:
                        label = a.label
                        # Draw scene:
                        sx0 = int(a.x0 / iw * scenew)
                        sx1 = int(a.x1 / iw * scenew)
                        sy0 = int(a.y0 / ih * sceneh)
                        sy1 = int(a.y1 / ih * sceneh)
                        # Pad it a bit so we can still see the dolphin
                        sx0 = max(0, sx0 - 10)
                        sx1 = min(scenew, sx1 + 10)
                        sy0 = max(0, sy0 - 10)
                        sy1 = min(sceneh, sy1 + 10)
                        cv2.rectangle(scene, (sx0, sy0), (sx1, sy1), color=color, thickness=fontthickness)
                        cv2.putText(
                            scene,
                            label,
                            (sx0, max(0, sy0 - fonth)),
                            font,
                            fontscale,
                            color=color,
                            thickness=fontthickness,
                        )
                        x0 = int(a.x0)
                        x1 = int(a.x1)
                        y0 = int(a.y0)
                        y1 = int(a.y1)  # Get crop:
                        w = x1 - x0
                        if w < self.max_column_detection_width:
                            h = y1 - y0
                            d = int(self.crop_size // 2)
                            nx0 = max(0, x0 - d)
                            nx1 = min(iw, x1 + d)
                            ny0 = max(0, y0 - d)
                            ny1 = min(ih, y1 + d)
                            crop = bgr[ny0:ny1, nx0:nx1, :].copy()
                            gt_crops.append((crop, 1))

                # Label it:
                color = (0, 255, 0)
                o: Detection
                for o in detections:
                    x0, y0, x1, y1 = map(int, (o.x0, o.y0, o.x1, o.y1))
                    # Draw scene:
                    sx0 = int(x0 * xscale)
                    sx1 = int(x1 * xscale)
                    sy0 = int(y0 * yscale)
                    sy1 = int(y1 * yscale)
                    cv2.rectangle(scene, (sx0, sy0), (sx1, sy1), color=color, thickness=1)
                    # Get crop:
                    w = x1 - x0
                    if w < self.max_column_detection_width:
                        h = y1 - y0
                        d = int(self.crop_size // 2)
                        nx0 = max(0, x0 - d)
                        nx1 = min(iw, x1 + d)
                        ny0 = max(0, y0 - d)
                        ny1 = min(ih, y1 + d)
                        crop = bgr[ny0:ny1, nx0:nx1, :].copy()
                        pred_crops.append((crop, o.score))

                # Draw crops:
                left_columns = self.draw_crops(gt_crops, sceneh)
                right_columns = self.draw_crops(pred_crops, sceneh, add_score=True)
                stacks = [i for i in (left_columns, scene, right_columns) if i is not None]
                out = np.hstack(stacks)

                # Pad the top and write some info:
                img_path = Path(img_path)
                msg = f"{self.title_prefix} | {img_path.name}"
                font = cv2.FONT_HERSHEY_PLAIN
                w, h = cv2.getTextSize(msg, font, fontScale=0.5, thickness=1)[0]  # label width, height
                header = np.zeros((h + 10, out.shape[1], 3), np.uint8)
                cv2.putText(header, msg, (5, h + 5), font, 0.5, (255, 255, 255), 1)
                out = np.vstack([header, out])

                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), out)
            except:  # NOQA
                logger.exception("Failed!")

    def empty(self):
        return not self._processing and self._buffer.qsize() == 0


def predict(
    detection_model,
    title_prefix,
    img_paths: List[Tuple[Path, Path]],
    odir,
    annotations: Optional[List[Any]] = None,
    model_confidence_threshold: float = 0.25,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    novisual: bool = False,
    batch_size: int = 1,
    pad_batches: bool = False,
    verbose: int = 0,
) -> Iterator[InferredImage]:
    if model_confidence_threshold < LOW_MODEL_CONFIDENCE and postprocess_type != "NMS":
        logger.warning(
            f"Switching postprocess type/metric to NMS/IOU since confidence threshold is low ({model_confidence_threshold})."
        )
        postprocess_type = "NMS"
        postprocess_match_metric = "IOU"

    odir = Path(odir)
    visual_with_gt_dir = odir / "visuals_with_gt"
    visual_with_gt_dir.mkdir(parents=True, exist_ok=True)  # make dir

    vis = ThreadedVisualizer(title_prefix=title_prefix, visual_with_gt_dir=visual_with_gt_dir)
    # Use a torch dataset/dataloader for nicer multiprocessing etc.
    dataset = AnnotatedDataset(paths=img_paths, annotations=annotations)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=8,
        batch_size=None,
        batch_sampler=None,
        collate_fn=lambda x: x,
        # prefetch_factor=2,
    )

    for task in tqdm(dataloader, "Inferring", total=len(dataloader)):
        # print(task)
        img = task["pil_img"]
        prediction_result = get_sliced_prediction(
            image=img,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
            verbose=verbose,
            batch_size=batch_size,
            pad_batches=pad_batches,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # Format detections nicer:
        detections = []
        for o in object_prediction_list:
            x0, y0, x1, y1 = o.bbox.to_xyxy()
            detections.append(Detection(x0=x0, y0=y0, x1=x1, y1=y1, score=o.score.value, label=o.category.name))

        if not novisual:
            vis.visualize(img, task["img_path"], task["vis_path"], detections, annotations=task["annotations"])

        yield InferredImage(img_path=task["img_path"], detections=detections, annotations=task["annotations"])

    # # export coco results
    # with open(odir / "result.json", "w") as f:
    #     json.dump([asdict(d) for d in detections], f, indent=2)

    # Wait until vis done:
    while not vis.empty():
        print("Waiting to finish vis")
        time.sleep(0.1)


def main(args):
    model_path = Path(args.modeldir)
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov9",
        model_path=str(model_path / "weights" / "best.pt"),
        config_path=args.modelconfig,
        confidence_threshold=args.model_confidence_threshold,
        device="cuda:1",
    )

    title_prefix = f"yolov9 ({model_path.name}) imgs @ {args.imgdir}"

    odir = model_path / "sahiruns" / datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    odir.mkdir()  # Fail if exists as we want a clean slate
    inference_dir = odir / "inferences"
    inference_dir.mkdir()

    # Read coco
    img_dir = Path(args.imgdir)
    assert img_dir.exists()

    img_paths = sorted([str(path) for path in img_dir.iterdir()])

    for inference in predict(
        model,
        title_prefix,
        img_paths,
        odir=odir,
        model_confidence_threshold=args.model_confidence_threshold,
        slice_height=args.slice_height,
        slice_width=args.slice_height,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        batch_size=36,
    ):
        opath = inference_dir / f"{inference.img_path.stem}.json"
        with open(opath, "w") as f:
            json.dump([asdict(d) for d in inference.detections], f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("modeldir")
    parser.add_argument("modelconfig")
    parser.add_argument("imgdir")
    parser.add_argument("slice_height", type=int)
    parser.add_argument("slice_width", type=int)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2)
    parser.add_argument("--model_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)

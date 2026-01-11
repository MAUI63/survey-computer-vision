import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import LOW_MODEL_CONFIDENCE, get_sliced_prediction
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from yolov9.models.yolo import Model  # Need this here to not break imports elsewhere ... weird

from common import Detection, InferredImage
from visualizer import ThreadedVisualizer


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

    vis = ThreadedVisualizer(title_prefix=title_prefix)
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

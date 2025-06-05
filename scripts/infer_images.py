import json
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from sahi.models.yolov9 import Yolov9DetectionModel

from common import Annotation
from infer import predict

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

INFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "inferences"


def main(args):
    print(args)
    model_path = Path(args.modeldir)
    logger.info("Loading model")
    model = Yolov9DetectionModel(
        model_path=str(model_path / "weights" / "best.pt"),
        config_path=args.modelconfig,
        confidence_threshold=args.model_confidence_threshold,
        iou_threshold=args.model_iou_threshold,
        device=args.device,
    )
    logger.info(f"Model loaded with device {model.device}")
    batchsize = model.batch_size
    if args.batch_size is not None:
        logger.info(f"Overriding batch size: {args.batch_size}")
        batchsize = args.batch_size
    logger.info(f"Model batch size: {batchsize}")
    pad_batches = model.model.engine  # If tensorrt, we're using fixed batches, so we need to pad the last batch
    if pad_batches:
        logger.info("Padding batches")
    outdir = (
        INFERENCE_DIR
        / model_path.name
        / f"{args.slice_height}x{args.slice_width}_{args.overlap_height_ratio}x{args.overlap_width_ratio}"
        / f"{args.model_confidence_threshold}_{args.model_iou_threshold}"
        / args.outname
    )
    outdir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output to {outdir}")

    annotations_by_img = {}
    if args.coco:
        with open(args.coco) as f:
            coco = json.load(f)
        classes = {str(cls["id"]): cls["name"] for cls in coco["categories"]}
        img_names = {img["id"]: img["file_name"] for img in coco["images"]}
        for ann in coco["annotations"]:
            filename = img_names[ann["image_id"]]
            if filename not in annotations_by_img:
                annotations_by_img[filename] = []
            annotations_by_img[filename].append(
                Annotation(
                    label=classes[str(ann["category_id"])],
                    x0=ann["bbox"][0],
                    y0=ann["bbox"][1],
                    x1=ann["bbox"][0] + ann["bbox"][2],
                    y1=ann["bbox"][1] + ann["bbox"][3],
                )
            )
    img_paths = []
    annotations = []
    for img_path in sorted(Path(args.imgdir).iterdir()):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", "*.JPG", "*.JPEG", "*.PNG"]:
            if (outdir / "visuals_with_gt" / f"{img_path.stem}.jpg").exists():
                logger.info(f"Skipping {img_path} as already processed")
                continue
            img_paths.append(img_path)
            annotations.append(annotations_by_img.get(img_path.name))

    logger.info(f"Inferring {len(img_paths)}")
    title_prefix = f"yolov9 ({model_path.name}) imgs @ {args.imgdir}"
    inference_dir = outdir / "inferences"
    inference_dir.mkdir(exist_ok=True)
    for inference in predict(
        model,
        title_prefix,
        img_paths,
        annotations=annotations,
        odir=outdir,
        model_confidence_threshold=args.model_confidence_threshold,
        slice_height=args.slice_height,
        slice_width=args.slice_height,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        batch_size=batchsize,
        pad_batches=pad_batches,
        novisual=args.novisual,
        verbose=args.verbose,
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
    parser.add_argument("outname")
    parser.add_argument("--coco", default=None)
    parser.add_argument("--slice_height", type=int, default=640)
    parser.add_argument("--slice_width", type=int, default=640)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2)
    parser.add_argument("--model_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--model_iou_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--novisual", action="store_true")
    args = parser.parse_args()

    main(args)

import concurrent.futures
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import _fixpath
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from loguru import logger
from PIL import Image, ImageDraw
from tqdm import tqdm

import constants
from common import Annotation
from visualizer import ThreadedVisualizer

sns.set_theme(style="ticks")


@dataclass
class CropTargets:
    img: Dict
    cx: float
    cy: float
    annotations: List[Dict]
    annotation: Optional[Dict]


def get_crop_targets(datadir, split, target_labels, n_positive, n_negative, headn=None):
    with open(datadir / f"{split}.json") as f:
        coco = json.load(f)
    targets = []

    if headn is not None:
        n_positive = min(n_positive, headn)
        n_negative = min(n_negative, headn)

    # Add the dolphins:
    imgs = {i["id"]: i for i in coco["images"]}
    anns_per_img = {}
    for a in coco["annotations"]:
        anns_per_img.setdefault(a["image_id"], [])
        anns_per_img[a["image_id"]].append(a)

    dolphin_categories = [c["id"] for c in coco["categories"] if c["name"] in target_labels]
    assert dolphin_categories
    while len(targets) < n_positive:
        n = len(targets)
        for a in coco["annotations"]:
            if a["category_id"] in dolphin_categories:
                b = a["bbox"]
                cx = b[0] + b[2] / 2
                cy = b[1] + b[3] / 2
                targets.append(
                    CropTargets(
                        img=imgs[a["image_id"]], cx=cx, cy=cy, annotation=a, annotations=anns_per_img[a["image_id"]]
                    )
                )
                if len(targets) >= n_positive:
                    break
        if len(targets) == n:
            raise RuntimeError("No positive examples! Something's wrong - bailing")

    # And fake some negatives
    for _ in range(n_negative):
        img = random.choice(list(imgs.values()))
        targets.append(
            CropTargets(
                img,
                cx=random.random(),
                cy=random.random(),
                annotation=None,
                annotations=anns_per_img.get(img["id"], []),
            )
        )
    return targets, coco["categories"]


def get_crop(
    img_idx: int,
    source_name: str,
    imgdir: Path,
    target: CropTargets,
    croph: int,
    cropw: int,
    gsd_crop_scale: Optional[float],
    expected_dolphin_diagonal_pixels: Optional[float],
    input_crop_scaler: Callable,
    annotation_overlap_ratio_to_keep: float,
    out_imgdir: Path,
):
    imgpath = imgdir / target.img["file_name"]
    imgh = target.img["height"]
    imgw = target.img["width"]
    assert imgpath.exists()
    # Figure out how much to resize by
    if gsd_crop_scale is not None:
        input_crop_scale = gsd_crop_scale
    else:
        if target.annotation is None:
            # OK, a negative example - let's just use scale of 1:
            input_crop_scale = 1
        else:
            # OK, we've got a an annotation and not GSD info - we'll use the dolphin size:
            b = target.annotation["bbox"]
            dolphin_size_diagonal_pixels = (b[2] ** 2 + b[3] ** 2) ** 0.5
            # OK, if our dolphin is too small, then we want to zoom in, i.e. we want the crop to be smaller. E.g. if
            # we expect a dolphin to be 100px and it's only 50px, then we want to zoom in by 2x, which means we want
            # to *decrease* our crop size by 2x (so that when we scale it up to the target size, it's 2x bigger)
            input_crop_scale = dolphin_size_diagonal_pixels / expected_dolphin_diagonal_pixels

    # OK, cool. Now apply our zoom scaler (which can add some randomness, and also a maximum on the scale)
    input_crop_scale = input_crop_scaler(input_crop_scale)

    input_crop_w = int(cropw * input_crop_scale)
    input_crop_h = int(croph * input_crop_scale)
    # If it's larger than the image, cap it:
    if input_crop_w > imgw or input_crop_h > imgh:
        logger.warning(f"Crop too large at {input_crop_w}x{input_crop_h} - capping to image size ({imgw}x{imgh})")
        downscale = max(input_crop_w / imgw, input_crop_h / imgh)
        input_crop_w = int(input_crop_w / downscale)
        input_crop_h = int(input_crop_w / downscale)

    # Done! Now we can get our crop. We've got crop centers, but we want to offset them so this is somewhere
    # withing the crop, as opposed to always in the middle.
    cx = random.randint(int(target.cx - input_crop_w / 2), int(target.cx + input_crop_w / 2))
    cy = random.randint(int(target.cy - input_crop_h / 2), int(target.cy + input_crop_h / 2))

    x0 = max(0, min(imgw - input_crop_w, int(cx - input_crop_w / 2)))
    y0 = max(0, min(imgh - input_crop_h, int(cy - input_crop_h / 2)))
    x1 = x0 + input_crop_w
    y1 = y0 + input_crop_h
    assert x1 - x0 == input_crop_w
    assert y1 - y0 == input_crop_h
    assert x0 >= 0 and x1 <= imgw
    assert y0 >= 0 and y1 <= imgh

    img = Image.open(imgpath)
    crop = img.crop((x0, y0, x1, y1))

    # Now resize it:
    logger.info(
        f"Used crop scale of {input_crop_scale:0.2f} to get crop of size {crop.size} - resizing ({cropw}, {croph})"
    )
    crop = crop.resize((cropw, croph))

    # Redo the coco stuff
    crop_name = f"{source_name}_{img_idx:06d}_{target.img['file_name']}_{x0}_{y0}_{x1}_{y1}.jpg"
    new_img = dict(
        id=img_idx,
        height=crop.height,
        width=crop.width,
        file_name=crop_name,
        src=dict(img=target.img, annotation=target.annotation),
    )

    # Get annotations for this:
    this_annotations = []
    for a in target.annotations:
        b = a["bbox"]
        ax0 = b[0]
        ax1 = ax0 + b[2]
        ay0 = b[1]
        ay1 = ay0 + b[3]
        # Get the intersection area:
        ix0 = max(x0, ax0)
        ix1 = min(x1, ax1)
        dx = ix1 - ix0
        if dx < 0:
            # No overlap
            continue
        iy0 = max(y0, ay0)
        iy1 = min(y1, ay1)
        dy = iy1 - iy0
        if dy < 0:
            # No overlap
            continue
        overlap_area = dx * dy
        overlap_ratio = overlap_area / (ax1 - ax0) / (ay1 - ay0)
        if overlap_ratio < annotation_overlap_ratio_to_keep:
            continue
        # OK, this annotation is in the crop - let's get is position in crop coordinates
        cx0 = max(0, ax0 - x0)
        cx1 = min(x1 - x0, max(0, ax1 - x0))
        cy0 = max(0, ay0 - y0)
        cy1 = min(y1 - y0, max(0, ay1 - y0))
        cx0 = cx0 / (x1 - x0) * cropw
        cx1 = cx1 / (x1 - x0) * cropw
        cy0 = cy0 / (y1 - y0) * croph
        cy1 = cy1 / (y1 - y0) * croph
        assert 0 <= cx0 < cx1 <= cropw
        assert 0 <= cy0 < cy1 <= croph
        new_a = dict(
            id=None,
            image_id=img_idx,
            bbox=(cx0, cy0, cx1 - cx0, cy1 - cy0),
            category_id=a["category_id"],
        )
        this_annotations.append(new_a)

    # Save the crop:
    out_imgdir.mkdir(parents=True, exist_ok=True)
    crop.save(out_imgdir / crop_name, quality=95)

    return new_img, this_annotations


def get_crops(
    source_name: str,
    imgdir: Path,
    targets: List[CropTargets],
    croph: int,
    cropw: int,
    src_gsd: float,
    target_gsd: float,
    expected_dolphin_diagonal_cm: float,
    input_crop_scaler: Callable,
    annotation_overlap_ratio_to_keep: float,
    out_imgdir: Path,
    num_workers: int,
):
    # Say we want crops of 100x100 and the src_gsd is 1cm/pixel while the target_gsd is 5cm/pixel. In this case, that
    # means our source has more resolution than we want, so we need to zoom it out by 5x. In that case, what we do is
    # take a crop 5x the size (which we'll then resize down by 5x)
    logger.info(f"Getting crops for {len(targets)} targets")
    gsd_crop_scale = None
    if src_gsd is not None:
        gsd_crop_scale = target_gsd / src_gsd
        logger.info(
            f"Source GSD={src_gsd}cm/pixel, target GSD={target_gsd}cm/pixel so will scale crops by {gsd_crop_scale} before resizing to {croph}x{cropw}"
        )
    expected_dolphin_diagonal_pixels = expected_dolphin_diagonal_cm / target_gsd
    logger.info(
        f"Expected dolphin diagonal in pixels: {expected_dolphin_diagonal_pixels} ({expected_dolphin_diagonal_cm}cm @ target GSD={target_gsd}cm/pixel)"
    )
    new_imgs = []
    new_annotations = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, target in enumerate(targets):
            futures.append(
                executor.submit(
                    get_crop,
                    img_idx=idx,
                    source_name=source_name,
                    imgdir=imgdir,
                    target=target,
                    croph=croph,
                    cropw=cropw,
                    gsd_crop_scale=gsd_crop_scale,
                    expected_dolphin_diagonal_pixels=expected_dolphin_diagonal_pixels,
                    input_crop_scaler=input_crop_scaler,
                    annotation_overlap_ratio_to_keep=annotation_overlap_ratio_to_keep,
                    out_imgdir=out_imgdir,
                )
            )
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc=f"getting crops for {source_name}"
        ):
            img, anns = future.result()
            new_imgs.append(img)
            for ann in anns:
                ann["id"] = len(new_annotations)
                new_annotations.append(ann)

    return new_imgs, new_annotations


def get_input_crop_scaler(spec):
    mn = spec["min"]
    mx = spec["max"]

    def f(scale):
        scale = min(mx, max(mn, scale))
        # Triangular about the scale i.e. it's more likely to be close to the scale than further away, but can still
        # range from mn to mx
        return random.triangular(mn, mx, scale)

    return f


def process_source(odir, name, spec, source, split, headn, num_workers):
    logger.info(f"Processing {source}")

    # Figure out the split name to use:
    split_name = split
    if split == "train":
        if "train_name" in source:
            split_name = source["train_name"]
    elif split == "test":
        if "test_name" in source:
            split_name = source["test_name"]
    else:
        raise RuntimeError(f"Unknown split {split}")

    logger.info(f"Using split name '{split_name}' for split '{split}'")

    # Resize
    fpath = constants.DATA_DIR / "ml" / "annotated" / source["source"]
    with open(fpath / "meta.json") as f:
        meta = json.load(f)

    size_spec = spec["size"]
    crop_spec = spec["crop"]
    # Figure out where we're aiming to take crops from
    n_pos, n_neg = source["n_positive"], source["n_negative"]
    if split == "test":
        n_pos = int(n_pos * spec["test_ratio"])
        n_neg = int(n_neg * spec["test_ratio"])
    crop_targets, categories = get_crop_targets(
        fpath, split_name, source["target_labels"], n_positive=n_pos, n_negative=n_neg, headn=headn
    )

    # Now get the crops (including handling sizes etc.)
    imgs, anns = get_crops(
        source_name=name,
        imgdir=fpath / split_name,
        targets=crop_targets,
        croph=crop_spec["h"],
        cropw=crop_spec["w"],
        src_gsd=meta["gsd"],
        target_gsd=size_spec["target_gsd"],
        expected_dolphin_diagonal_cm=size_spec["expected_dolphin_diagonal_cm"],
        input_crop_scaler=get_input_crop_scaler(crop_spec["scale_variation"]),
        annotation_overlap_ratio_to_keep=crop_spec["min_area_ratio"],
        out_imgdir=odir / split,
        num_workers=num_workers,
    )

    # And rename labels. If the rename is null, we remove the annotation. NB, this is why we do this after all the
    # filtering - sometimes we want to crop to e.g. false positives, which happens above, but here they get
    # removed:
    if "rename" not in source:
        new_categories = categories
    else:
        new_categories = []
        for c in categories:
            new_name = source["rename"][c["name"]]
            logger.info(f"Renaming category {c['name']} to {new_name}")
            if new_name is None:
                # Remove all these annotations:
                logger.info(f"Renamed to `None`, removing all annotations of category {c['name']}")
                anns = [d for d in anns if d["category_id"] != c["id"]]
            else:
                new_categories.append(dict(id=c["id"], name=new_name))
    return dict(images=imgs, annotations=anns, categories=new_categories)


def merge_cocos(cocos):
    images = []
    annotations = []
    categories = {}
    for coco in cocos:
        image_id_remaps = {}
        for d in coco["images"]:
            old_id = d["id"]
            new_id = len(images)
            d["id"] = new_id
            image_id_remaps[old_id] = new_id
            images.append(d)
        cats = coco["categories"]
        category_remaps = {}
        for c in cats:
            if c["name"] not in categories:
                categories[c["name"]] = len(categories)
            category_remaps[c["id"]] = categories[c["name"]]
        for a in coco["annotations"]:
            a["id"] = len(annotations)
            a["image_id"] = image_id_remaps[a["image_id"]]
            a["category_id"] = category_remaps[a["category_id"]]
            annotations.append(a)

    # Done
    return dict(
        images=images,
        annotations=annotations,
        categories=[dict(id=id, name=name) for name, id in categories.items()],
    )


def to_yolo(coco, img_dir: Path, oimgdir: Path, olabeldir: Path):
    logger.info(f"Making yolo from {img_dir}")
    for d in (oimgdir, olabeldir):
        if d.exists():
            logger.info(f"Cleaning out {d}/*")
            for fpath in d.iterdir():
                os.remove(fpath)
            os.rmdir(d)
        d.mkdir(parents=True)
    anns_by_img_id = {}
    for a in coco["annotations"]:
        id = a["image_id"]
        anns_by_img_id.setdefault(id, [])
        anns_by_img_id[id].append(a)
    for img in tqdm(coco["images"], desc="making yolo"):
        name = img["file_name"]
        os.link(img_dir / name, oimgdir / name)
        imgh, imgw = img["height"], img["width"]
        with open(olabeldir / f"{'.'.join(name.split('.')[:-1])}.txt", "w") as f:
            for a in anns_by_img_id.get(img["id"], []):
                x, y, w, h = a["bbox"]
                cx = (x + w / 2) / imgw
                cy = (y + h / 2) / imgh
                w /= imgw
                h /= imgh
                assert 0 <= cx <= 1
                assert 0 <= cy <= 1
                assert 0 <= w <= 1
                assert 0 <= h <= 1
                category = a["category_id"]
                f.write(f"{category} {cx} {cy} {w} {h}\n")


def profile_box_sizes(coco, cropw, croph, opath):
    logger.info("Profiling box sizes")
    widths = []
    heights = []
    for a in coco["annotations"]:
        b = a["bbox"]
        widths.append(b[2])
        heights.append(b[3])
    # import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
    f, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.kdeplot(
        x=widths,
        y=heights,
        cmap=cmap,
        clip=((0, cropw), (0, croph)),
        fill=True,
        cut=10,
        thresh=0,
        levels=15,
    )
    plt.xlim(0, cropw)
    plt.ylim(0, croph)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Dolphin Bounding Box Sizes")
    plt.savefig(opath)


def profile_box_locations(coco, cropw, croph, opath):
    logger.info("Profiling box locations")
    x = []
    y = []
    for a in coco["annotations"]:
        b = a["bbox"]
        x.append(b[0] + b[2] / 2)
        y.append(b[1] + b[3] / 2)
    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
    f, axes = plt.subplots(1, 1, figsize=(10, 10))
    sns.kdeplot(
        x=x,
        y=y,
        cmap=cmap,
        fill=True,
        clip=((0, cropw), (0, croph)),
        cut=10,
        thresh=0,
        levels=15,
    )
    plt.xlim(0, cropw)
    plt.ylim(0, croph)
    plt.xlabel("cx")
    plt.ylabel("cy")
    plt.title("Dolphin Bounding Box Center Positions")
    plt.savefig(opath)


def main(args):
    # Load spec
    with open(args.dataspec) as f:
        spec = yaml.safe_load(f)
    logger.info(f"Spec: {spec}")

    odir = Path(args.dataspec).parent
    review_dir = odir / "review"
    cocodir = odir / "coco"
    yolodir = odir / "yolo"
    for d in (review_dir, cocodir, yolodir):
        logger.info(f"Cleaning out {d}")
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        visualizers = [
            ThreadedVisualizer(title_prefix=f"Dataset {odir.name} [{split}]") for i in range(args.num_visualizers)
        ]
        cocos = []
        for idx, source in enumerate(spec["sources"]):
            name = source["name"] if "name" in source else source["source"]
            name = f"{idx}_{name}"
            cocos.append(process_source(cocodir, name, spec, source, split, args.headn, args.num_workers))
        # OK, merge things
        coco = merge_cocos(cocos)
        with open(cocodir / f"{split}.json", "w") as f:
            json.dump(coco, f, indent=2)

        # Now to yolo
        to_yolo(coco, cocodir / split, yolodir / "images" / split, yolodir / "labels" / split)

        # Now save it:
        cropw = spec["crop"]["w"]
        croph = spec["crop"]["h"]
        profile_box_locations(coco, cropw, croph, odir / f"box_centers_{split}.png")
        profile_box_sizes(coco, cropw, croph, odir / f"box_sizes_{split}.png")

        # Now process review images
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        for img_idx, img in enumerate(tqdm(coco["images"], desc="reviewing images", total=len(coco["images"]))):
            imgpath = cocodir / split / img["file_name"]
            # def visualize(self, image_as_pil, img_path, out_path, detections, annotations):
            coco_annotations = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
            annotations = []
            for a in coco_annotations:
                x, y, w, h = a["bbox"]
                category_id = a["category_id"]
                annotations.append(Annotation(x0=x, y0=y, x1=x + w, y1=y + h, label=categories[category_id]))
            visualizers[img_idx % args.num_visualizers].visualize(
                Image.open(imgpath), imgpath, review_dir / split / f"{imgpath.stem}.jpg", [], annotations
            )

        # Wait until they're done:
        for idx, v in enumerate(visualizers):
            logger.info(f"Waiting for visualizer {idx} to finish...")
            v.wait_until_done()

        logger.info(f"Finished processing split {split}!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataspec")
    parser.add_argument("--headn", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--num-visualizers", type=int, default=4)
    args = parser.parse_args()

    logger.add(Path(args.dataspec).parent / "log.txt", level="INFO", mode="w")
    main(args)

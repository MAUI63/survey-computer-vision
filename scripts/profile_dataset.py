import json
from pathlib import Path

import _fixpath
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")


def profile_box_sizes(coco, opath):
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
    sns.kdeplot(
        x=widths,
        y=heights,
        cmap=cmap,
        fill=True,
        cut=10,
        thresh=0,
        levels=15,
    )
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Dolphin Bounding Box Sizes")
    plt.savefig(opath)


def profile_box_locations(coco, opath):
    x = []
    y = []
    for a in coco["annotations"]:
        b = a["bbox"]
        x.append(b[0] + b[2] / 2)
        y.append(b[1] + b[3] / 2)
    sns.set_theme(style="white")
    cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
    fig = plt.figure(figsize=(10, 10))
    sns.kdeplot(
        x=x,
        y=y,
        cmap=cmap,
        fill=True,
        cut=10,
        thresh=0,
        levels=15,
    )
    print(x)
    plt.xlabel("cx")
    plt.ylabel("cy")
    plt.title("Dolphin Bounding Box Center Positions")
    plt.savefig(opath)


def main(args):
    odir = Path(args.dataspec).parent
    cocodir = odir / "coco"
    for split in ("train",):  # "test"):
        with open(cocodir / f"{split}.json") as f:
            coco = json.load(f)
        # Now save it:
        profile_box_locations(coco, odir / f"box_centers_{split}.png")
        profile_box_sizes(coco, odir / f"box_sizes_{split}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataspec")
    args = parser.parse_args()
    main(args)

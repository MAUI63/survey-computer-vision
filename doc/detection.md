
## Introduction

We're trying to detect maui's from aerial photos with the following constraints:

- Large images - generally 61mp
- Small dolphins - often 50px or so. They're also quite tricky and often submerged etc.
- Aerial pictures from a drone or plane, looking straight down. These often include footage of the take-off/landing, and the surf, and then all the non-dolphin things in the sea (rubbish, dead fish, birds, long line torpedos, other dolphins, etc.).
- Variable ground surface distance (GSD) - depends on the survey and camera used.
- This does not need to run live on the drone - we can throw whatever we've got at it.
- Very little (real-world) training data. When we started out we only had 7 images from a survey of a Maui, and about 50 of hectors (which are effectively identical).

With that in mind, the computer vision approach has been to:

- Throw the kitchen sink at it to give the best chance of success - we don't care about runtime. We've tried yolov9 and RT-DETR, and both performed similarly, but we stuck with yolov9 as it's more standard.
- Use SAHI to tile the images into smaller chunks, and infer on them. Why? A lot of object detection models perform poorly when detecting small objects. You can fine tune them so that you can downscale the image and fire it through once, but this takes a bit of work, and limits you to the (fine tuned) model, and presumably impacts accuracy. Instead, let's use full resolution, just tiled. The dolphins are "large" in these images, so we can use any model we want.
- Try to standardise the GSD of the images. We could have a model that works at different GSDs (and this does to a degree), but based on experience it's easier if we can normalise all the data to the same GSD before training/inference.

## Libraries

- ./lib/sahi - I used commit 0077a143abdabad2ac81eb375377c744662f7578 from https://github.com/obss/sahi and modified sahi/auto_model.py, sahi/models/rtdetr.py, and sahi/models/yolov9.py to add RT-DETR/yolov9 support, as per this repo. I just added it as files, and deleted .git as submodules are annoying.
- ./lib/yolov9 - I used commit 5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff of https://github.com/WongKinYiu/yolov9 and changed utils/general.py, and models/detect/yolov9-c.maui.yaml as per this repo. Again, deleted .git.

## Running stuff

### Image slicing

You can do it like so
```sh
python3 scripts/slice_dataset.py <path-to-dataset>/meta.yaml
```

e.g.

```
python scripts/slice_dataset.py data/ml/training-datasets/20240815-maui/meta.yaml --num-workers=16
```

Check out an example meta.yaml for all the options. It'll generate the images in coco/yolo formats in the dataset folder, and add review images.

### Training YoloV9

```
# Run below (installing what you need as it complains)
python train_dual.py --workers 8 --device 0 --batch 16 --data /coco/dataset.yolov9.yml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.dolphins.yaml --min-items 0 --epochs 50 --close-mosaic 15

# Test:
PYTHONPATH=lib python test_model_yolov9.py lib/yolov9/runs/train/yolov9-c11/ lib/yolov9/models/detect/yolov9-c.yaml data/dolphins/dolphins.json data/dolphins/images 640 640
```

## Findings

### ML

- The hectors 2021 data seemed to make the data worse. I'm dubious about the annotations too. So ignore it.
- I did a bunch on scaling the datasets to the same size. Interestingly, it seems important to keep full size images in so we can work more zoomed in. That is, keep the original mauis dataset all fullsize, even though these dolphins will be much larger than they should be on tiling. NB: this shouldn't be the case though, which makes me wonder if there's a bunch somewhere.
- RT-DETR and YoloV9 performed similarly. Yolov9 was a bit faster, and more of a standard object detection model, so stuck with that.

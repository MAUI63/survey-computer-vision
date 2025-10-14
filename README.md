# Maui63 computer vision

## TODO

- remove orientation flag for any data we keep (e.g. for ground truth) - some tools (e.g. annotation ones) load the image rotated, so we don't want to worry if the annotations are rotated too.
- copy exif over to detection visuals
- have script to profile ISO
- find the minimum confidence thta got all our dolphins. use that as our threshold for filtering in future.

## Survey findings

- TODO: summarise findings Greg. F5.6 is good for removing vignette around the edges. 1/2500 and above should be fine (and are) to get crips images.
- When it's overcast and cloudy, ISO is way too high (at F5.6 and 1/3200 or even 1/2500 and exposure compensation -1). I think we need to move to F4 or something.
  - Not sure how this impacts spotting Maui - haven't seen any. Which is probbly 'cos they're not there (it's up north in all these dark/grainy images). But could be that they are and the model misses them. Should probably fly over where we know mauis are, on a dark grainy day, and see what we see.

does rgb / bgr matter

## Getting it running

Open this in vscode and follow the prompt to open in dev container.

## Introduction

We're trying to detect maui's from aerial photos with the following constraints:

- Large images - generally 60mp
- Small dolphins - often 50px or so. They're also quite tricky and often submerged etc.
- Aerial pictures from a drone or plane, looking straight down. These often include footage of the take-off/landing, and the surf, and then all the non-dolphin things in the sea (rubbish, dead fish, birds, long line torpedos, other dolphins, etc.).
- Variable ground surface distance (GSD) - depends on the survey and camera used.
- This does not need to run live on the drone - we can throw whatever we've got at it.
- Very little (real-world) training data. Currently we've only got 7 images from a survey of a maui, and about 50 of hectors (which are effectively identical).

With that in mind, this repo has focused on:

- Using a big chonky yolov9 - we don't care about runtime.
- Using SAHI to tile the images into smaller chunks, and inferring on them. Why? A lot of object detection models perform poorly when detecting small objects. You can fine tune them (as was done with the original yolov4) so that you can downscale the image and fire it through once, but this takes a bit of work, and limits you to the model, and presumably accuracy. Instead, let's use full resolution (just tiled). The dolphins are "large" in these images, so we can use any model we want.
- A fair bit of work tidying/standardising the images. We could have a model that works at different altitudes etc. - but based on Tane's experience (and common sense) it's easier if we can normalise all the data to be as similar as possible first.

## Code structure

- Notebooks in ./notebooks are generally for data prep stuff.
- ./lib/sahi - I used commit 0077a143abdabad2ac81eb375377c744662f7578 from <https://github.com/obss/sahi> and modified sahi/auto_model.py, sahi/models/rtdetr.py, and sahi/models/yolov9.py to add RT-DETR/yolov9 support, as per this repo. I just added it as files, and deleted .git as submodules are annoying.
- ./lib/yolov9 - I used commit 5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff of <https://github.com/WongKinYiu/yolov9> and changed utils/general.py, and models/detect/yolov9-c.maui.yaml as per this repo. Again, deleted .git.

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

Check out an example meta.yaml for all the options. It'll a generate the images in coco/yolo formats in the dataset folder, and add review images.

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

### Flights

- The photo timestamps are weird for CW25. E.g. check out Flight 2 of 20230513 - I took DSC01106 to be there first, but the photo timestamps don't get sequential until around DSC1136.

### Yolo v4 tiny (the old model)

- OK, it looks like the lastest code I have is for scaling to 100m
- I found wednesday_data_backup_20220219_object_detector.zip but I think it's for birds etc.
- yolo is unpleasant to compile etc. I tried installing locally and docker, and various combinations. Headless and not wanting to mess with the server system made it a bit annoying.
- yolo python gives worse results than the compiled script, so I've got to use that. Annoying.
- The yolov4 tiny isn't great = (

### RT-DETR

- Need to manually set the LR or it eventually gets nans.
- Get a good recall with threshold of 0.25 or so, and not too many false positives. IOU threshold of 0.1 gives very good recall.

## Misc

### Getting set up on nectar

- ubuntu-drivers install​ didn't seem to work.
- From the ppa graphics-drivers it does. Not sure if 550 is important or not.
  - sudo apt-get install nvidia-driver-550-server​
  - sudo apt-get install nvidia-container-toolkit​
  - sudo systemctl restart docker​
  - sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
​- I've had to disable mig in the past - sudo nvidia-smi -pm 1 && sudo nvidia-smi -mig 0​
- Reboots etc. in between.

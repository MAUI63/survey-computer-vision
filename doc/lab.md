
## Lab notes

- 2026-01-12 - ran our new model on the test data:
  ```
  # This model:
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20251024_150305-yolov9-c-dataset-20251017-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml ./data/ml/annotated/202508-dolphins/all 202508-dolphins-all --device='cuda' --model_confidence_threshold=0.01 --batch_size=32 --coco ./data/ml/annotated/202508-dolphins/all.json

  # Old models:
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240912_052208-yolov9-s-20240815-maui lib/yolov9/models/detect/yolov9-c.maui.yaml ./data/ml/annotated/202508-dolphins/all 202508-dolphins-all --device='cuda' --model_confidence_threshold=0.01 --batch_size=32 --coco ./data/ml/annotated/202508-dolphins/all.json

  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui lib/yolov9/models/detect/yolov9-c.maui.yaml ./data/ml/annotated/202508-dolphins/all 202508-dolphins-all --device='cuda' --model_confidence_threshold=0.01 --batch_size=32 --coco ./data/ml/annotated/202508-dolphins/all.json
  ```
  Conclusion: generally looks pretty good, much fewer false positives, missed a few tricky ones the old model got at Cloudy Bay.
  -

- 2025-10-24 - fixed below and created dataset. Triggered a train:
  ```
  cp /mnt/mauidata/ml/training-datasets/20251017-maui /mnt/tmp/
  # Edit /mnt/tmp/20251017-maui/dataset.yml to point to tmp dir
  nohup docker run --rm --runtime=nvidia --gpus=all --shm-size=80gb --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/cv:/workspaces/cv -v /mnt/mauidata:/workspaces/cv/data -v /mnt/tmp:/workspaces/cv/.tmp -w /workspaces/cv/lib/yolov9 --user=1001 cv python train_dual.py --workers 16 --batch 16 --device 0 --data /workspaces/cv/.tmp/20251017-maui/dataset.yml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --project /workspaces/cv/data/ml/trains/ --name $(date +%Y%m%d_%H%M%S)-yolov9-c-dataset-20251017-maui --hyp hyps/hyp.dolphins.yaml --min-items 0 --epochs 50 --close-mosaic 15
  ```
- 2025-10-17 - attempted to created corresponding dataset with all the latest maui and more false positives:
  ```
  python scripts/slice_dataset.py data/ml/training-datasets/20251017-maui/meta.yaml --num-workers=16
  ```
- 2025-10-15 - created 202508-dolphins dataset. Tidied up code and merged etc.
- 2025/08 - ran all SD cards from winter survey as below:

  ```
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml /media/ko/6CE7-5780/DCIM/  20240822-r09 --device='cuda' --model_confidence_threshold=0.01 --batch_size=32
  ```
- 2025/04/24: running full inference
  ```
  nohup docker run --rm --runtime=nvidia --gpus=all --shm-size=80gb --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/cv:/workspaces/cv -v /mnt/mauidata:/workspaces/cv/data -e PYTHONPATH=".:lib:lib/sahi" -w /workspaces/cv --user=1001 cv python scripts/infer_surveys.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml data/surveys/action-aviation-multicamera/20250331_west_coast_4camera_test/ data/inferences/ 640 640 --device='cuda' --model_confidence_threshold=0.01

  nohup docker run --rm --runtime=nvidia --gpus=all --shm-size=80gb --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/cv:/workspaces/cv -v /mnt/mauidata:/workspaces/cv/data -e PYTHONPATH=".:lib:lib/sahi" -w /workspaces/cv --user=1001 cv python scripts/infer_surveys.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml data/surveys/action-aviation-multicamera/20250405_west_coast_4camera data/inferences/ 640 640 --device='cuda' --model_confidence_threshold=0.01 &
  ```
- Sometime: fixing up the second flight. See notebook `look_for_multicamera_frame_skips.ipynb`

- 2025/04/08: second run-through:
  ```
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml downloads/L09/DCIM/10250405/ 20240405-l09 --device='cuda' --model_confidence_threshold=0.01 --batch_size=32 &&  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml downloads/L28/DCIM/10250405/ 20240405-l28 --device='cuda' --model_confidence_threshold=0.01 --batch_size=32

  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml downloads/L28/DCIM/10150405/ 20240405-l28 --device='cuda' --model_confidence_threshold=0.01 --batch_size=32

  ```
- 2025/04/02: infer on latest 4cam survey
  ```
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml data/surveys/action-aviation/20250331_west_coast_4camera_test/r09/10050331/ r09-10050331 --device='cuda' --model_confidence_threshold=0.1
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.maui.yaml data/surveys/action-aviation/20250331_west_coast_4camera_test/r09/10150331/ r09-10150331 --device='cuda' --model_confidence_threshold=0.1
  ```

- 2024/09/19: backup: `tar -czvf .tmp/$(date +%Y%m%d)-maui-computer-vision.tar.gz --exclude=.ignore --exclude=.tmp --exclude=data/surveys --exclude=data/inferences --exclude='*.old*' --exclude='*.pyc' .`
- 2024/09/13: Smaller has similar mAP of 0.93, as expected. Test the model:
  ```
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240912_052208-yolov9-s-20240815-maui lib/yolov9/models/detect/yolov9-s.maui.yaml data/ml/annotated/202409-dolphins/all/ 202409-dolphins-all --coco=data/ml/annotated/202409-dolphins/all.json --device='cuda' --model_confidence_threshold=0.1
  ```
  OK, it's looking pretty good. Misses a few the big model gets, and gets a few the big model misses. Looking into inference time though, and it didn't help much. Need to dig into SAHI as that seems to be the cause - the model runs in <0.5s but all the other stuff (??) takes another 3-4s.

  Export to TensorRT 'cos why not.
  ```
  python export.py --data=/workspaces/cv/data/ml/training-datasets/20240815-maui/dataset.yml --weights=/workspaces/cv/data/ml/trains/20240912_052208-yolov9-s-20240815-maui/weights/best.pt --img-size 640 640 --device=0 --verbose --batch-size=32 --include=engine
  ```
  Then rename the best.engine to best.bs32.engine so we can keep track of different exported models. Then run like so:
    ```
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240912_052208-yolov9-s-20240815-maui/weights/best.bs32 lib/yolov9/models/detect/yolov9-s.maui.yaml data/ml/annotated/202409-dolphins/all/ 202409-dolphins-all --coco=data/ml/annotated/202409-dolphins/all.json --device='cuda' --model_confidence_threshold=0.1
  ```
  Unfortunately it doesn't work = ( Like, it runs, but the model fails to detect anything. Not sure why. Park it for now.

  Speed up a lot by getting rid of image copying in sahi PredictionResult. Bunch of performance testing. Anyway, now 1-1.5s per image instead of 4-5s.

- 2024/09/12: finished reviewing and comparing. Filtered to the 13 new images with dolphins, labelled. Train a smaller one just to see how it does. Had to update yolov9 lib.
  ```
  python train_dual.py --workers 16 --batch 16 --device 0 --data /workspaces/cv/.tmp/20240815-maui/dataset.yml --img 640 --cfg models/detect/yolov9-s.maui.yaml --weights '' --project /workspaces/cv/data/ml/trains/ --name $(date +%Y%m%d_%H%M%S)-yolov9-s-20240815-maui --hyp hyps/hyp.dolphins.yaml --min-items 0 --epochs 50 --close-mosaic 15
  ```
- 2024/09/11: inference finished, started filtering
- 2024/08/29: inference died this morning, restarting with command below. Got through to start of 2024, with 2k detections so far (confidence > 0.1). So a few more false positives ... hopefully more dolphins!
- 2024/08/22: model seemed to train fine. Testings:
  ```
  PYTHONPATH=".:lib:lib/sahi" python scripts/infer_images.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.yaml data/ml/annotated/202408-dolphins/test/ 202408-dolphins-test --coco=data/ml/annotated/202408-dolphins/test.json --device='cuda' --model_confidence_threshold=0.1
  ```
  Model looking pretty good on the test set. Let's kick off another run through:
  ```
  nohup docker run --rm --runtime=nvidia --gpus=all --shm-size=80gb --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/cv:/workspaces/cv -v /mnt/mauidata:/workspaces/cv/data -e PYTHONPATH=".:lib:lib/sahi" -w /workspaces/cv --user=1001 cv python scripts/infer_surveys.py data/ml/trains/20240815_060150-yolov9-c-20240815-maui/ lib/yolov9/models/detect/yolov9-c.yaml data/surveys/ data/inferences 640 640  --model_confidence_threshold=0.1 &
  ```
  Had to do this first though since the new VM has user=1001 and it made permissions annoying.
  ```
  docker build . --build-arg USER=1001 --tag=cv
  ```

- 2024/08/15 - creating a new dataset. Made the code more efficient and didn't use SAHI for the creation. Adding in all the mauis/hectors we've seen from surveys too, and playing with size. Then I'll kick off a train. Training:
  ```
  python train_dual.py --workers 16 --batch 16 --device 0 --data /workspaces/cv/.tmp/20240815-maui/dataset.yml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --project /workspaces/cv/data/ml/trains/ --name $(date +%Y%m%d_%H%M%S)-yolov9-c-20240815-maui --hyp hyps/hyp.dolphins.yaml --min-items 0 --epochs 50 --close-mosaic 15
  ```
- 2024/08/14 - bunch of messing around with the VMs so the Uni don't break them accidentally again. Seems good. Also thumbnailed all the images and removed 350gb of images with no ocean (i.e. drone taking off or landing).
- 2024/08/01 - grabbed all of the dolphins we know about. This was mostly exporting the annotations from cloudy bay to use in LabelStudio, then annotating, then merging all. Copying data to the new volume.
- Training example:
  ```
  python scripts/slice_dataset.py data/training_datasets/20240420_maui_hectors_false_positives

  cd lib/yolov9
  python train_dual.py --workers 8 --batch 16 --device 0 --data /workspaces/cv/data/training_datasets/20240420_maui_hectors_false_positives/dataset.yml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --project /workspaces/cv/trains/ --name $(date +%Y%m%d_%H%M%S)-yolov9-c --hyp hyps/hyp.dolphins.yaml --min-items 0 --epochs 50 --close-mosaic 15

  python train_dual.py --workers 16 --batch 16 --device 0 --data /workspaces/cv/data/training_datasets/20240429_maui_hectors_false_positives/dataset.yml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --project /workspaces/cv/trains/ --name $(date +%Y%m%d_%H%M%S)-yolov9-c --hyp hyps/hyp.dolphins.yaml --min-items 0 --epochs 50 --close-mosaic 15
  ```
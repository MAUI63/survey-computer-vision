import queue
import threading
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger

from common import Annotation, Detection


class ThreadedVisualizer:
    def __init__(
        self,
        title_prefix,
        output_height=720,
        max_colum_detection_width=200,
        crop_size=100,
        max_detections=20,
    ):
        self.title_prefix = title_prefix
        self.output_height = output_height
        self.max_column_detection_width = max_colum_detection_width
        self.crop_size = crop_size
        self.max_detections = max_detections
        self._buffer = queue.Queue()
        self._processing = False
        self._stopped = False
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
        while not self._stopped:
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

    def size(self):
        return self._buffer.qsize()

    def stop(self, timeout=5):
        self._stopped = True
        self._t.join(timeout=timeout)

    def wait_until_done(self, poll_interval=0.1):
        while not self.empty():
            time.sleep(poll_interval)

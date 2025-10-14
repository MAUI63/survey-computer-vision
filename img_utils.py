import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import exif
import numpy as np
from PIL import Image

SEG_PREFIX = b"\xff"
APP1_SUFFIX = b"\xe1"


def get_exif_lazy(imgpath, chunksize=16000) -> exif.Image:
    # Don't load the whole image into memory - this is slow. Just find the first bit of the image that has the exif data
    all_bites = b""
    with open(imgpath, "rb") as f:
        while True:
            bites = f.read(chunksize)
            if not bites:
                raise RuntimeError("No EXIF data found")
            all_bites += bites
            exifimg = exif.Image(all_bites)
            if exifimg.has_exif:
                # print(len(all_bites))
                break
    if not exifimg.has_exif:
        raise ValueError(f"Image {imgpath} has no EXIF data")
    return exifimg


def get_a7rv_image_timestamp(imgpath, chunksize=16000) -> datetime:
    """
    Get's the time from the EXIF. At this stage, we just parse the raw string without the timezone info, because we
    don't know the offset yet - offset_time_original is empty. (Though that may change depending on firmware and
    camera settings?)
    """
    exifimg = get_exif_lazy(imgpath, chunksize=chunksize)
    t = datetime.strptime(exifimg["datetime_original"], "%Y:%m:%d %H:%M:%S")
    milliseconds = exifimg["subsec_time_original"]
    return t + timedelta(milliseconds=int(milliseconds))


def is_exif_oriented(imgpath, chunksize=16000) -> bool:
    exifimg = get_exif_lazy(imgpath, chunksize=chunksize)
    return exifimg.orientation != 1


def remove_exif_orientation_if_needed(src, target, validate=True):
    assert src != target, "Source and target must be different files"
    assert src.suffix == target.suffix, "Source and target must have the same suffix"

    if not is_exif_oriented(src):
        # Just copy:
        with open(src, "rb") as fsrc, open(target, "wb") as ftgt:
            ftgt.write(fsrc.read())
        return

    img_bytes = None
    if validate:
        with Image.open(src) as im:
            img_bytes = np.array(im).tobytes()

    # Actually edit the exif:
    with open(src, "rb") as f:
        image = exif.Image(f)
    assert image.orientation != 1, "Image is already correctly oriented"
    image.orientation = 1  # Reset to constant:
    with tempfile.NamedTemporaryFile(suffix=target.suffix) as tmp:
        tmp_path = Path(tmp.name)
        with open(tmp_path, "wb") as fo:
            fo.write(image.get_file())
        if validate:
            with Image.open(tmp_path) as im:
                new_img_bytes = np.array(im).tobytes()
            if img_bytes != new_img_bytes:
                raise RuntimeError("Image data changed when removing EXIF orientation!")

        # Write to target:
        tmp_path.rename(target)

from datetime import datetime, timedelta

import exif
from plum.bigendian import uint16

SEG_PREFIX = b"\xff"
APP1_SUFFIX = b"\xe1"


def get_exif_lazy(imgpath, chunksize=1024) -> exif.Image:
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


def get_a7rv_image_timestamp(imgpath, chunksize=1024):
    """
    Get's the time from the EXIF. At this stage, we just parse the raw string without the timezone info, because we
    don't know the offset yet - offset_time_original is empty. (Though that may change depending on firmware and
    camera settings?)
    """
    exifimg = get_exif_lazy(imgpath, chunksize=chunksize)
    t = datetime.strptime(exifimg["datetime_original"], "%Y:%m:%d %H:%M:%S")
    milliseconds = exifimg["subsec_time_original"]
    return t + timedelta(milliseconds=int(milliseconds))

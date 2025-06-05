import exif

SEG_PREFIX = b"\xff"
APP1_SUFFIX = b"\xe1"


def get_exif_lazy(img_path, chunksize=1024):
    # Don't load the whole image into memory - this is slow. Just find the first bit of the image that has the exif data
    all_bites = b""
    with open(img_path, "rb") as f:
        while True:
            bites = f.read(chunksize)
            if not bites:
                raise RuntimeError("No EXIF data found")
            all_bites += bites
            exif_img = exif.Image(all_bites)
            if exif_img.has_exif:
                # print(len(all_bites))
                break
    if not exif_img.has_exif:
        raise ValueError(f"Image {img_path} has no EXIF data")
    return exif_img

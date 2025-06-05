import csv
import imghdr
import json
from pathlib import Path
from typing import Iterator, List

import arrow
import exif
from loguru import logger

from drone.drone import DroneFlight, LocatedPhoto, Location


class CW25DroneFlight(DroneFlight):
    def __init__(self, root):
        super().__init__()
        self._root = Path(root)
        self._photo_dir = self._root / "photos"
        self._log_path = self._get_log_path()

    def _get_log_path(self):
        log_dir = self._root / "logs"
        if not log_dir.exists():
            return None
        logs = list(log_dir.rglob("GPS_Data.txt"))
        if len(logs) == 0:
            logger.warning("Not GPS_Data.txt found!")
            return None
        elif len(logs) > 1:
            logger.warning("More than one GPS_Data.txt found!")
            return None
        log_path = logs[0]
        logger.info(f"Log path: {log_path}")
        return log_path

    def has_photos(self) -> bool:
        return self._photo_dir.exists()

    def has_location_data(self) -> bool:
        return self._log_path is not None

    def _get_photo_paths(self) -> Iterator[Path]:
        if not self._photo_dir.exists():
            logger.warning("No photo dir!")
            return
        yield from sorted([i for i in self._photo_dir.iterdir() if not i.is_dir() and i.name.endswith(".JPG")])

    def _get_located_photos(self) -> Iterator[LocatedPhoto]:
        meta_path = self._root / "meta.json"
        if not meta_path.exists():
            logger.warning("No meta.json! Won't find times/locations")
            for idx, fpath in enumerate(sorted(self._get_photo_paths())):
                yield LocatedPhoto(idx=idx, path=fpath, location=None, time=None)
            return
        else:
            with open(self._root / "meta.json") as f:
                meta = json.load(f)
            first_photo = meta["first-airborne-photo"]
            first_photo_name = first_photo["name"]
            first_photo_time = arrow.get(first_photo["time"])
            time_offset = None
            locations = list(self._get_locations())
            img_paths = list(sorted(self._get_photo_paths()))
            logger.info(f"#locations={len(locations)}, #imgs={len(img_paths)}")
            for idx, fpath in enumerate(img_paths):
                if fpath.name < first_photo_name:
                    continue
                imgtype = imghdr.what(fpath)
                if imgtype not in ("jpeg", "png"):
                    logger.warning(f"Unsupported imgtype {imgtype} for {fpath} - ignoring")
                    continue
                with open(fpath, "rb") as f:
                    exifimg = exif.Image(f)
                t = exifimg.get("datetime")
                if t is None:
                    if time_offset is None:
                        raise RuntimeError("First photo has no exif!")
                    logger.warning("No EXIF time metadata")
                    continue
                if not (exifimg.get("datetime_original") == exifimg.get("datetime_digitized") == t):
                    raise AssertionError("I don't know what all these times are ... let's assume they're the same")
                t = arrow.get(t, "YYYY:MM:DD HH:mm:ss")
                if time_offset is None:
                    time_offset = first_photo_time - t
                t += time_offset
                location = locations[idx] if idx < len(locations) else None
                yield LocatedPhoto(idx=idx, path=fpath, time=t, location=location)

    def _get_locations(self) -> Iterator[Location]:
        """
        Date	Time	Height	Lat	Lng	WindSpped
        01.06.1980	08.00.00	0m	0	0	0
        01.06.1980	08.00.00	0m	0	0	0
        ...
        10.23.2023	06.22.30	8.242m	-37.2871	174.656	14.2279
        10.23.2023	06.22.30	8.241m	-37.2871	174.656	14.2279
        10.23.2023	06.22.30	8.241m	-37.2871	174.656	14.1281
        """
        if self._log_path is None:
            logger.info("No location data!")
            return

        with open(self._log_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # date = datetime.strptime("%m.%d.%Y", row["Date"])
                # time = datetime.strptime("%H.%M.%S", row["Time"])
                t = arrow.get(row["Date"] + row["Time"], "MM.DD.YYYYHH.mm.ss")
                # TODO: timezone
                if t.year < 2020:
                    continue
                # Timestamps are for china, so convert to NZ
                t = t.replace(tzinfo="Asia/Shanghai").to("Pacific/Auckland")
                yield Location(lat=float(row["Lat"]), lon=float(row["Lng"]), altitude=float(row["Height"][:-1]), time=t)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    flight = CW25DroneFlight(args.root)
    for loc in flight._get_locations():
        print(loc)
        break
    for photo in flight._get_photos_with_exif():
        print(photo)

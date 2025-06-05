import concurrent.futures
import imghdr
import io
import struct
import sys
from pathlib import Path
from typing import Iterator

import arrow
import exif
from loguru import logger
from pymavlink import mavutil

from drone.drone import DroneFlight, LocatedPhoto, Location
from img_utils import get_exif_lazy


class RocDroneFlight(DroneFlight):
    def __init__(self, root):
        super().__init__()
        self._root = Path(root)
        self._logpath = self._get_log_file()
        self._photo_dir = self._root / "photos"

    def has_location_data(self) -> bool:
        return self._logpath is not None

    def has_photos(self) -> bool:
        return self._photo_dir.exists()

    def _get_photo_paths(self) -> Iterator[Path]:
        photo_dir = self._root / "photos"
        assert photo_dir.exists()
        yield from sorted([i for i in photo_dir.iterdir() if not i.is_dir() and i.name.endswith(".JPG")])

    def _get_located_photos(self, max_header_position_bytes=1024 * 1024) -> Iterator[LocatedPhoto]:
        locations = list(self._get_locations())

        def do(fidx, fpath):
            imgtype = imghdr.what(fpath)
            if imgtype not in ("jpeg", "png"):
                logger.warning(f"Unsupported imgtype {imgtype} for {fpath} - ignoring")
                return None
            exifimg = get_exif_lazy(fpath, chunksize=100000)
            if exifimg is None:
                logger.warning(f"Couldn't find exif for {fpath}")
                return None
            t = exifimg.get("datetime")
            if t is None:
                logger.warning("No EXIF time metadata")
                return None
            if not (exifimg.get("datetime_original") == exifimg.get("datetime_digitized") == t):
                raise AssertionError("I don't know what all these times are ... let's assume they're the same")
            t = arrow.get(t, "YYYY:MM:DD HH:mm:ss")
            location = None
            if fidx < len(locations):
                location = locations[fidx]
            else:
                logger.warning(f"Location missing for {fpath.name}")
            return LocatedPhoto(idx=fidx, path=fpath, time=t, location=location)

        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
            tasks = [ex.submit(do, i, fpath) for i, fpath in enumerate(sorted(self._get_photo_paths()))]
            for res in concurrent.futures.as_completed(tasks):
                if res is not None:
                    yield res.result()

    def _get_log_file(self):
        logs = [fpath for fpath in (self._root / "logs").iterdir() if fpath.name.endswith(".BIN")]
        if len(logs) == 0:
            logger.info("No log file found!")
            return None
        elif len(logs) > 1:
            logger.info("Multiple log files found!")
            return None
        else:
            log_path = logs[0]
            logger.info(f"Log path: {log_path}")
            return log_path

    def _get_locations(self) -> Iterator[Location]:
        # OK, there's some bad code it seems here
        # https://github.com/ArduPilot/pymavlink/blob/5d0496a3ad654cd47f294b5baba027e25bfd15c7/DFReader.py#L1130
        # I.e. a whole of of prints to stdout. This is really slow. Let's catch it.
        # from IPython.utils.io import capture_output

        # with capture_output():
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            mlog = mavutil.mavlink_connection(str(self._logpath))
            while True:
                m = mlog.recv_match(type=["GPS_RAW", "GPS_RAW_INT", "GPS", "GPS2"])
                if m is None:
                    break
                if m.get_type() == "GPS_RAW_INT":
                    lat = m.lat / 1.0e7
                    lon = m.lon / 1.0e7
                    alt = m.alt / 1.0e3
                    fix = m.fix_type
                elif m.get_type() == "GPS_RAW":
                    lat = m.lat
                    lon = m.lon
                    alt = m.alt
                    fix = m.fix_type
                elif m.get_type() == "GPS" or m.get_type() == "GPS2":
                    lat = m.Lat
                    lon = m.Lng
                    alt = m.Alt
                    fix = m.Status
                else:
                    pass
                if fix >= 2 and lat != 0 and lon != 0:  # i.e. a fix
                    yield Location(lat=lat, lon=lon, altitude=alt, time=arrow.get(m._timestamp))
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        stdout = stdout_buffer.getvalue()
        if stdout:
            logger.info(f"mavlink stdout (first 1000 chars): {stdout[:1000]}")
        stderr = stderr_buffer.getvalue()
        if stderr:
            logger.info(f"mavlink stderr (first 1000 chars): {stderr[:1000]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    flight = RocDroneFlight(args.root)
    # for loc in flight._get_locations():
    #     print(loc)
    #     break
    for photo in flight._get_photos_with_exif():
        print(photo)

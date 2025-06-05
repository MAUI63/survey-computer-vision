import csv
from pathlib import Path
from typing import Iterator

import arrow
from loguru import logger

from drone.drone import DroneFlight, LocatedPhoto, Location


class ActionAviationFlight(DroneFlight):
    def __init__(self, root):
        super().__init__()
        self._root = Path(root)
        self._photo_dir = self._root / "photos"
        self._log_path = self._root / "FRAME_CENTRES.txt"

    def has_photos(self) -> bool:
        return self._photo_dir.exists()

    def has_location_data(self) -> bool:
        return self._log_path.exists()

    def _get_photo_paths(self) -> Iterator[Path]:
        if not self._photo_dir.exists():
            logger.warning("No photo dir!")
            return
        yield from sorted(self._photo_dir.rglob("*.JPG"))

    def _get_located_photos(self) -> Iterator[LocatedPhoto]:
        if not self._log_path.exists():
            logger.warning("No frame centers! Won't find times/locations")
            for idx, fpath in enumerate(sorted(self._get_photo_paths())):
                yield LocatedPhoto(idx=idx, path=fpath)
            return
        else:
            img_lookup = {i.image_name: i for i in self._get_locations()}
            for idx, fpath in enumerate(sorted(self._get_photo_paths())):
                location = img_lookup.get(fpath.name)
                if location is None:
                    logger.warning(f"Frame centres missing {fpath.name}")
                yield LocatedPhoto(idx=idx, path=fpath, time=location.time if location else None, location=location)

    def _get_locations(self) -> Iterator[Location]:
        """
        NAME	LATITUDE	LONGITUDE	HEIGHT	TRACK	DATE	TIMESTAMP
        DSC08471.JPG	-37.31117333	174.6684417	442.199	157.16	2/04/2024	22.10.08.566
        DSC08472.JPG	-37.311825	174.66879	441.699	156.47	2/04/2024	22.10.10.345
        DSC08473.JPG	-37.31251333	174.6691683	440.899	156.54	2/04/2024	22.10.12.145
        DSC08474.JPG	-37.31320833	174.6695367	440.199	157.36	2/04/2024	22.10.13.925
        """
        if self._log_path is None:
            logger.info("No location data!")
            return
        with open(self._log_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                filename_key = reader.fieldnames[0]
                yield Location(
                    lat=float(row["LATITUDE"]),
                    lon=float(row["LONGITUDE"]),
                    altitude=float(row["HEIGHT"]),
                    time=arrow.get(row["DATE"] + " " + row["TIMESTAMP"], "D/MM/YYYY HH.mm.ss.SSS", tzinfo="UTC"),
                    image_name=row[filename_key],
                )

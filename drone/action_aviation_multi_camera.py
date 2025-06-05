import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import arrow

from drone.drone import DroneFlight, LocatedPhoto, Location


@dataclass
class FrameInfo:
    camera: str
    idx: int
    lat: float
    lon: float
    altitude: float


class ActionAviationMultiCameraFlight(DroneFlight):
    def __init__(self, root):
        super().__init__()
        self._root = Path(root)
        self._located_photos: List[LocatedPhoto] = self._get_photos(self._root / "processed_frames.json")

    def has_photos(self) -> bool:
        return True

    def has_location_data(self) -> bool:
        return True

    @staticmethod
    def _get_photos(meta_path: Path) -> List[LocatedPhoto]:
        # Load meta:
        with open(meta_path) as f:
            meta = json.load(f)

        frames = meta["frames"]

        idx = 0
        all_photos = []
        for frame in frames:
            location = Location(
                lat=frame["lat"],
                lon=frame["lon"],
                altitude=frame["alt"],
            )
            for camera, img in frame["images"].items():
                if img is None:
                    continue
                path = meta_path.parent / img["path"]
                assert path.exists()
                located_photo = LocatedPhoto(
                    idx=idx, path=path, time=arrow.get(img["t"]), camera_name=camera, location=location
                )
                all_photos.append(located_photo)
                idx += 1

        return all_photos

    def _get_photo_paths(self) -> Iterator[Path]:
        for photo in self._located_photos:
            yield photo.path

    def _get_located_photos(self) -> Iterator[LocatedPhoto]:
        for photo in self._located_photos:
            yield photo

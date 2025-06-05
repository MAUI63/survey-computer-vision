import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import arrow
from loguru import logger


@dataclass
class Location:
    lat: float
    lon: float
    altitude: float
    time: Optional[arrow.arrow.Arrow] = None
    image_name: Optional[str] = None


@dataclass
class LocatedPhoto:
    idx: int
    path: Path
    time: Optional[arrow.arrow.Arrow] = None
    location: Optional[Location] = None
    camera_name: Optional[str] = None


class DroneFlight(metaclass=abc.ABCMeta):
    def get_photo_paths(self) -> Iterator[Path]:
        if not self.has_photos():
            logger.warning("No photos!")
            return []
        return self._get_photo_paths()

    def get_located_photos(self) -> Iterator[LocatedPhoto]:
        if not self.has_photos() or not self.has_location_data():
            logger.warning("No photos!")
            return []
        return self._get_located_photos()

    @abc.abstractmethod
    def _get_photo_paths(self) -> Iterator[Path]:
        pass

    @abc.abstractmethod
    def _get_located_photos(self) -> Iterator[LocatedPhoto]:
        pass

    @abc.abstractmethod
    def has_location_data(self) -> bool:
        pass

    @abc.abstractmethod
    def has_photos(self) -> bool:
        pass

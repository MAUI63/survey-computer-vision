import json
from pathlib import Path

import requests
from PIL import Image


class API:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def get(self, path: str):
        url = f"{self.base_url}/{path}"
        headers = {"x-api-key": self.api_key}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, data: dict, files=None):
        url = f"{self.base_url}/{path}"
        if files is None:
            headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=data)
        else:
            headers = {"x-api-key": self.api_key}
            response = requests.post(url, headers=headers, data=data, files=files)
        try:
            response.raise_for_status()
        except:
            print(response.text)
            raise
        return response.json()


def main(args):
    api = API(args.url, args.api_key)

    species = api.get("animal-species")
    maui = [i for i in species["animal_species"] if i["name"] == "Maui or Hectors"][0]

    with open(args.coco) as f:
        coco = json.load(f)

    for image in coco["images"]:
        print("-" * 80)
        print("Processing image:", image["file_name"])
        print(image)
        annotations = [i for i in coco["annotations"] if i["image_id"] == image["id"]]
        assert len(annotations) > 0

        meta = image["meta"]
        lat = None
        lon = None
        if meta["location"]:
            lat = meta["location"]["latitude"]
            lon = meta["location"]["longitude"]
        t = meta["capture_time"]
        if t is None:
            t = "2020-01-01T00:00:00Z"
        is_plane = "action" in meta["survey_type"]
        location_accuracy = "PLANE-SURVEY" if is_plane else "DRONE-SURVEY"
        device = "PLANE-CAMERA" if is_plane else "DRONE-CAMERA"
        sighting = dict(
            timestamp=t,
            state="NEW",
            animal_count=len(annotations),
            latitude=lat,
            longitude=lon,
            location_accuracy=location_accuracy,
            device=device,
            comment="Maui(s) spotted during MAUI63 dolphin survey",
            user_auth_id=args.user_auth_id,
            user_email=args.user_email,
            animal_group_id=maui["animal_group_id"],
            animal_species_id=maui["id"],
            answers=[],
        )
        sighting_res = api.post("sightings", data=sighting)

        # Now upload the media:
        img = Image.open(Path(args.imgdir) / image["file_name"])
        for annotation in annotations:
            bbox = annotation["bbox"]
            x, y, w, h = bbox
            padding = args.padding
            x1 = max(0, int(x - padding))
            y1 = max(0, int(y - padding))
            x2 = min(int(x + w + padding), img.width)
            y2 = min(int(y + h + padding), img.height)
            crop = img.crop((x1, y1, x2, y2))

            data = {
                "name": image["file_name"],
                "media_type": "IMAGE",
                "sighting_id": sighting_res["sighting"]["id"],
            }

            # Read/write 'cos posting the file handle uses the name of the file etc. I think.
            tmp_path = Path("/tmp") / image["file_name"]
            crop.save(tmp_path)

            files = [("file", open(tmp_path, "rb"))]
            res = api.post("media", data=data, files=files)
            print(res)
            tmp_path.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("coco")
    parser.add_argument("imgdir")
    parser.add_argument("url")
    parser.add_argument("api_key")
    parser.add_argument("user_auth_id")
    parser.add_argument("user_email")
    parser.add_argument("--padding", type=int, default=200)
    args = parser.parse_args()

    main(args)

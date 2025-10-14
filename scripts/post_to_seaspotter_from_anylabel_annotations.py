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
    # print([i["name"] for i in species["animal_species"]])
    maui = [i for i in species["animal_species"] if i["name"] == "Māui or Hector's dolphin"][0]

    for ann_path in sorted(Path(args.imgdir).glob("*.json")):
        img_path = ann_path.with_suffix(".jpg")
        print("-" * 80)
        print("Processing image:", img_path.name)
        assert ann_path.exists(), f"Annotation file {ann_path} not found"
        with open(ann_path) as f:
            ann = json.load(f)

        annotations = ann["shapes"]
        assert len(annotations) > 0

        meta = ann["src"]["frame"]
        lat = meta["lat"]
        lon = meta["lon"]
        t = meta["gps_time"]
        location_accuracy = "PLANE-SURVEY"
        device = "PLANE-CAMERA"
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
            user_full_name=args.user_full_name,
            animal_group_id=maui["animal_group_id"],
            animal_species_id=maui["id"],
            answers=[],
        )
        sighting_res = api.post("sightings", data=sighting)

        # Now upload the media:
        img = Image.open(img_path)
        for annotation in annotations:
            bbox = annotation["points"]
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            padding = args.padding
            x1 = max(0, int(x1 - padding))
            y1 = max(0, int(y1 - padding))
            x2 = min(int(x2 + padding), img.width)
            y2 = min(int(y2 + padding), img.height)
            crop = img.crop((x1, y1, x2, y2))

            data = {
                "name": img_path.name,
                "media_type": "IMAGE",
                "sighting_id": sighting_res["sighting"]["id"],
            }

            # Read/write 'cos posting the file handle uses the name of the file etc. I think.
            tmp_path = Path("/tmp") / img_path.name
            crop.save(tmp_path)

            files = [("file", open(tmp_path, "rb"))]
            res = api.post("media", data=data, files=files)
            print(res)
            tmp_path.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("imgdir")
    parser.add_argument("url")
    parser.add_argument("api_key")
    parser.add_argument("user_auth_id")
    parser.add_argument("user_email")
    parser.add_argument("user_full_name")
    parser.add_argument("--padding", type=int, default=200)
    args = parser.parse_args()

    main(args)

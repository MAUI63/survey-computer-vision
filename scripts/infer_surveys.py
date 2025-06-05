import json
from dataclasses import asdict
from pathlib import Path

from loguru import logger
from sahi import AutoDetectionModel

from drone.action_aviation import ActionAviationFlight
from drone.action_aviation_multi_camera import ActionAviationMultiCameraFlight
from drone.cw25 import CW25DroneFlight
from drone.roc import RocDroneFlight
from infer import predict


def get_tasks(args):
    survey_dir = Path(args.surveydir)
    for flight_dir in sorted(survey_dir.iterdir()):
        if not flight_dir.is_dir():
            continue
        assert flight_dir.name.startswith("flight")
        cls = {
            "roc1": RocDroneFlight,
            "roc2": RocDroneFlight,
            "cw25": CW25DroneFlight,
            "action-aviation": ActionAviationFlight,
            "action-aviation-multicamera": ActionAviationMultiCameraFlight,
        }
        cls = cls[survey_dir.parent.name]
        logger.info(f"Flight {flight_dir} using {cls}")
        flight = cls(flight_dir)
        yield survey_dir.parent.name, survey_dir.name, flight_dir.name, list(flight.get_photo_paths())


def main(args):
    model_path = Path(args.modeldir)
    logger.info("Loading model")
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov9",
        model_path=str(model_path / "weights" / "best.pt"),
        config_path=args.modelconfig,
        confidence_threshold=args.model_confidence_threshold,
        device=args.device,
    )
    logger.info(f"Model loaded with device {model.device}")

    # t = datetime.now().astimezone(pytz.timezone("Pacific/Auckland"))
    outdir = (
        Path(args.outdir)
        / model_path.name
        / f"{args.slice_height}x{args.slice_width}_{args.overlap_height_ratio}x{args.overlap_width_ratio}"
        / f"{args.model_confidence_threshold}"
        # / t.strftime("%Y.%m.%d_%H.%M.%S")
    )
    logger.info(f"Output to {outdir}")
    for survey_type, survey, flight, img_paths in get_tasks(args):
        logger.info(f"Processing {survey_type}/{survey}/{flight}")
        try:
            title_prefix = f"yolov9 ({model_path.name}) imgs @ {survey_type}/{survey}/{flight}"
            odir = outdir / survey_type / survey / flight
            odir.mkdir(parents=True, exist_ok=True)  # Fail if exists as we want a clean slate
            inference_dir = odir / "inferences"
            inference_dir.mkdir(exist_ok=True)

            # Filter out images already completed:
            filtered = []
            for img_path in img_paths:
                opath = inference_dir / f"{img_path.stem}.json"
                if opath.exists():
                    try:
                        with open(opath) as f:
                            _ = json.load(f)
                        continue
                    except:  # NOQA
                        pass
                filtered.append(img_path)

            if not filtered:
                logger.info("Already complete!")
                continue

            logger.info(f"Inferring {len(filtered)}")
            for inference in predict(
                model,
                title_prefix,
                filtered,
                odir=odir,
                model_confidence_threshold=args.model_confidence_threshold,
                slice_height=args.slice_height,
                slice_width=args.slice_height,
                overlap_height_ratio=args.overlap_height_ratio,
                overlap_width_ratio=args.overlap_width_ratio,
                batch_size=36,
                novisual=args.novisual,
            ):
                opath = inference_dir / f"{inference.img_path.stem}.json"
                with open(opath, "w") as f:
                    json.dump([asdict(d) for d in inference.detections], f, indent=2)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - exiting")
            exit(0)
        except:  # NOQA
            logger.exception("Failed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("modeldir")
    parser.add_argument("modelconfig")
    parser.add_argument("surveydir")
    parser.add_argument("outdir")
    parser.add_argument("slice_height", type=int)
    parser.add_argument("slice_width", type=int)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2)
    parser.add_argument("--model_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--novisual", action="store_true")
    args = parser.parse_args()

    main(args)

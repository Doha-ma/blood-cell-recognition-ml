"""Utilities for downloading the dataset and instantiating Keras generators."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Tuple

from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.preprocessing.image import ImageDataGenerator


LOGGER = logging.getLogger(__name__)


def download_blood_dataset(
    dataset_path: str = "paultimothymooney/blood-cell-images",
    dest_path: str | Path = "data/raw",
    unzip: bool = True,
    force: bool = False,
) -> Path:
    """Download the Kaggle dataset locally."""

    dest = Path(dest_path)
    if dest.exists() and any(dest.iterdir()):
        if not force:
            LOGGER.info(
                "Dataset already present in %s. Skipping download (use --force to overwrite).",
                dest,
            )
            return dest
        LOGGER.warning("Removing existing destination %s because --force was supplied", dest)
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Authenticating to Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    LOGGER.info("Downloading dataset %s into %s (unzip=%s)", dataset_path, dest, unzip)
    api.dataset_download_files(dataset_path, path=str(dest), unzip=unzip)
    LOGGER.info("Dataset downloaded%s at %s", " and unzipped" if unzip else "", dest)
    return dest


def get_data_generators(
    train_dir: str | Path = "data/raw/train",
    test_dir: str | Path = "data/raw/test",
    img_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
):
    """Create data generators with sane augmentation defaults."""

    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory {train_dir} does not exist")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test/validation directory {test_dir} does not exist")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, test_generator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle white blood cell dataset and inspect directories."
    )
    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser("download", help="Download the dataset from Kaggle")
    download_parser.add_argument("--dataset", default="paultimothymooney/blood-cell-images")
    download_parser.add_argument("--dest", default="data/raw")
    download_parser.add_argument("--no-unzip", action="store_true", help="Skip unzipping the archive")
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the destination folder before downloading again",
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="Create ImageDataGenerators and print dataset statistics"
    )
    inspect_parser.add_argument("--train-dir", default="data/raw/train")
    inspect_parser.add_argument("--test-dir", default="data/raw/test")
    inspect_parser.add_argument(
        "--img-size", nargs=2, type=int, default=(64, 64), metavar=("WIDTH", "HEIGHT")
    )
    inspect_parser.add_argument("--batch-size", type=int, default=32)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "download":
        download_blood_dataset(
            dataset_path=args.dataset,
            dest_path=args.dest,
            unzip=not args.no_unzip,
            force=args.force,
        )
        return 0

    if args.command == "inspect":
        train_gen, test_gen = get_data_generators(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
        )
        LOGGER.info("Classes: %s", train_gen.class_indices)
        LOGGER.info(
            "Samples -> train: %d | test: %d",
            train_gen.samples,
            test_gen.samples,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

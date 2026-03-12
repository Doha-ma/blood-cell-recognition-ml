"""Dataset splitting utilities for the blood cell project."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_class_dirs(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path {source_dir} is not a directory")

    dirs = sorted([item for item in source_dir.iterdir() if item.is_dir()])
    if not dirs:
        raise ValueError(f"No class folders found in {source_dir}")
    return dirs


def _filter_images(class_dir: Path) -> list[Path]:
    return [img for img in class_dir.iterdir() if img.suffix.lower() in VALID_EXTENSIONS]


def _prepare_output(output_dir: Path, clear_output: bool) -> None:
    if output_dir.exists() and clear_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def split_dataset(
    source_dir: str | Path,
    output_dir: str | Path = "data/raw",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    clear_output: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Split the dataset into train/validation/test folders."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    source = Path(source_dir)
    output = Path(output_dir)
    _prepare_output(output, clear_output)

    rng = random.Random(seed)
    class_dirs = _list_class_dirs(source)

    summary: Dict[str, Dict[str, int]] = {"train": {}, "validation": {}, "test": {}}

    for cls_dir in class_dirs:
        images = _filter_images(cls_dir)
        if not images:
            raise ValueError(f"No images found in {cls_dir}")
        rng.shuffle(images)

        n_images = len(images)
        train_end = int(n_images * train_ratio)
        val_end = train_end + int(n_images * val_ratio)

        splits = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, files in splits.items():
            dest_dir = output / split_name / cls_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dest_dir / src.name)
            summary[split_name][cls_dir.name] = len(files)

    return summary


def _format_summary(summary: Dict[str, Dict[str, int]]) -> str:
    lines = []
    for split, classes in summary.items():
        total = sum(classes.values())
        lines.append(f"{split} ({total} images):")
        for cls_name, count in sorted(classes.items()):
            lines.append(f"  - {cls_name}: {count}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a blood cell dataset into standard folders.")
    parser.add_argument("--source-dir", required=True, help="Path containing the eight class folders")
    parser.add_argument("--output-dir", default="data/raw", help="Destination directory for the splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of images for training")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of images for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove the output directory before copying files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = split_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        clear_output=args.clear_output,
    )
    print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

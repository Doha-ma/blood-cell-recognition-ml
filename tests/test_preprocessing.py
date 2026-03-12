from pathlib import Path

import pytest

from preprocessing import split_dataset


def _create_dummy_dataset(root: Path, classes: list[str], files_per_class: int = 10) -> None:
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for idx in range(files_per_class):
            (cls_dir / f"{cls}_{idx}.png").write_text("placeholder", encoding="utf-8")


def test_split_dataset_creates_expected_structure(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "splits"
    classes = ["basophil", "neutrophil", "platelet"]
    _create_dummy_dataset(source, classes, files_per_class=10)

    summary = split_dataset(
        source_dir=source,
        output_dir=output,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=123,
        clear_output=False,
    )

    for cls in classes:
        assert summary["train"][cls] == 6
        assert summary["validation"][cls] == 2
        assert summary["test"][cls] == 2

    for split in ("train", "validation", "test"):
        for cls in classes:
            files = list((output / split / cls).glob("*.png"))
            assert len(files) == summary[split][cls]


def test_split_dataset_ratio_validation(tmp_path: Path):
    source = tmp_path / "source"
    _create_dummy_dataset(source, ["basophil"], files_per_class=1)
    with pytest.raises(ValueError):
        split_dataset(source_dir=source, output_dir=tmp_path / "out", train_ratio=0.9, val_ratio=0.2)

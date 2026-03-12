"""Run inference with the trained blood cell classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_class_map(class_map_path: str | Path | None) -> Dict[int, str]:
    if not class_map_path:
        return {}
    path = Path(class_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Class map file {path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Stored as {class_name: index}; invert to {index: class_name}
    return {int(idx): name for name, idx in raw.items()}


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path
        return
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
            yield file


def preprocess_image(image_path: Path, img_size: tuple[int, int]) -> np.ndarray:
    image = load_img(image_path, target_size=img_size)
    array = img_to_array(image) / 255.0
    return np.expand_dims(array, axis=0)


def predict_images(
    model_path: str | Path,
    target_path: str | Path,
    class_map_path: str | Path | None = None,
    img_size: tuple[int, int] = (128, 128),
    top_k: int = 3,
) -> List[dict]:
    model = load_model(model_path)
    class_map = load_class_map(class_map_path) if class_map_path else {}
    predictions = []

    target = Path(target_path)
    if not target.exists():
        raise FileNotFoundError(f"Path {target} does not exist")

    image_files = list(iter_images(target))
    if not image_files:
        raise FileNotFoundError(f"Aucun fichier image valable trouvé dans {target}")

    for image_file in image_files:
        batch = preprocess_image(image_file, img_size)
        probs = model.predict(batch, verbose=0)[0]
        indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "class": class_map.get(int(idx), str(int(idx))),
                "confidence": float(probs[idx]),
            }
            for idx in indices
        ]
        predictions.append(
            {
                "image": str(image_file),
                "prediction": top_predictions[0]["class"],
                "confidence": top_predictions[0]["confidence"],
                "top_k": top_predictions,
            }
        )
    return predictions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict blood cell types from images.")
    parser.add_argument("path", help="Image file or directory to classify")
    parser.add_argument("--model-path", default="models/blood_cell_classifier.keras")
    parser.add_argument("--class-map", default=None, help="Path to the *_classes.json file")
    parser.add_argument("--img-size", nargs=2, type=int, default=(128, 128), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    img_size = (int(args.img_size[0]), int(args.img_size[1]))
    results = predict_images(
        model_path=args.model_path,
        target_path=args.path,
        class_map_path=args.class_map,
        img_size=img_size,
        top_k=args.top_k,
    )
    for item in results:
        primary = item["prediction"]
        confidence = item["confidence"]
        print(f"{item['image']}: {primary} ({confidence:.2%})")
        for alt in item["top_k"][1:]:
            print(f"    alt -> {alt['class']} ({alt['confidence']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

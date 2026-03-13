"""Model training entry-point for blood cell classification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from keras import callbacks, layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator


def build_generators(
    dataset_dir: str | Path,
    img_size: Tuple[int, int],
    batch_size: int,
):
    """Create Keras generators for train/validation/test splits."""

    dataset_dir = Path(dataset_dir)
    splits = {
        "train": dataset_dir / "train",
        "validation": dataset_dir / "validation",
        "test": dataset_dir / "test",
    }
    for split, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected directory {path} for split '{split}'")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(splits["train"]),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        str(splits["validation"]),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        str(splits["test"]),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 1e-3,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """Create a simple CNN classifier."""

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(dropout),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(generator) -> Dict[int, float]:
    """Compute balanced class weights based on the generator samples."""

    labels = generator.classes
    num_classes = generator.num_classes
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = {}
    for idx in range(num_classes):
        count = counts[idx] if idx < len(counts) else 0
        if count == 0:
            weights[idx] = 0.0
        else:
            weights[idx] = float(total / (num_classes * count))
    return weights


def save_metadata(
    model_path: Path,
    train_generator,
    history: tf.keras.callbacks.History,
    test_metrics: Dict[str, float],
) -> None:
    """Persist class indices and the training history to disk."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path = model_path.with_name(model_path.stem + "_classes.json")
    history_path = model_path.with_name(model_path.stem + "_history.json")

    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(train_generator.class_indices, f, indent=2)

    serializable_history = {key: [float(v) for v in values] for key, values in history.history.items()}
    payload = {"history": serializable_history, "test_metrics": test_metrics}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CNN to classify white blood cells.")
    parser.add_argument("--dataset-dir", default="data/raw", help="Directory containing train/validation/test splits")
    parser.add_argument("--model-path", default="models/blood_cell_classifier.keras")
    parser.add_argument("--img-size", nargs=2, type=int, default=(128, 128), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable automatic class weighting")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(argv: list[str] | None = None) -> Dict[str, float]:
    args = parse_args(argv)
    set_seeds(args.seed)

    height, width = (int(args.img_size[0]), int(args.img_size[1]))
    img_size = (height, width)
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen = build_generators(args.dataset_dir, img_size, args.batch_size)
    model = build_model(
        input_shape=(height, width, 3),
        num_classes=train_gen.num_classes,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )

    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(1, args.patience // 2), factor=0.5),
    ]

    class_weights = None if args.no_class_weights else compute_class_weights(train_gen)
    if class_weights:
        # map class index to weight explicitly used by Keras
        class_weights = {int(idx): weight for idx, weight in class_weights.items()}

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks_list,
        class_weight=class_weights,
    )

    evaluation = model.evaluate(test_gen, verbose=0)
    metrics = {name: float(value) for name, value in zip(model.metrics_names, evaluation)}

    model.save(model_path)
    save_metadata(model_path, train_gen, history, metrics)

    print(f"Model saved to {model_path}")
    for name, value in metrics.items():
        print(f"Test {name}: {value:.4f}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    train(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessor:
    """Preprocessing helper for blood cell classification.

    Responsibilities:
    1. Validate dataset directories and files.
    2. Analyze class distribution, detect imbalance, and report balance.
    3. Compute class weights.
    4. Build data generators with augmentation.
    5. Save augmentation examples and preprocessing info including extra metrics.
    """

    def __init__(
        self,
        dataset_dir: str,
        results_dir: str = "../data/processed",
        models_dir: str = "../data/processed",
        img_size: tuple = (224, 224),
        batch_size: int = 32,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.img_size = img_size
        self.batch_size = batch_size

        self.generators = {}
        self.class_indices = None
        self.preprocessing_info = {}

    def validate_dataset(self):
        """Ensure the expected splits and class folders exist and contain images."""
        print("[INFO] Validating dataset structure and sample files...")
        splits = ["train", "validation", "test"]
        errors = []
        for split in splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                errors.append(f"Missing split: {split_dir}")
                continue
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = split_dir / cls
                if not cls_dir.is_dir():
                    errors.append(f"Expected directory but found file: {cls_dir}")
                    continue
                imgs = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
                if not imgs:
                    errors.append(f"No images found in {cls_dir}")
        if errors:
            for e in errors:
                print(f"[ERROR] {e}")
            raise RuntimeError("Dataset validation failed; see errors above")
        print("[INFO] Dataset validation passed.")
        return True

    def analyze_distribution(self):
        """Count images per class for train/val/test, plot distribution, and report balance."""
        print("[INFO] Analyzing class distribution...")
        counts = {split: {} for split in ("train", "validation", "test")}

        for split in counts:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                print(f"[WARNING] {split_dir} not found, skipping")
                continue
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = split_dir / cls
                if cls_dir.is_dir():
                    counts[split][cls] = len(list(cls_dir.glob("*.png"))) + len(
                        list(cls_dir.glob("*.jpg")))

        # print results
        for split, data in counts.items():
            print(f"  {split}:")
            for cls, cnt in data.items():
                print(f"    {cls}: {cnt}")

        # plot training distribution
        train_counts = counts.get("train", {})
        if train_counts:
            fig, ax = plt.subplots(figsize=(8, 4))
            classes = list(train_counts.keys())
            values = [train_counts[c] for c in classes]
            ax.bar(classes, values, color="skyblue")
            ax.set_title("Training set class distribution")
            ax.set_ylabel("Number of images")
            ax.set_xticklabels(classes, rotation=45, ha="right")
            plt.tight_layout()
            plot_path = self.results_dir / "class_distribution.png"
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"[INFO] Saved distribution plot to {plot_path}")

        # compute balance metrics
        balance_report = {}
        if train_counts:
            max_count = max(train_counts.values())
            min_count = min(train_counts.values())
            mean_count = sum(train_counts.values()) / len(train_counts)
            balance_report["max"] = max_count
            balance_report["min"] = min_count
            balance_report["mean"] = mean_count
            balance_report["ratio_max_min"] = max_count / (min_count if min_count else 1)

            if min_count == 0 or balance_report["ratio_max_min"] > 1.5:
                print("[WARNING] Detected class imbalance in training set")
            else:
                print("[INFO] No significant class imbalance detected")

            # show percentages
            total = sum(train_counts.values())
            print("[INFO] Class percentages:")
            for cls, cnt in train_counts.items():
                pct = cnt / total * 100 if total else 0
                print(f"    {cls}: {pct:.2f}%")
        self.preprocessing_info["balance_report"] = balance_report
        self.preprocessing_info["counts"] = counts

        return counts

    def compute_class_weights(self, counts: dict):
        """Compute and save class weights based on training distribution.

        Returns a dict mapping class name to weight.
        """
        print("[INFO] Computing class weights...")
        train_counts = counts.get("train", {})
        classes = sorted(train_counts.keys())
        if not classes:
            raise ValueError("No training classes found for computing weights")

        # manual balanced weight formula: N / (num_classes * count_i)
        total = sum(train_counts.values())
        num_cls = len(classes)
        weights = {}
        for cls in classes:
            cnt = train_counts.get(cls, 0)
            weights[cls] = (total / (num_cls * cnt)) if cnt > 0 else 0.0

        out_path = self.models_dir / "class_weights.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(weights, f)
        print(f"[INFO] Saved class weights to {out_path}")

        # record names and weights in preprocessing info
        self.preprocessing_info["class_names"] = classes
        self.preprocessing_info["class_weights"] = weights
        self.preprocessing_info["class_weights_path"] = str(out_path)
        return weights

    def make_generators(self):
        """Create Keras ImageDataGenerators for train/validation/test."""
        print("[INFO] Creating data generators...")
        train_gen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )
        test_val_gen = ImageDataGenerator(rescale=1.0 / 255)

        self.generators["train"] = train_gen.flow_from_directory(
            self.dataset_dir / "train",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True,
        )
        self.generators["validation"] = test_val_gen.flow_from_directory(
            self.dataset_dir / "validation",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        self.generators["test"] = test_val_gen.flow_from_directory(
            self.dataset_dir / "test",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        # save class indices for reference
        self.class_indices = self.generators["train"].class_indices
        self.preprocessing_info["class_indices"] = self.class_indices
        self.preprocessing_info["img_size"] = self.img_size
        self.preprocessing_info["batch_size"] = self.batch_size
        print("[INFO] Generators created and class indices recorded.")
        return self.generators

    def show_augmentation_examples(self, num_examples: int = 9):
        """Display and save a grid of augmented images from the training generator."""
        print("[INFO] Generating augmentation examples...")
        gen = self.generators.get("train")
        if gen is None:
            raise RuntimeError("Train generator not created")
        images, labels = next(gen)
        n = min(num_examples, images.shape[0])
        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i in range(n):
            axes[i].imshow(images[i])
            axes[i].axis("off")
        plt.tight_layout()
        save_path = self.results_dir / "augmentation_examples.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[INFO] Saved augmentation examples to {save_path}")

    def save_preprocessing_info(self):
        """Dump preprocessing info dictionary to disk."""
        info_path = self.models_dir / "preprocessing_info.pkl"
        with open(info_path, "wb") as f:
            pickle.dump(self.preprocessing_info, f)
        print(f"[INFO] Saved preprocessing info to {info_path}")
        return info_path


if __name__ == "__main__":
    # default usage example
    base = Path(__file__).parent
    dataset_root = base / "../data/raw"
    pre = ImagePreprocessor(str(dataset_root))
    # validate structure before anything else
    pre.validate_dataset()
    counts = pre.analyze_distribution()
    pre.compute_class_weights(counts)
    pre.make_generators()
    pre.show_augmentation_examples()
    pre.save_preprocessing_info()
    print("[INFO] Preprocessing completed.")

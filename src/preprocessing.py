import os
import shutil
import random

# Original dataset folder containing the 8 classes
source_dir = "C:\\Ai_project\\blood-cell-recognition-ml\\bloodcells_dataset"  # dataset path

# New folder structure
base_dir = "dataset"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# 8 classes
classes = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

# Create train/validation/test folders
for folder in [train_dir, val_dir, test_dir]:
    for cls in classes:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

# Split images for each class
for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    train_split = int(0.7 * len(images))
    val_split = int(0.85 * len(images))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    # Copy images to new folders
    for img in train_images:
        shutil.copy(os.path.join(source_dir, cls, img),
                    os.path.join(train_dir, cls, img))

    for img in val_images:
        shutil.copy(os.path.join(source_dir, cls, img),
                    os.path.join(val_dir, cls, img))

    for img in test_images:
        shutil.copy(os.path.join(source_dir, cls, img),
                    os.path.join(test_dir, cls, img))

print("Dataset has been split into train, validation, and test for 8 classes.")
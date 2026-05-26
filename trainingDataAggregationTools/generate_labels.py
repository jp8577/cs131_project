# generate_labels.py

# This helper script takes non-bounding box images and converts them to

import os
import shutil
import random
from pathlib import Path

random.seed(42)

CLASS_MAP = {
    # Version 0: Initial Roboflow dataset
    'freshapple': 0,
    'freshbanana': 1,
    'freshcucumber': 2,
    'freshokra': 3,
    'freshorange': 4,
    'freshpotato': 5,
    'freshtomato': 6,
    'rottenapple': 7,
    'rottenbanana': 8,
    'rottencucumber': 9,
    'rottenokra': 10,
    'rottenorange': 11,
    'rottenpotato': 12,
    'rottentomato': 13,
    # Version 1: Added Kaggle Dataset
    'freshbellpepper': 14,
    'freshbittergourd': 15,
    'freshcapsicum': 16,
    'freshcarrot': 17,
    'freshmango': 18,
    'freshstrawberry': 19,
    'rottenbellpepper': 20,
    'rottenbittergourd': 21,
    'rottencapsicum': 22,
    'rottencarrot': 23,
    'rottenmango': 24,
    'rottenstrawberry': 25,
}

SPLIT_RATIOS = {'train': 0.70, 'valid': 0.15, 'test': 0.15}

# Where the raw Kaggle images are (two-level: Fresh/FreshApple/, Rotten/RottenApple/)
source_root = Path("dataset")

# Where to output into existing dataset structure
output_root = Path("formattedDataset")

# Create output dirs
for split in SPLIT_RATIOS:
    (output_root / split / "images").mkdir(parents=True, exist_ok=True)
    (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

# Lowercase all folder names recursively
for folder in sorted(source_root.rglob("*")):
    if folder.is_dir() and folder.name != folder.name.lower():
        folder.rename(folder.parent / folder.name.lower())

# Walk all class folders (handles two-level structure)
total_copied = 0
for class_name, class_id in CLASS_MAP.items():
    # Search recursively for matching folder
    matches = list(source_root.rglob(class_name))
    if not matches:
        print(f"  [SKIP] {class_name}: folder not found")
        continue

    img_dir = matches[0]
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))

    if not images:
        print(f"  [SKIP] {class_name}: no images found")
        continue

    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLIT_RATIOS['train'])
    n_valid = int(n * SPLIT_RATIOS['valid'])

    splits = {
        'train': images[:n_train],
        'valid': images[n_train:n_train + n_valid],
        'test':  images[n_train + n_valid:],
    }

    for split, split_images in splits.items():
        for img_path in split_images:
            # Copy image
            dest_img = output_root / split / "images" / img_path.name
            shutil.copy(img_path, dest_img)

            # Write label
            dest_lbl = output_root / split / "labels" / (img_path.stem + ".txt")
            with open(dest_lbl, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    print(f"  {class_name}: {n} images -> train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")
    total_copied += n

print(f"\nDone. {total_copied} total images processed.")
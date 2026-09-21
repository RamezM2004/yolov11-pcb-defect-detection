"""
Oversample weak classes (6 & 7) in the training set.
Safe: never deletes originals.
Creates copies with prefix 'os2_'.

Run BEFORE training Exp B.
"""

import shutil
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT  = Path(__file__).parent / "DSPCBSD+-1"
TRAIN_IMG_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LBL_DIR = DATASET_ROOT / "train" / "labels"

TARGET_CLASSES = {6, 7}
COPIES = 2          # 2 = total about 2x for target images
DRY_RUN = False
# ──────────────────────────────────────────────────────────────────────────────


def get_classes(lbl_path: Path) -> set:
    classes = set()

    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(int(parts[0]))

    return classes


def main():
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Train images:  {TRAIN_IMG_DIR}")
    print(f"Train labels:  {TRAIN_LBL_DIR}")

    if not TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {TRAIN_IMG_DIR}")

    if not TRAIN_LBL_DIR.exists():
        raise FileNotFoundError(f"Labels folder not found: {TRAIN_LBL_DIR}")

    label_files = sorted(TRAIN_LBL_DIR.glob("*.txt"))
    print(f"\nTotal label files before oversampling: {len(label_files)}")

    # Avoid duplicating already duplicated files
    prefixes = tuple(f"os{i}_" for i in range(2, COPIES + 1))
    originals = [f for f in label_files if not f.stem.startswith(prefixes)]

    targets = [f for f in originals if TARGET_CLASSES & get_classes(f)]
    print(f"Original images containing classes {TARGET_CLASSES}: {len(targets)}")

    if DRY_RUN:
        print("\n[DRY RUN] Files that would be created:")

    added_imgs, added_lbls, skipped, already_exists = 0, 0, 0, 0

    for copy_n in range(2, COPIES + 1):
        prefix = f"os{copy_n}_"

        for lbl in targets:
            img = TRAIN_IMG_DIR / (lbl.stem + ".jpg")

            if not img.exists():
                skipped += 1
                continue

            new_img = TRAIN_IMG_DIR / (prefix + img.name)
            new_lbl = TRAIN_LBL_DIR / (prefix + lbl.name)

            if new_img.exists() or new_lbl.exists():
                already_exists += 1
                continue

            if DRY_RUN:
                print(f"  {new_img.name}")
                continue

            shutil.copy2(img, new_img)
            shutil.copy2(lbl, new_lbl)

            added_imgs += 1
            added_lbls += 1

    if DRY_RUN:
        print("\nDry run finished. No files were copied.")
        return

    print(f"\nDone.")
    print(f"Added images: {added_imgs}")
    print(f"Added labels: {added_lbls}")
    print(f"Skipped missing images: {skipped}")
    print(f"Already existed: {already_exists}")

    # Report new class distribution
    all_labels = list(TRAIN_LBL_DIR.glob("*.txt"))
    counts = Counter()

    for f in all_labels:
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1

    total = sum(counts.values())

    print(f"\nNew training distribution ({total} total instances):")
    for cls in sorted(counts):
        print(f"  Class {cls}: {counts[cls]:5d}  ({counts[cls] / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
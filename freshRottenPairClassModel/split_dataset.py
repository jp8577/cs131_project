# split_dataset.py  —  stratified 70 / 15 / 15 split by primary class
import os, shutil, random
from collections import defaultdict, Counter

random.seed(42)

DATASET_ROOT = 'dataset'
SPLITS       = ['train', 'valid', 'test']
VALID_FRAC   = 0.15
TEST_FRAC    = 0.15

# ── 1. Collect every image path (across all pre-existing splits) ──────────────
all_images = []   # list of (abs_image_path, abs_label_path_or_None)

for split in SPLITS:
    img_dir = os.path.join(DATASET_ROOT, split, 'images')
    lbl_dir = os.path.join(DATASET_ROOT, split, 'labels')
    if not os.path.isdir(img_dir):
        continue
    for fname in os.listdir(img_dir):
        if fname.startswith('.'):
            continue
        img_path = os.path.join(img_dir, fname)
        stem     = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, stem + '.txt')
        all_images.append((img_path, lbl_path if os.path.exists(lbl_path) else None))

print(f"Total images found across all splits: {len(all_images)}")

# ── 2. Assign each image a primary class (most-frequent class in its labels) ──
def primary_class(lbl_path):
    """Return the most common class index string in the label file, or 'unknown'."""
    if lbl_path is None:
        return 'unknown'
    counts = Counter()
    try:
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[line.split()[0]] += 1
    except OSError:
        return 'unknown'
    return counts.most_common(1)[0][0] if counts else 'unknown'

by_class = defaultdict(list)
for item in all_images:
    cls = primary_class(item[1])
    by_class[cls].append(item)

# ── 3. Stratified split within each class ─────────────────────────────────────
train_set, valid_set, test_set = [], [], []

for cls, items in sorted(by_class.items()):
    random.shuffle(items)
    n      = len(items)
    n_v    = max(1, int(n * VALID_FRAC))
    n_t    = max(1, int(n * TEST_FRAC))
    valid_set.extend(items[:n_v])
    test_set.extend(items[n_v:n_v + n_t])
    train_set.extend(items[n_v + n_t:])

print(f"After stratified split  →  Train: {len(train_set)} | Valid: {len(valid_set)} | Test: {len(test_set)}")

# ── 4. Ensure destination directories exist ───────────────────────────────────
for split in SPLITS:
    os.makedirs(os.path.join(DATASET_ROOT, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, split, 'labels'), exist_ok=True)

# ── 5. Move files to their assigned split (skip if already in the right place) ─
def move_pair(img_src, lbl_src, dest_split):
    dest_img_dir = os.path.join(DATASET_ROOT, dest_split, 'images')
    dest_lbl_dir = os.path.join(DATASET_ROOT, dest_split, 'labels')

    img_dest = os.path.join(dest_img_dir, os.path.basename(img_src))
    if os.path.abspath(img_src) != os.path.abspath(img_dest):
        shutil.move(img_src, img_dest)

    if lbl_src:
        lbl_dest = os.path.join(dest_lbl_dir, os.path.basename(lbl_src))
        if os.path.abspath(lbl_src) != os.path.abspath(lbl_dest):
            shutil.move(lbl_src, lbl_dest)

for img_path, lbl_path in train_set:
    move_pair(img_path, lbl_path, 'train')
for img_path, lbl_path in valid_set:
    move_pair(img_path, lbl_path, 'valid')
for img_path, lbl_path in test_set:
    move_pair(img_path, lbl_path, 'test')

# ── 6. Per-class breakdown ────────────────────────────────────────────────────
print("\nPer-class split counts (class | train | valid | test):")
all_assigned = (
    [(img, lbl, 'train') for img, lbl in train_set] +
    [(img, lbl, 'valid') for img, lbl in valid_set] +
    [(img, lbl, 'test')  for img, lbl in test_set]
)
class_split_counts = defaultdict(lambda: {'train': 0, 'valid': 0, 'test': 0})
for img, lbl, split in all_assigned:
    cls = primary_class(lbl)
    class_split_counts[cls][split] += 1

for cls, counts in sorted(class_split_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
    total = sum(counts.values())
    print(f"  class {cls:>7}: train={counts['train']:5d} ({counts['train']/total*100:.1f}%)  "
          f"valid={counts['valid']:4d} ({counts['valid']/total*100:.1f}%)  "
          f"test={counts['test']:4d} ({counts['test']/total*100:.1f}%)")

print("\nDone.")

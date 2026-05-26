# cut_and_dump.py
import shutil
from pathlib import Path

source_root = Path("formattedDataset")
dest_root   = Path("../freshRottenPairClassModel/dataset")

# Create destination dirs if they don't exist
for split in ["train", "valid", "test"]:
    (dest_root / split / "images").mkdir(parents=True, exist_ok=True)
    (dest_root / split / "labels").mkdir(parents=True, exist_ok=True)

total_moved = 0

for split in ["train", "valid", "test"]:
    for subfolder in ["images", "labels"]:
        src_dir  = source_root / split / subfolder
        dest_dir = dest_root   / split / subfolder

        if not src_dir.exists():
            print(f"  [SKIP] {src_dir} — not found")
            continue

        files = list(src_dir.iterdir())
        for f in files:
            if f.is_file():
                shutil.copy(f, dest_dir / f.name)
                f.unlink()  # delete after copying
        
        print(f"  {split}/{subfolder}: moved {len(files)} files")
        total_moved += len(files)

print(f"\nDone. {total_moved} total files moved to {dest_root}")
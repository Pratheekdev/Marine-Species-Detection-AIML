"""
Build a flat dataset structure from the nested coral/fish folders.
The existing dataset has:
  dataset/coral/Boulder_Coral/...
  dataset/coral/Branched_Coral/...
  dataset/fish/Trout/...
  etc.

We need a flat structure for flow_from_directory:
  dataset_flat/Boulder_Coral/...
  dataset_flat/Branched_Coral/...
  dataset_flat/Trout/...
  etc.

This script uses junction points (Windows) or copies to build it.
"""

import os
import shutil

BASE_DIR = "../dataset"
FLAT_DIR = "../dataset_flat"

os.makedirs(FLAT_DIR, exist_ok=True)

skipped = []
created = []

for category in os.listdir(BASE_DIR):
    category_path = os.path.join(BASE_DIR, category)
    if not os.path.isdir(category_path):
        continue
    for species in os.listdir(category_path):
        species_path = os.path.join(category_path, species)
        if not os.path.isdir(species_path):
            continue
        # Clean up the species name for use as a class label
        class_name = species.strip().replace(" ", "_")
        dest = os.path.join(FLAT_DIR, class_name)
        if os.path.exists(dest):
            print(f"  [SKIP] {class_name} already exists")
            skipped.append(class_name)
            continue
        # Count images
        images = [f for f in os.listdir(species_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        if len(images) < 10:
            print(f"  [SKIP] {class_name} has too few images ({len(images)}), skipping")
            skipped.append(class_name)
            continue
        # Create junction (symlink on Windows requires admin; use copy instead)
        print(f"  [COPY] {class_name} ({len(images)} images)...")
        shutil.copytree(species_path, dest)
        created.append(class_name)

print("\n✅ Done!")
print(f"Classes created: {len(created)}")
for c in sorted(created):
    print(f"  - {c}")
if skipped:
    print(f"Skipped: {skipped}")

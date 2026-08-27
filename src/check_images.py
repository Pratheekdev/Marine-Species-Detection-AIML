import os
from PIL import Image

dataset_path = "../dataset"

for root, _, files in os.walk(dataset_path):
    for f in files:
        path = os.path.join(root, f)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            print("Removing corrupted file:", path)
            os.remove(path)

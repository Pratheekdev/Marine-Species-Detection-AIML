import sys
import os
import argparse
import numpy as np
import tensorflow as tf

# Fix Windows console UTF-8 output encoding if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from utils import load_class_labels, get_species_category, preprocess_image

# Default paths relative to src/
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "trained_model_multiclass.h5")
DEFAULT_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "class_labels.json")


def run_prediction(image_path, model_path=DEFAULT_MODEL_PATH, labels_path=DEFAULT_LABELS_PATH, top_n=3):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found at: '{image_path}'")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: '{model_path}'")
        return

    print("[INFO] Loading trained multi-class model...")
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_labels(labels_path)

    print("[INFO] Preprocessing image and running inference...")
    img_array = preprocess_image(image_path, target_size=(150, 150))
    predictions = model.predict(img_array, verbose=0)[0]

    # Rank predictions
    top_indices = np.argsort(predictions)[::-1][:top_n]

    print("\n" + "=" * 60)
    print(f"Image: {os.path.basename(image_path)}")
    print("=" * 60)

    for rank, idx in enumerate(top_indices, start=1):
        species = class_names.get(int(idx), f"Class_{idx}")
        confidence = float(predictions[idx]) * 100
        category = get_species_category(species).upper()
        marker = "[TOP]" if rank == 1 else "     "
        print(f" {marker} Rank {rank}: {species:<24} [{category:<5}] -> {confidence:6.2f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Marine Species & Coral Classifier CLI")
    parser.add_argument("image", help="Path to input image file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .h5 model file")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Path to class_labels.json")
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions to display (default: 3)")

    args = parser.parse_args()
    run_prediction(
        image_path=args.image,
        model_path=args.model,
        labels_path=args.labels,
        top_n=args.top
    )


if __name__ == "__main__":
    main()

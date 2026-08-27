"""
Utility functions for Marine Species & Coral Detection.
Includes image preprocessing, label loading, and taxonomic category mapping.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

CORAL_CLASSES = {
    "Boulder Coral", "Boulder_Coral",
    "Branched Coral", "Branched_Coral",
    "Plate Coral", "Plate_Coral",
    "Soft Coral", "Soft_Coral"
}


def load_class_labels(labels_path):
    """
    Load class label dictionary from JSON file and return a cleaned map.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Class labels file not found at: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        raw_labels = json.load(f)

    # Clean display labels
    class_names = {}
    for idx_str, raw_name in raw_labels.items():
        display = raw_name.replace("_", " ").title()
        class_names[int(idx_str)] = display

    return class_names


def get_species_category(species_name):
    """
    Determine if a species belongs to Coral or Fish taxonomy.
    """
    norm_name = species_name.replace("_", " ").title()
    for coral in CORAL_CLASSES:
        if coral.replace("_", " ").title() == norm_name:
            return "coral"
    return "fish"


def preprocess_image(img_path, target_size=(150, 150)):
    """
    Load and preprocess an image for model input.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = "../dataset_flat"
MODEL_SAVE_PATH = "../models/trained_model_multiclass.h5"
CLASS_LABELS_PATH = "../models/class_labels.json"

# -----------------------------
# 2. Image Preprocessing
# -----------------------------
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

train_data = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_data.num_classes
print(f"\nFound {num_classes} classes:")
class_indices = train_data.class_indices
# Reverse map: index -> class name
index_to_class = {v: k for k, v in class_indices.items()}
for idx, name in sorted(index_to_class.items()):
    print(f"  [{idx}] {name}")

# Save class labels
with open(CLASS_LABELS_PATH, 'w') as f:
    json.dump(index_to_class, f, indent=2)
print(f"\nClass labels saved to {CLASS_LABELS_PATH}")

# -----------------------------
# 3. Build Model using MobileNetV2 transfer learning
# -----------------------------
base_model = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 4. Train Model
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
]

print("\n[INFO] Starting training...\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks
)

# -----------------------------
# 5. Save Model
# -----------------------------
model.save(MODEL_SAVE_PATH)
print(f"\n[OK] Model saved at: {MODEL_SAVE_PATH}")
print(f"Class labels saved at: {CLASS_LABELS_PATH}")
print(f"\nFinal val_accuracy: {max(history.history['val_accuracy']):.4f}")

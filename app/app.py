from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Determine base paths relative to this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "trained_model_multiclass.h5")
CLASS_LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "class_labels.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print(f"[INFO] Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels from JSON (index -> display name)
with open(CLASS_LABELS_PATH, 'r', encoding="utf-8") as f:
    raw_labels = json.load(f)

# Build display-friendly names: "Boulder_Coral" -> "Boulder Coral"
class_names = {}
for idx_str, raw_name in raw_labels.items():
    display = raw_name.replace("_", " ").title()
    class_names[int(idx_str)] = display

# Map class to category (Fish or Coral)
CORAL_CLASSES = {"Boulder Coral", "Branched Coral", "Plate Coral", "Soft Coral"}

def get_category(label):
    if label in CORAL_CLASSES:
        return "coral"
    return "fish"

print(f"[OK] Loaded {len(class_names)} classes:")
for idx, name in sorted(class_names.items()):
    print(f"  [{idx}] {name}")

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path, top_n=3):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:top_n]

    results = []
    for idx in top_indices:
        label = class_names.get(int(idx), f"Class {idx}")
        confidence = round(100 * float(predictions[idx]), 2)
        category = get_category(label)
        results.append({
            "label": label,
            "confidence": confidence,
            "category": category,
            "class_idx": int(idx)
        })

    return results

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "classes_count": len(class_names)
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload an image (PNG, JPG, JPEG, WEBP)."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        results = predict_image(filepath, top_n=3)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    image_url = f"/static/uploads/{filename}"
    return jsonify({
        "image_url": image_url,
        "predictions": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

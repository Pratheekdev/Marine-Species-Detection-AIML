from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import json
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

# -----------------------------------------------
# Lazy-load model & labels (avoids gunicorn startup
# timeout on Render's free tier — model loads on
# first /predict request instead of at import time)
# -----------------------------------------------
_model = None
_class_names = None


def get_model():
    """Load the Keras model on first request (lazy loading)."""
    global _model
    if _model is None:
        import tensorflow as tf
        print(f"[INFO] Loading model from: {MODEL_PATH}", flush=True)
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("[OK] Model loaded successfully.", flush=True)
    return _model


def get_class_names():
    """Load class label mapping on first request (lazy loading)."""
    global _class_names
    if _class_names is None:
        with open(CLASS_LABELS_PATH, 'r', encoding="utf-8") as f:
            raw_labels = json.load(f)
        _class_names = {}
        for idx_str, raw_name in raw_labels.items():
            display = raw_name.replace("_", " ").title()
            _class_names[int(idx_str)] = display
        print(f"[OK] Loaded {len(_class_names)} classes.", flush=True)
    return _class_names


# Map class to category (Fish or Coral)
CORAL_CLASSES = {"Boulder Coral", "Branched Coral", "Plate Coral", "Soft Coral"}


def get_category(label):
    if label in CORAL_CLASSES:
        return "coral"
    return "fish"


# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path, top_n=3):
    from tensorflow.keras.preprocessing import image as keras_image

    model = get_model()
    class_names = get_class_names()

    img = keras_image.load_img(img_path, target_size=(150, 150))
    img_array = keras_image.img_to_array(img) / 255.0
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
        "model_loaded": _model is not None,
        "classes_count": len(_class_names) if _class_names else 0
    })


@app.route("/warmup", methods=["GET"])
def warmup():
    """Pre-load model after deploy — visit this URL once to avoid cold-start on first classify."""
    try:
        get_model()
        get_class_names()
        return jsonify({"status": "warmed up", "classes": len(_class_names)})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


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
        print(f"[ERROR] Prediction failed: {e}", flush=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    image_url = f"/static/uploads/{filename}"
    return jsonify({
        "image_url": image_url,
        "predictions": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# 🌊 Marine Species & Coral Detection System

An end-to-end Deep Learning system for identifying and classifying underwater marine organisms (corals and fish species), featuring **Transfer Learning with MobileNetV2** and an interactive, responsive modern web dashboard.

---

## 📌 Project Overview

Monitoring marine biodiversity and coral reef health is vital for conservation efforts. Underwater photography often suffers from murky waters, lighting variations, and complex backgrounds. 

This project employs **Transfer Learning with MobileNetV2** fine-tuned on marine datasets to deliver fast, high-accuracy multi-class classification and taxonomic grouping for underwater imagery.

---

## ✨ Key Features

- 🐠 **Multi-Class Marine Classification:** Classifies across 13 distinct marine organisms, including diverse fish species and coral reef structures.
- 🏷️ **Taxonomic Grouping:** Automatically categorizes predictions into high-level ecological domains (`Coral` vs. `Fish`).
- 📊 **Top-N Predictions & Confidence:** Returns ranked probabilities and visual confidence bars for the top candidates.
- 🌐 **Interactive Web Dashboard:** Modern glassmorphic web interface with drag-and-drop uploads, instant preview, category badges, and ranked predictions.
- 💻 **Lightweight CLI Tool:** Fast command-line inference for single or batch image classifications.

---

## 🗂️ Supported Species & Classes

| Class ID | Species Name | Taxonomy Group |
| :--- | :--- | :--- |
| `0` | **Black Sea Sprat** | 🐟 Fish |
| `1` | **Boulder Coral** | 🪸 Coral |
| `2` | **Branched Coral** | 🪸 Coral |
| `3` | **Gilt-Head Bream** | 🐟 Fish |
| `4` | **Horse Mackerel** | 🐟 Fish |
| `5` | **Plate Coral** | 🪸 Coral |
| `6` | **Red Mullet** | 🐟 Fish |
| `7` | **Red Sea Bream** | 🐟 Fish |
| `8` | **Sea Bass** | 🐟 Fish |
| `9` | **Shrimp** | 🦐 Marine Invertebrate |
| `10` | **Soft Coral** | 🪸 Coral |
| `11` | **Striped Red Mullet** | 🐟 Fish |
| `12` | **Trout** | 🐟 Fish |

---

## 🏗️ Architecture & Pipeline

```
[ Input Image ] ──► [ Preprocessing (150x150, Normalization) ]
                         │
                         ▼
        [ MobileNetV2 Base (Feature Extractor) ]
                         │
                         ▼
        [ GlobalAveragePooling2D ]
                         │
        [ Dense(256) + BatchNorm + Dropout(0.4) ]
                         │
                         ▼
        [ Softmax Classifier (13 Classes) ]
                         │
                         ▼
   [ Top-N Predictions + Confidence Scoring ]
```

---

## 📂 Project Directory Structure

```
marine_species_coral_detection_new/
├── app/
│   ├── app.py                      # Flask web server & inference API
│   ├── static/
│   │   └── uploads/                # Uploaded images
│   └── templates/
│       └── index.html              # Modern web dashboard interface
├── dataset/                        # Raw source dataset (nested directories)
├── dataset_flat/                   # Flattened class directories for training
├── models/
│   ├── class_labels.json           # JSON mapping of class indices to names
│   ├── trained_model.h5            # Binary classification model
│   └── trained_model_multiclass.h5 # Active 13-class MobileNetV2 model
├── src/
│   ├── build_flat_dataset.py       # Script to flatten nested dataset
│   ├── check_images.py             # Image integrity and corruption checker
│   ├── predict.py                  # CLI inference tool
│   ├── train_model.py              # Binary model training script
│   ├── train_multiclass.py         # Multi-class transfer learning training script
│   └── utils.py                    # Reusable helper functions & taxonomy mappings
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and install required packages:

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation & Training *(Optional)*

If you wish to train the model from scratch:

```bash
# 1. Verify and clean dataset images
python src/check_images.py

# 2. Build flat dataset directory structure
python src/build_flat_dataset.py

# 3. Train multi-class MobileNetV2 model
python src/train_multiclass.py
```

---

## 💻 Usage

### A. Run the Web Application

Launch the Flask server:

```bash
cd app
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### B. Command Line Interface (CLI)

Run inference on any image directly:

```bash
# Basic top-3 prediction
python src/predict.py path/to/sample.jpg

# Display top-5 predictions
python src/predict.py path/to/sample.jpg --top 5
```

---

## 🌐 API Reference

### `GET /health`
Returns backend health and model status.

### `POST /predict`
Uploads an image and returns multi-class predictions.

- **Request:** `multipart/form-data` with key `file` (image file).
- **Response Format:**
```json
{
  "image_url": "/static/uploads/sample.jpg",
  "predictions": [
    {
      "class_idx": 1,
      "label": "Boulder Coral",
      "category": "coral",
      "confidence": 98.42
    },
    {
      "class_idx": 2,
      "label": "Branched Coral",
      "category": "coral",
      "confidence": 1.15
    }
  ]
}
```

---

## 📜 License & Acknowledgments
- Pretrained weights provided via TensorFlow Keras Applications (`MobileNetV2`).
- Designed for marine ecological analysis, educational usage, and conservation technology.

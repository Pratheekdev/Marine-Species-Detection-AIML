"""
WSGI entrypoint for Cloud Deployment (Render, Hugging Face, Heroku, Railway).
"""
import sys
import os

# Ensure repo root and app directory are in Python path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "app"))

from app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

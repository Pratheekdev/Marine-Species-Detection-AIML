"""
WSGI entrypoint for Cloud Deployment (Render, Hugging Face, Heroku, Railway).
"""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(REPO_ROOT, "app")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from app.app import app
except Exception:
    from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

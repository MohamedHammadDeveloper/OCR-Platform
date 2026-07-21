"""
Application-level configuration for the OCR Platform.

Per-model settings do NOT live here — each model owns its own config under
``services/<model>/config.py``. This file only holds settings that belong to
the shared FastAPI application.
"""

import os

SERVICE_NAME = os.getenv("OCR_SERVICE_NAME", "ocr-platform")
SERVICE_VERSION = "1.0"

# Root under which every model gets its own sub-directory.
# Layout: <MODEL_ROOT>/<model-folder>/<model-name>
MODEL_ROOT = os.getenv("OCR_MODEL_ROOT", "/workspace/models")

# Uploaded images are written here and deleted after inference.
UPLOAD_DIR = os.getenv("OCR_UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))

# Model used when a request does not specify one, so requests written against
# either original service keep working unchanged.
DEFAULT_MODEL = os.getenv("OCR_DEFAULT_MODEL", "legal-documents")

HOST = os.getenv("OCR_HOST", "0.0.0.0")
PORT = int(os.getenv("OCR_PORT", "10100"))

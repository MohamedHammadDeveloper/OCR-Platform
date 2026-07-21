"""Configuration for the Arabic/English handwritten OCR model."""

import os

import config as app_config

HF_REPO_ID = os.getenv(
    "HANDWRITTEN_HF_REPO_ID", "sherif1313/Arabic-English-handwritten-OCR-v3"
)

# Directory naming: <MODEL_ROOT>/arabic-handwritten/Arabic-English-handwritten-OCR-v3
PROJECT_NAME = os.getenv("HANDWRITTEN_PROJECT_NAME", "arabic-handwritten")
MODEL_NAME = os.getenv("HANDWRITTEN_MODEL_NAME", "Arabic-English-handwritten-OCR-v3")

MODEL_DIR = os.getenv(
    "HANDWRITTEN_MODEL_DIR",
    os.path.join(app_config.MODEL_ROOT, PROJECT_NAME, MODEL_NAME),
)

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# The repo is otherwise clean — assets/ is just ~30 README screenshots.
# Everything else at root is needed to load the model, including
# chat_template.jinja.
DOWNLOAD_IGNORE_PATTERNS = [
    "assets/*",
]

"""
Model download for Arabic-English-handwritten-OCR-v3.

NOTE: the original project shipped no download script — its ``run.py`` loads
from a pre-populated ``/workspace/models/...`` directory with
``local_files_only=True``. This file therefore mirrors the download code that
already exists in the legal-documents project rather than inventing a new
mechanism.

Runnable standalone:

    python -m services.arabic_handwritten.download_model
"""

import logging
import os

from huggingface_hub import snapshot_download

from services.arabic_handwritten import config

logger = logging.getLogger(__name__)


def is_model_present(model_dir: str = config.MODEL_DIR) -> bool:
    """A model directory counts as usable when config + weights are both there."""
    if not os.path.isdir(model_dir):
        return False
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    return any(name.endswith(".safetensors") for name in os.listdir(model_dir))


def ensure_model_downloaded(
    model_dir: str = config.MODEL_DIR,
    repo_id: str = config.HF_REPO_ID,
) -> str:
    """Guarantee that ``model_dir`` holds a complete local copy of ``repo_id``."""
    if is_model_present(model_dir):
        logger.info("Model already present at %s — skipping download.", model_dir)
        return model_dir

    logger.info("Model not found at %s. Downloading %s ...", model_dir, repo_id)
    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        token=config.HF_TOKEN,
        ignore_patterns=config.DOWNLOAD_IGNORE_PATTERNS,
    )

    if not is_model_present(model_dir):
        raise RuntimeError(
            f"Download of '{repo_id}' finished but {model_dir} does not contain "
            "config.json and *.safetensors."
        )

    logger.info("Model downloaded to %s", model_dir)
    return model_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_model_downloaded()

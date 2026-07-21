"""
Configuration for the Arabic Legal Documents OCR model.

Copied from the original project's config.py; only the storage layout was
re-pointed at the shared per-model root and the env var names were prefixed so
models cannot collide.
"""

import os

import config as app_config

# --------------------------------------------------------------------------- #
# Model identity — the ONLY place the HuggingFace repository is declared
# --------------------------------------------------------------------------- #

HF_REPO_ID = os.getenv("LEGAL_HF_REPO_ID", "bakrianoo/arabic-legal-documents-ocr-1.0")

# Directory naming: <MODEL_ROOT>/legal-documents/arabic-legal-documents-ocr-1.0
PROJECT_NAME = os.getenv("LEGAL_PROJECT_NAME", "legal-documents")
MODEL_NAME = os.getenv("LEGAL_MODEL_NAME", "arabic-legal-documents-ocr-1.0")

MODEL_DIR = os.getenv(
    "LEGAL_MODEL_DIR",
    os.path.join(app_config.MODEL_ROOT, PROJECT_NAME, MODEL_NAME),
)

# Optional token for gated repositories (Gemma is a gated base model).
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# Skip everything in the repo that inference does not need.
#
# The repo ships training artefacts alongside the model: six LoRA checkpoints
# (each with optimizer.pt + rng_state.pth), an __archive/ of an unrelated
# llava finetune, and data/ (a 160MB image zip plus the SFT jsonl). Excluding
# those four directories leaves only the root-level files the model loads:
# config.json, generation_config.json, model-0000{1,2}-of-00002.safetensors,
# model.safetensors.index.json, preprocessor/processor config, the tokenizer
# files and chat_template.jinja.
#
# NOTE: the original project's list excluded "*.safetensors" (the weights),
# "*.jinja" (the chat template apply_chat_template needs) and "*.model", while
# keeping all of the training junk — so its download could never produce a
# loadable model. Corrected here.
DOWNLOAD_IGNORE_PATTERNS = [
    "__archive/*",
    "checkpoints/*",
    "data/*",
    "scripts/*",
]

# --------------------------------------------------------------------------- #
# Inference behaviour
#
# These mirror the original implementation (scripts/eval.py + README.md) and
# should not be changed casually — they alter OCR output.
# --------------------------------------------------------------------------- #

# Default extraction task: "task_1" (markdown content + structural elements)
# or "task_2" (document classification, source, marks, signatures, ...).
DEFAULT_TASK = os.getenv("OCR_DEFAULT_TASK", "task_1")

# README.md marks image preprocessing as mandatory for best OCR results.
DEFAULT_PREPROCESS = os.getenv("OCR_PREPROCESS", "true").lower() in ("1", "true", "yes")

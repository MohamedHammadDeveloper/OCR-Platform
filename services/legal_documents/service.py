"""
Arabic Legal Documents OCR service.

Copied from the original project's model.py. The prompts, ``preprocess_image``,
``parse_json`` and the whole of ``run_ocr`` (message construction, chat
template, generation parameters, trimming and decoding) are unchanged. The only
changes are:

  * the download helpers moved to this package's download_model.py
  * ``load()`` is called lazily on first OCR request instead of at startup,
    and ``unload()`` was added
  * ``run()`` shapes the response exactly as the original app.py did

The original module docstring's provenance notes still apply:

  * ``preprocess_image``  -> README.md ("Mandatory Image Preprocessing")
  * ``TASK_1_MESSAGE`` /
    ``TASK_2_MESSAGE``    -> scripts/eval.py and scripts/split-datasets.py
  * ``parse_json``        -> scripts/eval.py
  * ``run_ocr``           -> scripts/eval.py
"""

import base64
import logging
import os
import threading
from io import BytesIO
from typing import Optional, Union

import json_repair
import torch
from PIL import Image, ImageEnhance
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from services.legal_documents import config
from services.legal_documents.download_model import (
    ensure_model_downloaded,
    is_model_present,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# Original prompts — scripts/eval.py (identical to the finetuning data)
# =========================================================================== #

TASK_1_MESSAGE = "\n".join([
    "You are a professional OCR Details Extractor.",
    "Your rule to extract: the page markdown content in addition to the structural_elements of the document.",
    "Extract the final output into a json format.",
    "Do not generate any introduction or conclusion."
])

TASK_2_MESSAGE = "\n".join([
    "You are a professional OCR Details Extractor.",
    "Your rule to extract the: document_classification, source, physical_properties, official_marks, signatures_authorization, routing_distribution, attachments_references, condition_notes and confidence_quality of the document.",
    "Extract the final output into a json format.",
    "Do not generate any introduction or conclusion."
])

TASK_MESSAGES = {
    "task_1": TASK_1_MESSAGE,
    "task_2": TASK_2_MESSAGE,
}


# =========================================================================== #
# Original helpers — README.md / scripts/eval.py (unchanged)
# =========================================================================== #

def preprocess_image(image_path, max_width=1024, do_enhance=True, return_base64=False):
    image = Image.open(image_path)

    # 1. Convert to grayscale
    gray_image = image.convert('L')

    # 2. Resize maintaining aspect ratio
    if gray_image.width > max_width:
        ratio = max_width / float(gray_image.width)
        new_height = int(gray_image.height * ratio)
        gray_image = gray_image.resize((max_width, new_height), Image.LANCZOS)

    # 3. Enhance contrast
    if do_enhance:
        enhancer = ImageEnhance.Contrast(gray_image)
        gray_image = enhancer.enhance(1.5)

    if return_base64:
        buffered = BytesIO()
        gray_image.save(buffered, format="JPEG", optimize=True, quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"

    return gray_image


def parse_json(text):
    try:
        return json_repair.loads(text)
    except Exception:
        return None


class LegalDocumentsService:
    """Holds the model and processor. Loaded on first use, never at startup."""

    name = "legal-documents"

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.model_dir: Optional[str] = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def is_downloaded(self) -> bool:
        return is_model_present(config.MODEL_DIR)

    def download(self) -> str:
        return ensure_model_downloaded()

    def load(self) -> None:
        """Download if needed, then load model + processor exactly once."""
        if self.is_loaded:
            return

        self.model_dir = ensure_model_downloaded()

        logger.info("Loading model from %s ...", self.model_dir)

        # Loading call preserved from scripts/eval.py.
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_dir, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_dir)

        logger.info("Model loaded on device: %s", self.model.device)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded %s", self.name)

    def device(self):
        if not self.is_loaded:
            return None
        return str(self.model.device)

    # ----------------------------------------------------------------- #
    # Original inference — scripts/eval.py, unchanged
    # ----------------------------------------------------------------- #

    def run_ocr(
        self,
        image_path: str,
        task: str = config.DEFAULT_TASK,
        preprocess: bool = config.DEFAULT_PREPROCESS,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Run the original OCR pipeline against a local image path.

        Returns the raw decoded model output. Structured parsing is left to the
        caller via :func:`parse_json`, mirroring the original script.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        if task not in TASK_MESSAGES:
            raise ValueError(
                f"Unknown task '{task}'. Expected one of {sorted(TASK_MESSAGES)}."
            )

        task_message = TASK_MESSAGES[task]

        # README.md marks this preprocessing step as mandatory for best results.
        image_input: Union[str, Image.Image] = (
            preprocess_image(image_path) if preprocess else image_path
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": task_message}
                ]
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # enable_thinking=False,
        )

        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        with self._lock:
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    # ----------------------------------------------------------------- #
    # Entry point used by the shared /ocr endpoint
    # ----------------------------------------------------------------- #

    def run(self, image_path: str, task=None, preprocess=None, max_new_tokens=None) -> dict:
        """Response shape copied from the original app.py's OCRResponse."""
        text = self.run_ocr(
            image_path,
            task=config.DEFAULT_TASK if task is None else task,
            preprocess=config.DEFAULT_PREPROCESS if preprocess is None else preprocess,
            max_new_tokens=1024 if max_new_tokens is None else max_new_tokens,
        )
        return {
            "success": True,
            "text": text,
            "data": parse_json(text),
        }

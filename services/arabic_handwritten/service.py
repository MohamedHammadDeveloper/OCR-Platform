"""
Arabic/English handwritten OCR service.

The inference code below is copied verbatim from the original project's
``run.py``. The prompt, preprocessing, generation parameters, trimming and
decoding are byte-for-byte identical. The only changes are:

  * ``model`` / ``processor`` module globals became ``self.model`` /
    ``self.processor`` so the model can be loaded lazily and unloaded.
  * the model path comes from this package's ``config.py``.

``_ocr_response`` is copied verbatim from the original project's ``main.py``
so the response the React frontend already consumes is unchanged.
"""

import logging
import os
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from services.arabic_handwritten import config
from services.arabic_handwritten.download_model import (
    ensure_model_downloaded,
    is_model_present,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# Original helper — run.py, unchanged
# =========================================================================== #

def process_vision_info(messages: List[dict]):
    image_inputs = []
    video_inputs = []
    for message in messages:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "image":
                    image = item["image"]
                    if isinstance(image, str):
                        # فتح الصورة مع تحسين الجودة
                        image = Image.open(image).convert("RGB")
                    elif isinstance(image, Image.Image):
                        pass
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                    image_inputs.append(image)
                elif item["type"] == "video":
                    video_inputs.append(item["video"])
    return image_inputs if image_inputs else None, video_inputs if video_inputs else None


# دالة مساعدة لتحسين جودة الصور
def enhance_image_quality(image_path):
    """تحسين جودة الصورة لتحسين دقة OCR"""
    try:
        img = Image.open(image_path)
        # زيادة الدقة إذا كانت الصورة صغيرة
        if max(img.size) < 800:
            new_size = (img.size[0] * 2, img.size[1] * 2)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return img
    except:
        return Image.open(image_path)


class ArabicHandwrittenService:
    """Holds the model and processor. Loaded on first use, never at startup."""

    name = "arabic-handwritten"

    def __init__(self) -> None:
        self.model = None
        self.processor = None

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
        if self.is_loaded:
            return

        model_name = ensure_model_downloaded()

        logger.info("Loading %s from %s ...", self.name, model_name)

        # Loading calls preserved from run.py.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # إعداد المعالج مع دقة أعلى لتحسين دقة OCR
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )

        logger.info("Loaded %s on device: %s", self.name, self.model.device)

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
    # Original inference — run.py, unchanged
    # ----------------------------------------------------------------- #

    def extract_text_from_image(self, image_path):
        try:
            # ✅ استخدام prompt أكثر وضوحًا وطلبا للنص الكامل
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:"},
                    ],
                }
            ]

            # تحضير النص والصور
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            # معالجة المدخلات مع إعدادات محسنة
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # ✅ إعدادات توليد محسنة للنصوص الطويلة
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # زيادة كبيرة لاستيعاب النصوص الطويلة
                min_new_tokens=50,   # الحد الأدنى لضمان عدم القص المبكر
                do_sample=False,      # للحصول على نتائج متسقة
                temperature=0.01,      # توازن بين الإبداع والاستقرار
                top_p=0.1,           # للتنوع المعتدل
                repetition_penalty=1.1,  # منع التكرار
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                num_return_sequences=1
            )

            # استخراج النص المُولَّد فقط (بدون مطالبة المستخدم)
            input_len = inputs.input_ids.shape[1]
            output_text = self.processor.batch_decode(
                generated_ids[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True  # تحسين المسافات
            )[0]

            return output_text.strip()

        except Exception as e:
            return f"حدث خطأ أثناء معالجة الصورة: {str(e)}"

    # ----------------------------------------------------------------- #
    # Response shaping — main.py, unchanged
    # ----------------------------------------------------------------- #

    def _ocr_response(self, text: str) -> dict:
        """Shape the plain OCR text into the shared service response contract.

        This project performs OCR only — no classification, keywords, layout,
        tables or charts — so ``output`` exposes ``full_text`` and nothing else.
        """
        return {
            "success": True,
            "text": text,
            "data": {
                "output": {
                    "full_text": text,
                }
            },
        }

    # ----------------------------------------------------------------- #
    # Entry point used by the shared /ocr endpoint
    # ----------------------------------------------------------------- #

    def run(self, image_path: str, task=None, preprocess=None, max_new_tokens=None) -> dict:
        """
        ``task`` / ``preprocess`` / ``max_new_tokens`` are accepted for request
        compatibility and ignored, exactly as the original main.py did: this
        project is OCR-only and its generation settings are fixed inside the
        inference code above, which must stay unchanged.
        """
        text = self.extract_text_from_image(image_path)
        return self._ocr_response(text)

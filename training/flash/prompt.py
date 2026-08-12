"""
Shared prompt + target schema for Flash fine-tuning and evaluation.

The SAME instruction must be used at training time and inference time, otherwise
the model sees a different task than it was trained on. Import from here in both
build_dataset.py and evaluate.py — never redefine the strings.
"""
import json

# 18 document types -> id is (index + 1). Must match labels/document_types.json.
TYPES = [
    "عريضة / صحيفة", "حافظة مستندات", "محاضر جلسات", "تقرير", "أحكام", "شهادات",
    "إعلام وراثة", "عقد", "صورة تنفيذية", "إيصال", "إنذار", "قرارات / أوامر",
    "طلبات", "مذكرات", "إعلان", "مكاتبات", "إشكال", "توكيل",
]
ENUM_LINE = " · ".join(f"{i+1}={t}" for i, t in enumerate(TYPES))

INSTRUCTION = "\n".join([
    "انسخ هذا المستند القانوني المصري بالكامل والتزم بالقواعد:",
    "- كل النص (مطبوع + خط يد) بترتيب القراءة يمين‑شمال وفوق‑تحت، سطر بصري = سطر.",
    "- الجداول: الأعمدة بعلامة | . الهوامش [هامش: ...]. الأختام [ختم: النص]. التوقيع [توقيع].",
    "- اللي مش واضح: [غير واضح]. ممنوع التخمين. الأرقام عربي-هندي زي الورقة.",
    "ثم حدّد نوع المستند من: " + ENUM_LINE,
    "و subject في سطر واحد، و 5-12 keyword للبحث.",
    "رجّع JSON فقط بالحقول: document_id, document_type, subject, keywords, full_text.",
])

# Fields the model must learn to output (labeling-only fields are dropped).
TARGET_FIELDS = ["document_id", "document_type", "subject", "keywords", "full_text"]


def build_target(record: dict) -> str:
    """The assistant target string = compact JSON of the schema fields."""
    obj = {k: record.get(k) for k in TARGET_FIELDS}
    return json.dumps(obj, ensure_ascii=False)


def user_content_llamafactory() -> str:
    """User turn text for LLaMA-Factory (image placeholder + instruction)."""
    return "<image>\n" + INSTRUCTION


def inference_messages(image_path: str):
    """Messages for transformers/qwen-vl inference (no assistant turn)."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": INSTRUCTION},
        ],
    }]

"""
Run the model on ONE image and print the result — a quick sanity check before
uploading to HF Hub.

Works on either:
  * the merged standalone model  (default: merged/legal-flash-v1), or
  * the base model + LoRA adapter (--model <base> --adapter saves/flash-qwen25vl-3b-lora)

Usage:
  # merged model (after `llamafactory-cli export export_merge.yaml`)
  python predict.py --image dataset/images/OCR2/Alex/2022/5-2022/30040000520220111_p001.png

  # or test base + adapter BEFORE merging
  python predict.py --model sherif1313/Arabic-English-handwritten-OCR-v3 \
      --adapter saves/flash-qwen25vl-3b-lora \
      --image dataset/images/OCR2/Alex/2022/5-2022/30040000520220111_p001.png
"""
import argparse, json

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import json_repair

from prompt import INSTRUCTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="base (default = v1 7B base) or a merged model dir/repo id")
    ap.add_argument("--adapter", default="saves/qwen25vl-7b-lora",
                    help="LoRA on top of --model. Local v1 run by default; or the HF repo "
                         "m-hammad/legal-flash-7b-lora-v1 on a fresh box. Pass '' for base only.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--bits", type=int, default=16, choices=[4, 16])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    if args.bits == 4:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    # Auto class, not a hardcoded one: a hardcoded Qwen2.5 class mis-loads other
    # architectures (Qwen3-VL loses its vision merger weights silently).
    try:
        from transformers import AutoModelForImageTextToText as _cls
    except ImportError:
        from transformers import AutoModelForVision2Seq as _cls
    model = _cls.from_pretrained(args.model, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": args.image},
        {"type": "text", "text": INSTRUCTION}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True,
                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    raw = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    print("=== RAW OUTPUT ===\n" + raw)
    parsed = json_repair.loads(raw)
    print("\n=== PARSED JSON ===")
    if isinstance(parsed, dict):
        for k in ("document_id", "document_type", "subject", "keywords"):
            print(f"{k}: {parsed.get(k)}")
        print("full_text:\n" + str(parsed.get("full_text", "")))
    else:
        print("(model did not return valid JSON — see RAW above)")


if __name__ == "__main__":
    main()

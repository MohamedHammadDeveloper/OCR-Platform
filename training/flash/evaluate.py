"""
Evaluate the fine-tuned Flash model on the held-out HUMAN test gold
(v1_test_gold_human.jsonl, 53 pages). The old v1_test_gold.jsonl is Opus-labeled;
numbers measured against it are NOT comparable to human-ground-truth numbers.

Loads the base handwriting model + the trained LoRA adapter, runs inference on each
test page, and reports:
  - JSON parse rate         (did the model emit valid JSON?)
  - document_type accuracy  (predicted document_id == gold document_id)
  - full_text similarity    (normalized Levenshtein similarity vs gold, 0..1)
  - keyword recall          (fraction of gold keywords whose text appears in prediction)

Usage:
  python evaluate.py \
      --base sherif1313/Arabic-English-handwritten-OCR-v3 \
      --adapter saves/flash-qwen25vl-3b-lora \
      --test dataset/labels/v1_test_gold_human.jsonl \
      --images-root /data/flash/images \
      --out report.json
"""
import argparse, json, os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import json_repair
from rapidfuzz.distance import Levenshtein

from prompt import INSTRUCTION


def load(base, adapter, bits):
    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    if bits == 4:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, **kw)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    return model, proc


def infer(model, proc, image_path, max_new_tokens, max_pixels=0, min_pixels=0):
    image_ele = {"type": "image", "image": image_path}
    if max_pixels:            # cap the image to this many pixels before the vision encoder
        image_ele["max_pixels"] = max_pixels
    if min_pixels:
        image_ele["min_pixels"] = min_pixels
    messages = [{"role": "user", "content": [
        image_ele,
        {"type": "text", "text": INSTRUCTION}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True,
                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")   # v1 base (Apache, working)
    ap.add_argument("--adapter", default="saves/qwen25vl-7b-lora",
                    help="LoRA dir (default = the v1 7B run). Pass '' or a base-only run to eval the base.")
    ap.add_argument("--test", default="dataset/labels/v1_test_gold_human.jsonl")
    ap.add_argument("--images-root", default="dataset/images")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 16])
    ap.add_argument("--max-new-tokens", type=int, default=3072)  # entities lengthen target (~2.1k tok max)
    ap.add_argument("--max-pixels", type=int, default=1048576,
                    help="cap each image to this many pixels at eval. DEFAULT MATCHES the training "
                         "image_max_pixels (1048576). Evaluating at a different budget than training "
                         "is a distribution shift: measured on v4, scoring at full res (~8.7MP) instead "
                         "of 1MP cost 5.7 points of type_accuracy (0.906 -> 0.849) for no text gain. "
                         "Pass 0 for the qwen default (~12.8M ≈ full res).")
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, proc = load(args.base, args.adapter, args.bits)
    recs = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    rows, parsed_ok, type_ok, sims, krecs = [], 0, 0, [], []
    for i, r in enumerate(recs, 1):
        img = os.path.join(args.images_root, r["image"].replace("\\", "/"))
        raw = infer(model, proc, img, args.max_new_tokens, args.max_pixels, args.min_pixels)
        pred = json_repair.loads(raw)
        ok = isinstance(pred, dict)
        parsed_ok += ok
        tacc = ok and pred.get("document_id") == r.get("document_id")
        type_ok += bool(tacc)
        gt = r.get("full_text", "")
        pt = (pred.get("full_text", "") if ok else raw) or ""
        sim = Levenshtein.normalized_similarity(gt, pt)
        sims.append(sim)
        gk = r.get("keywords", []) or []
        kr = (sum(1 for k in gk if k and k in pt) / len(gk)) if gk else None
        if kr is not None:
            krecs.append(kr)
        rows.append({"image": r["image"], "parsed": ok, "type_ok": bool(tacc),
                     "text_sim": round(sim, 3),
                     "gold_id": r.get("document_id"),
                     "pred_id": pred.get("document_id") if ok else None})
        print(f"[{i}/{len(recs)}] type_ok={bool(tacc)} sim={sim:.3f} {r['image'].split('/')[-1]}")

    n = len(recs)
    summary = {
        "n": n,
        "json_parse_rate": round(parsed_ok / n, 3),
        "type_accuracy": round(type_ok / n, 3),
        "text_similarity_mean": round(sum(sims) / n, 3),
        "keyword_recall_mean": round(sum(krecs) / len(krecs), 3) if krecs else None,
        "adapter": args.adapter or "(base model baseline)",
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

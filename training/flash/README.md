# Flash fine-tuning (v1)

Continue-fine-tune the Arabic-handwritten **Qwen2.5-VL-3B**
(`sherif1313/Arabic-English-handwritten-OCR-v3`) on the Egyptian MoJ legal-docs
gold, producing one model that outputs, per page:
`{ document_id, document_type, subject, keywords, full_text }`.

QLoRA (4-bit) via **LLaMA-Factory** (handles Qwen2.5-VL multimodal SFT + correct
label masking). Evaluation is a standalone transformers script.

**Everything ships in this repo** — scripts **and** the v1 data
(`dataset/images/` = 359 pages, `dataset/labels/` = the gold). One `git clone`
on the GPU box gets you everything; no separate data upload.

## GPU
3B QLoRA needs ~10–14 GB VRAM. Any 16 GB+ card works (RTX 4090 / A100 / etc.).

## 1. Clone (first thing on the vast.ai box)
```bash
git clone https://github.com/MohamedHammadDeveloper/OCR-Platform.git
cd OCR-Platform/training/flash
```

## 2. Install
```bash
pip install -r requirements.txt
git clone https://github.com/hiyouga/LLaMA-Factory
pip install -e "LLaMA-Factory[torch,metrics,bitsandbytes]"
```

## 3. Build the SFT dataset (paths already default to the in-repo data)
```bash
python build_dataset.py --val 30
```
Produces `data/flash_train.json` (269), `data/flash_val.json` (30), `data/dataset_info.json`.

## 4. Smoke test first (cheap — catch config/path errors before the real run)
In `qwen2_5vl_lora_sft.yaml` temporarily set `num_train_epochs: 1` and add
`max_samples: 8`, run step 5, confirm it trains + saves, then revert.

## 5. Train
```bash
llamafactory-cli train qwen2_5vl_lora_sft.yaml
```
Adapter saved to `saves/flash-qwen25vl-3b-lora`. ~3 epochs over 269 samples is short
(tens of minutes on a 4090). Watch eval loss on `flash_val`.

## 6. Evaluate on the held-out test — baseline vs fine-tuned
```bash
python evaluate.py --out report_base.json                              # base model, no adapter
python evaluate.py --adapter saves/flash-qwen25vl-3b-lora --out report_ft.json   # fine-tuned
```
Metrics: JSON parse rate, `document_type` accuracy, full_text similarity, keyword recall.
The gap between the two reports is what the fine-tuning actually bought you.

---

## 7. After training — getting the model out (IMPORTANT)

**GitHub is for code, NOT model weights.** Get the trained model off the box via
**Hugging Face Hub** (best) or direct download:

**Option A — push the LoRA adapter to HF Hub (recommended, small ~30–100 MB):**
```bash
huggingface-cli login
huggingface-cli upload <your-hf-user>/flash-qwen25vl-3b-lora-v1 saves/flash-qwen25vl-3b-lora .
```
Version each run as `-v1`, `-v2`, … (or use HF repo revisions/branches).

**Option B — merge to a standalone model, then push:**
```bash
llamafactory-cli export \
  --model_name_or_path sherif1313/Arabic-English-handwritten-OCR-v3 \
  --adapter_name_or_path saves/flash-qwen25vl-3b-lora \
  --template qwen2_vl --finetuning_type lora --export_dir merged/legal-flash
huggingface-cli upload <your-hf-user>/legal-flash-v1 merged/legal-flash .
```
Merged 3B ≈ ~6 GB — HF Hub only, never git.

**Option C — no Hub:** download `saves/flash-qwen25vl-3b-lora/` directly (vast.ai file
browser / `scp` / `rsync`).

### Serving it in the platform
Add a `legal-flash` service to this app (same pattern as `services/arabic_handwritten`,
see PROJECT.md §11): point it at the merged HF repo, or load base + adapter at runtime.
Each new training round → new HF version → bump the service's repo id.

### Where each thing lives (mental model)
| Artifact | Home |
| --- | --- |
| Training/eval **code** | GitHub (this repo) |
| v1 **data** (359 imgs + gold) | GitHub (this repo, `dataset/`) — OK at 114 MB |
| Trained **model** (adapter / merged) | **Hugging Face Hub**, versioned per run |
| v2+ **data** (thousands of imgs, GBs) | **NOT git** → HF Datasets or a bucket / Git LFS |

## Known caveats / next iterations
- **Type coverage gap:** ids 9 (صورة تنفيذية) & 17 (إشكال) ABSENT from train; 7/11/13/16 rare.
  v1 will be weak/blind on these — fix with targeted sampling in v2.
- **~300 samples is a v1 proof.** Expect a stronger model at 1.5–3k+; expand labeling
  (`/flash-resume`, `flash_label_tools.py`) toward the full 7,622 pages.
- Labels are Opus-distilled (no human review yet); a human QC pass raises the ceiling.

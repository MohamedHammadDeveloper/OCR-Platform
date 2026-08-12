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

## 2b. Get the base model (~7.5 GB) — the handwriting checkpoint we continue from
`sherif1313/Arabic-English-handwritten-OCR-v3` (Qwen2.5-VL-3B, **public, not gated**).
Training auto-downloads it on first run, but pre-download it to catch network/space
issues early and to time the ~7.5 GB pull separately from training:
```bash
huggingface-cli download sherif1313/Arabic-English-handwritten-OCR-v3
```
No HF token needed (public). It caches to `~/.cache/huggingface`; the yaml's
`model_name_or_path` then loads from cache. Make sure the box has **~20 GB free**
(base 7.5 GB + 4-bit runtime + adapter/checkpoints).

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

## 6b. Try it on ONE image first (sanity check before uploading)
Eyeball the model on a single page before committing to the merge + HF upload.

Test the **base + adapter** (before merging):
```bash
python predict.py --model sherif1313/Arabic-English-handwritten-OCR-v3 \
  --adapter saves/flash-qwen25vl-3b-lora \
  --image dataset/images/OCR2/Alex/2022/5-2022/30040000520220111_p001.png
```
Or, after you've merged (step 7), test the **merged model**:
```bash
python predict.py --image dataset/images/OCR2/Alex/2022/5-2022/30040000520220111_p001.png
```
Prints the raw output + parsed `document_type / subject / keywords / full_text`. Point
`--image` at any page (in `dataset/images/…`, or your own). Add `--bits 4` if VRAM is tight.

---

## 7. After training — getting the model out (IMPORTANT)

**GitHub is for code, NOT model weights.** Get the trained model off the box via
**Hugging Face Hub** (best) or direct download:

**Option A — push the LoRA adapter to HF Hub (recommended, small ~30–100 MB):**
```bash
huggingface-cli login
huggingface-cli upload m-hammad/flash-qwen25vl-3b-lora-v1 saves/flash-qwen25vl-3b-lora .
```
Version each run as `-v1`, `-v2`, … (or use HF repo revisions/branches).

**Option B — merge to a standalone model, then push (RECOMMENDED for a clean v1 model):**
```bash
# 1) merge adapter into base -> merged/legal-flash-v1  (config: export_merge.yaml)
llamafactory-cli export export_merge.yaml

# 2) verify the export folder has: config.json, *.safetensors, tokenizer files,
#    preprocessor_config.json, chat_template.jinja
ls merged/legal-flash-v1

# 3) log in with a WRITE token (https://huggingface.co/settings/tokens)
huggingface-cli login

# 4) create the repo as PRIVATE (recommended for MoJ data) — website, or:
python -c "from huggingface_hub import create_repo; create_repo('m-hammad/legal-flash-v1', repo_type='model', private=True)"

# 5) upload the merged folder to it
huggingface-cli upload m-hammad/legal-flash-v1 merged/legal-flash-v1 .
```
Note: a bare `huggingface-cli upload` auto-creates the repo but **public** — create it
first (step 4) to keep it private. Merged 3B ≈ ~7.5 GB — HF Hub only, never git. Load it
anywhere with just the repo id (no base, no adapter):
`Qwen2_5_VLForConditionalGeneration.from_pretrained("m-hammad/legal-flash-v1")`.

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

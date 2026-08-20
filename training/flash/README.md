# Flash fine-tuning (v1)

Fine-tune **Qwen2.5-VL-7B-Instruct** (Apache-2.0) on the Egyptian MoJ legal-docs
gold via QLoRA, producing a model that outputs, per page:
`{ document_id, document_type, subject, keywords, full_text }`.

## ✅ STATUS — v1 DONE (continue from here, don't restart)
- **v1 trained + evaluated + uploaded.** Base = `Qwen/Qwen2.5-VL-7B-Instruct`; adapter (private) =
  **`m-hammad/legal-flash-7b-lora-v1`**. On the 60 held-out test: text-sim 0.29→0.46 (+60% rel),
  type-acc 0→45%, JSON 100%.
- **The 3B run (Run A) is ⛔ BLOCKED** — its base won't tie lm_head (loss stuck 11.93). Don't retry.
- **To CONTINUE (v2):** label more of your own data (esp. degraded handwriting) →
  `python build_dataset.py --val 30` → retrain Run B → upload as `...-v2`. Same working recipe;
  only the gold grows. Inference now: `python predict.py --image <page>` (defaults already point at v1).

QLoRA (4-bit) via **LLaMA-Factory** (handles Qwen2.5-VL multimodal SFT + correct
label masking). Evaluation is a standalone transformers script.

**Everything ships in this repo** — scripts **and** the v1 data
(`dataset/images/` = 359 pages, `dataset/labels/` = the gold). One `git clone`
on the GPU box gets you everything; no separate data upload.

## GPU
Run A (3B, **bf16 LoRA**) ~16–24 GB · Run B (7B, 4-bit QLoRA) ~16–24 GB. A 24 GB+ card
(RTX 4090 / A100 / 5090) is comfortable for both.

> **Troubleshooting — loss stuck at 11.93 (= ln(vocab), grad_norm 0, model learns nothing):**
> the 3B base has `tie_word_embeddings=true` and bnb **4-bit breaks the lm_head tie** → random
> output head. Fix: train the 3B in **bf16 (no `quantization_bit`)** — already set in Run A's config.
> The 7B base is untied (lm_head saved) so 4-bit is fine there.


## ▶️ v4 (2026-08-20): 7x data, Gemini-labeled, human test set
Gold = `dataset/labels/labels_resolved.jsonl` (5694 pages, in git). Build the SFT data
ON THE BOX (`data/` is gitignored — deterministic rebuild, seed 42):
```bash
python build_dataset.py --gold dataset/labels/labels_resolved.jsonl     --images-root dataset/images --out-dir ./data --val 150 --no-dedup
```
(`--no-dedup` is correct: labels_resolved is already md5-unique.)
Expect: flash_train 4685 / flash_val 150. Images are NOT in git — unzip
`v4_images_bundle.zip` so its `images/` lands at `training/flash/dataset/images/`
(it also carries `labels/labels_resolved.jsonl` + `labels/v1_test_gold_human.jsonl`).

On the GPU box, from `training/flash/`:
```bash
git pull
unzip -o /path/to/v4_images_bundle.zip -d dataset/
llamafactory-cli train runs/qwen25vl-7b/lora_sft.yaml
python evaluate.py --adapter saves/qwen25vl-7b-lora     --test dataset/labels/v1_test_gold_human.jsonl     --images-root dataset/images --out report_v4_human.json
# baseline the OLD adapters on the SAME human test for a fair curve:
python evaluate.py --adapter m-hammad/legal-flash-7b-lora-v3     --test dataset/labels/v1_test_gold_human.jsonl --out report_v3_human.json
python compare_reports.py report_v3_human.json report_v4_human.json
```
⚠️ Evaluate v4 AND v3 against `v1_test_gold_human.jsonl` (53 pages, human ground truth).
Old `report_v3*.json` numbers were measured against Opus labels — NOT comparable.
Upload: `m-hammad/legal-flash-7b-lora-v4` (private). Remember `unset HF_TOKEN` first.

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

## 2b. Get the base model (~16 GB, Apache-2.0, public)
```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct
```
No token needed. Caches to `~/.cache/huggingface`. Free space: **~35 GB**.

## 3. Build the SFT dataset (paths already default to the in-repo data)
```bash
python build_dataset.py --val 30
```
Produces `data/flash_train.json` (269), `data/flash_val.json` (30), `data/dataset_info.json`.

## The runs (run from `training/flash/`)
| Run | Config | Base | Status |
| --- | --- | --- | --- |
| **A (3B)** | `runs/qwen25vl-3b/lora_sft.yaml` | sherif v3 (Qwen2.5-VL-3B) | ⛔ **BLOCKED** — lm_head won't tie (loss 11.93). Don't run. |
| **B (7B)** | `runs/qwen25vl-7b/lora_sft.yaml` | Qwen2.5-VL-7B-Instruct | ✅ **WORKING — this is v1.** Apache-2.0. |

Run B is the model. Retrain it on the growing gold for each new version.

## 4. Smoke test first (cheap — catch config/path errors before the real run)
In the chosen run's yaml temporarily set `num_train_epochs: 1` and add `max_samples: 8`,
run step 5, confirm it trains + saves, then revert.

## 5. Train (Run B = the 7B model)
```bash
llamafactory-cli train runs/qwen25vl-7b/lora_sft.yaml   # -> saves/qwen25vl-7b-lora
```
~3 epochs over the current gold is short (tens of minutes). Watch eval loss on `flash_val`.
For v2, just regrow the gold (label more) and re-run this — bump the HF version on upload.

## 6. Evaluate on the held-out test (defaults already point at the 7B v1)
```bash
python evaluate.py --out report_7b_ft.json                 # fine-tuned (default adapter)
python evaluate.py --adapter "" --out report_7b_base.json  # base only (compare gain)
```
Metrics: JSON parse rate, `document_type` accuracy, full_text similarity, keyword recall.

## 6b. Try it on ONE image (defaults = 7B base + v1 adapter)
```bash
python predict.py --image dataset/images/OCR2/Alex/2022/5-2022/30040000520220111_p001.png
```
On a fresh box (no local `saves/`) pull the uploaded adapter instead:
`python predict.py --adapter m-hammad/legal-flash-7b-lora-v1 --image <page>`.
Add `--bits 4` if VRAM is tight.

---

## 7. After training — getting the model out (IMPORTANT)

**GitHub is for code, NOT model weights.** Get the trained model off the box via
**Hugging Face Hub** (best) or direct download:

**Option A — push the LoRA adapter to HF Hub (recommended, small ~50–100 MB). v1 already done:**
```bash
hf auth login   # paste your WRITE token when prompted (never in chat/command history)
python -c "from huggingface_hub import create_repo; create_repo('m-hammad/legal-flash-7b-lora-v1', repo_type='model', private=True)"
hf upload m-hammad/legal-flash-7b-lora-v1 saves/qwen25vl-7b-lora .
```
Version each retrain as `-v2`, `-v3`, …

**Option B — merge to a standalone model, then push (RECOMMENDED for a clean v1 model):**
```bash
# 1) merge adapter into base -> merged/legal-flash-v1  (config: export_merge.yaml)
llamafactory-cli export export_merge.yaml

# 2) verify the export folder has: config.json, *.safetensors, tokenizer files,
#    preprocessor_config.json, chat_template.jinja
ls merged/legal-flash-v1

# 3) log in with a WRITE token (https://huggingface.co/settings/tokens)
hf auth login

# 4) create the repo as PRIVATE (recommended for MoJ data) — website, or:
python -c "from huggingface_hub import create_repo; create_repo('m-hammad/legal-flash-v1', repo_type='model', private=True)"

# 5) upload the merged folder to it
hf upload m-hammad/legal-flash-v1 merged/legal-flash-v1 .
```
Note: a bare `hf upload` auto-creates the repo but **public** — create it
first (step 4) to keep it private. Merged 7B ≈ ~16 GB — HF Hub only, never git. Load it
anywhere with just the repo id (no base, no adapter):
`Qwen2_5_VLForConditionalGeneration.from_pretrained("m-hammad/legal-flash-v1")`.

**Option C — no Hub:** download `saves/qwen25vl-7b-lora/` directly (vast.ai file
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

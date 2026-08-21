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

**✅ v4 RESULT (2026-08-21, 53-page HUMAN test, both adapters scored on the same file):**

| adapter | json | type_acc | text_sim | kw_recall |
|---|---|---|---|---|
| v3 (`legal-flash-7b-lora-v3`) | 1.000 | 0.528 | 0.561 | 0.405 |
| **v4 (`legal-flash-7b-lora-v4`)** | 1.000 | **0.906** | **0.710** | 0.455 |
| delta | — | **+0.378** | **+0.149** | +0.050 |

(v4 scored at MATCHING 1.05M pixels — `report_v4_matched.json`. Scoring the same adapter at
full res gave 0.849 / 0.712 / 0.479: same text, **5.7 fewer points of type accuracy**.
`evaluate.py --max-pixels` now defaults to the training budget so this cannot recur.)

Per kind (text_sim v3→v4): مختلط 0.534→0.720 · مطبوع 0.674→0.798 · خط يد 0.186→0.217 (n=4).
Page-level: v4 better on 43/53, worse on 1, tie 9. Pages <0.4: 18→5. Train 5h40m, eval_loss 0.3465.
Reports: `report_v3_human.json`, `report_v4_human.json`. **The data plateau was a labels problem.**
Gold = `dataset/labels/labels_resolved.jsonl` (5694 pages, in git). Build the SFT data
ON THE BOX (`data/` is gitignored — deterministic rebuild, seed 42):
```bash
python build_dataset.py
```
(defaults now point at labels_resolved.jsonl with --val 150; a bare run was verified
byte-identical to the explicit-flags build. The gold is already md5-unique.)
Expect: flash_train 4685 / flash_val 150. Images are NOT in git — unzip
`v4_images_bundle.zip` so its `images/` lands at `training/flash/dataset/images/`
(it also carries `labels/labels_resolved.jsonl` + `labels/v1_test_gold_human.jsonl`).

On the GPU box, from `training/flash/`:
```bash
git pull
unzip -o /path/to/v4_images_bundle.zip -d dataset/
ls dataset/images | head          # sanity: OCR1/ OCR2/ should be there
python build_dataset.py           # MUST print: flash_train 4685 / flash_val 150
llamafactory-cli train runs/qwen25vl-7b/lora_sft.yaml
python evaluate.py --adapter saves/qwen25vl-7b-lora     --test dataset/labels/v1_test_gold_human.jsonl     --images-root dataset/images --out report_v4_human.json
# baseline the OLD adapters on the SAME human test for a fair curve:
python evaluate.py --adapter m-hammad/legal-flash-7b-lora-v3     --test dataset/labels/v1_test_gold_human.jsonl --out report_v3_human.json
python compare_reports.py report_v3_human.json report_v4_human.json
```
⚠️ Evaluate v4 AND v3 against `v1_test_gold_human.jsonl` (53 pages, human ground truth).
Old `report_v3*.json` numbers were measured against Opus labels — NOT comparable.
Upload: `m-hammad/legal-flash-7b-lora-v4` (private). Remember `unset HF_TOKEN` first.

## ▶️ Run C: Qwen3-VL-8B base experiment (`runs/qwen3vl-8b/lora_sft.yaml`)
Controlled A/B vs v4 — only the base model, template and output_dir differ. Post-v4
diagnostics ruled out eval resolution and generation truncation, leaving the frozen vision
encoder as the remaining bottleneck; Qwen3-VL's headline upgrade is a newer vision encoder.
```bash
python -c "from llamafactory.data.template import TEMPLATES; print('qwen3_vl' in TEMPLATES)"
hf download Qwen/Qwen3-VL-8B-Instruct
llamafactory-cli train runs/qwen3vl-8b/lora_sft.yaml
python evaluate.py --base Qwen/Qwen3-VL-8B-Instruct --adapter saves/qwen3vl-8b-lora     --out report_qwen3vl_human.json
python compare_reports.py report_v4_matched.json report_qwen3vl_human.json
```
**RESULT (2026-08-21): Qwen3-VL-8B LOST.** type 0.774 / text_sim 0.592 / kw 0.391 vs
v4's 0.906 / 0.710 / 0.455. Not a uniform gap: p90 is identical (0.955 vs 0.954) and it is
page-level better on 16 / worse on 23 / tied on 14. The mean is dragged down by **7 pages at
text_sim < 0.1** (v4 has 1) — and 5 of those 7 still got document_type RIGHT, so the model
read the page but emitted a broken `full_text`. Notable counter-signal: it BEAT v4 on
handwriting, 0.200 → 0.347. **Verdict: v4 stays the production model.** Chasing the 7
failures would at best reach parity, so the vision-tower experiment (Run D) is the better bet.

## ▶️ Run D: trainable vision tower (`runs/qwen25vl-7b/lora_sft_vision.yaml`)
The surviving hypothesis. v1-v4 all froze the vision encoder, so it never learned Arabic
script — which explains why extra pixels bought nothing, and why the newer encoder in
Qwen3-VL was the one thing that helped handwriting.
The config pushes checkpoints to the Hub and logs to W&B, so **log in to both BEFORE
launching** or the first push fails several minutes in:
```bash
unset HF_TOKEN && hf auth login     # -> m-hammad/legal-flash-7b-vision (private)
wandb login                         # -> run_name flash-v5-vision-tower
llamafactory-cli train runs/qwen25vl-7b/lora_sft_vision.yaml
python evaluate.py --adapter saves/qwen25vl-7b-lora-vision --out report_vision_human.json
python compare_reports.py report_v4_matched.json report_vision_human.json
```
Resuming after a dead box: add `resume_from_checkpoint: <path-or-hub-checkpoint>` to the yaml.
An earlier attempt reached step 500/1758 before the host went into maintenance; its eval_loss
beat v4 at every checkpoint from 200 on (delta -0.013 -> -0.023), which is why this run exists.
**Beat this:** v4 @1.05M px → type 0.906 / text_sim 0.710 / kw 0.455.

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
python build_dataset.py
```
v4 defaults: labels_resolved.jsonl, --val 150 -> flash_train 4685 / flash_val 150.
(The v1 run was `--gold dataset/labels/v1_gold.jsonl --val 30` -> 269/30.)

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

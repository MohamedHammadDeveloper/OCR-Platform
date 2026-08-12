---
description: Resume the Flash (MoJ OCR) project — continue from v1 (do NOT restart), using memory + gold state
---

You are resuming the **Flash / Ministry-of-Justice OCR fine-tuning** project. **v1 is already DONE**
(trained + uploaded). Do NOT re-do setup or retry the 3B base — continue toward v2. Do this:

> **State (2026-08-12):** base = **Qwen/Qwen2.5-VL-7B-Instruct** (Apache); v1 adapter (private) =
> **`m-hammad/legal-flash-7b-lora-v1`**; test: text-sim 0.29→0.46, type 0→45%, JSON 100%.
> The **3B run is ⛔ BLOCKED** (lm_head won't tie → loss 11.93) — never retry it.
> Training pkg + working recipe: `training/flash/` (README has the current commands).

## 1. Load context (read these before acting)
Read the project memory so you know the plan and decisions:
- `C:\Users\Mohamed Hammad\.claude\projects\D--Projects-Python-OCR-Platform\memory\MEMORY.md`
- and the Flash memory files it points to — especially `flash-progress.md` (LIVE status),
  `flash-labeling-pipeline.md`, `flash-pipeline-design.md`, `flash-dataset-nature.md`,
  `flash-transcription-guidelines.md`.

Ground truth for progress is on disk, not memory — check it:
```
python "E:/Work/Namaa/Flash/labels/flash_label_tools.py" status
```

## 2. Report where we are
Tell the user: v1 is trained+uploaded (7B); gold labeled so far (run `status`); and that the
next step is v2 — label MORE (esp. degraded handwriting) then retrain the 7B. Confirm before spending tokens.

## 3. Continue labeling (batches of 50, ~1.8M Opus tokens each — confirm each batch)
- Prep the next batch of un-labeled TRAIN images:
  ```
  python "E:/Work/Namaa/Flash/labels/flash_label_tools.py" prep-next 50
  ```
  (use `prep-test 60` once train is done, to label the held-out test set).
- Launch the labeling workflow, passing that JSON array as `args`:
  `Workflow({ scriptPath: "E:/Work/Namaa/Flash/labels/workflow_label_opus.js", args: <the array> })`
- When it completes, its result is in the task `.output` file. Merge it:
  ```
  python "E:/Work/Namaa/Flash/labels/flash_label_tools.py" merge "<path-to-task.output>"
  ```
- Give a short batch card (types, content_kind mix, avg length, unclear markers).

## 4. ⚠️ Keep memory up to date (required)
After each batch, UPDATE `flash-progress.md` counts + "Next action". Stale memory breaks the next
resume. If the method or plan changed, update the relevant memory file too.

## v2 = retrain on the grown gold (base stays 7B)
After labeling more, on the GPU box: `git pull` → `python build_dataset.py --val 30` →
`llamafactory-cli train runs/qwen25vl-7b/lora_sft.yaml` → `python evaluate.py --out report.json`
→ upload the new adapter as `m-hammad/legal-flash-7b-lora-v2`. Same recipe; only the gold grows.
Also worth trying (research-backed levers): effective resolution (max_pixels sweep / detect→crop for
hard pages) and a decoupled text-stage classifier. See `training/flash/RESEARCH_*.md`.

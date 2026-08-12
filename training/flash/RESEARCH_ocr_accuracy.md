# Reaching specialized-Arabic-OCR accuracy — research + plan

Question we chased: a specialized Arabic-OCR website read a degraded national-ID and a
dense old newspaper **much better and faster** than Claude Opus on the raw image. Why,
and how do we reach/beat it? (Multi-angle research + a skeptic pass + a local empirical test.)

## Honest diagnosis — why the site won (mechanism, not magic)

1. **Effective resolution is the #1 cause.** A VLM (Claude/Qwen zero-shot) downscales the
   **whole page** inside its vision encoder; at ~200 dpi, thin Arabic strokes/diacritics fall
   below the legibility threshold. Specialized pipelines recognize **native-resolution line/
   region crops**, so they see detail we throw away. This is exactly why the ID and the dense
   newspaper failed.
2. **Modular assembly line** (ABBYY / Google / Azure / Paddle share it):
   `light restore → text DETECT (DBNet/CRAFT) → deskew → layout + RTL reading order →
   small Arabic recognizer (SVTR/CRNN+CTC) per line → LM/dictionary post-correct → field validate`.
3. **Small non-autoregressive recognizer** (5–35M params, CTC) decodes lines in parallel →
   why it was **faster**; a VLM emits token-by-token.
4. **Constrained decoding + field validation** (e.g. Egyptian NID = 14 digits: century/DOB/
   governorate/serial) → looks "smart" because it can't invent implausible strings.

**It did NOT win via a generic CLAHE/binarization filter** (that *hurt* our smudged ID), nor
because "Claude can't read Arabic."

### Empirical check (run locally): a generic engine is NOT the answer
PaddleOCR 2.7.3 (Arabic) on our pages: **failed the degraded ID entirely** (garbage), and on a
**clean** receipt returned Arabic with **letters reversed** (`دعوى→ىوعد`) + errors. So a generic
dedicated engine is *worse than our Opus/Qwen approach*. The site uses something genuinely strong
(top commercial API or a SOTA model), not an off-the-shelf engine — **we already beat generic OCR.**

## The real levers, reordered by ROI (after the skeptic pass)

> Correction to first-pass thinking: cranking `max_pixels` alone does **not** fix effective
> resolution — the vision-token budget caps out before dense pages become line-legible. The
> actual fix is detect→crop→VLM. And "fine-tuning closes the gap" is a **hypothesis to prove on
> our gold**, not a cited fact (the CER numbers found online mix datasets/units).

0. **Measure our document mix first** (empty %, printed/mixed/handwritten split, dpi buckets).
   Cheapest step; the whole router/scale ROI depends on it.
1. **Detector → native-res line/region crops → VLM recognizes the crops.** Keeps our unified VLM
   but feeds it native resolution — the true fix for dense ID/newspaper. Pair with a
   `min_pixels/max_pixels` sweep (Qwen is resolution-sensitive; pin the value). **Highest ROI.**
2. **Fine-tune on our data (in progress) + more training signal:** Augraphy degradation
   augmentation (stamps-over-text, bleed-through, fade, skew, downscale-to-130dpi) **and**
   synthetic rendered Arabic-legal pages (corpus × font bank × degradation) — far more signal
   than a few hundred clean scans.
3. **Field priors + validation:** 18-type lexicon, governorate/court dictionaries, NID format,
   case-number (رقم/حرف/سنة). Treat the NID check-digit as a *soft* signal (algorithm unpublished).
4. **Confidence-driven hard-example mining (data flywheel):** use logprob confidence not just to
   escalate, but to auto-select low-confidence pages for labeling → retrain. Cheapest path to
   closing the handwritten gap over time.
5. **Scale/serving:** router (blank/dup drop → cheap fast-path for clean print → VLM only for
   mixed/handwritten); **distill the tuned 3B into the print fast-path** (matches our gold, unlike
   bolting on PaddleOCR/Surya); FP8 + vLLM + pixel cap. Template/near-dup routing for high-volume
   structured docs (ID cards, standard forms) — fixed field-crops, skip the VLM on the easy slice.

**Excluded (proven or near-certain losers on our data):** blanket CLAHE, global binarizers
(DocEnTr/Sauvola), heavy SR/diffusion (DocRes/NAF-DPM ~45s/img + hallucinates letter shapes).
Redundant with Augraphy-130dpi training anyway.

**Guardrails:** pure-text post-correction hallucinates on legal names/numbers — only gated,
multimodal, low-confidence, and **never allowed to alter a validated field**. No-guess `[غير واضح]`
stays. Add a **critical-field** char-error metric (catches legible-but-wrong names/numbers, the
dangerous legal failure). And a **gold QC / label-noise** number — our decision gates are
meaningless without it.

## How we'll KNOW — benchmark on our hard pages

- **Sample:** 150–300 pages, stratified by dpi bucket × content_kind × noise type (clean, stamp-
  over-text, bleed-through, faded, skew) × held-out source (Giza + Alex2017) + easy controls.
  Gold = our locked layout-preserving transcriptions.
- **Contenders:** the specialized site (the bar) · cloud APIs as teachers/ceiling (Google Vision-ar,
  Azure Read-ar, Gemini-2.0-Flash) · open specialized (Surya, PaddleOCR-ar, Tesseract-ara) · general
  (Claude Opus raw, GPT-4o) · **us: Qwen2.5-VL-3B base vs fine-tuned, each at 2–3 pixel budgets**.
- **Metrics:** normalized Arabic CER/WER (jiwer + CAMeL normalizer: NFC, alef/hamza, ya/alef-maqsura,
  ta-marbuta, Hindi↔Latin digits, strip tatweel) **with and without diacritics**, **per bucket**;
  field accuracy/F1; hallucination rate; efficiency (p50/p95 latency, pages/GPU-min, $/1k).
- **Three ablations that diagnose "why it won" before spending:** (1) our model low vs high pixel /
  crop-vs-fullpage → isolates resolution; (2) raw vs CLAHE → confirms preprocessing hurts;
  (3) base vs fine-tuned → isolates domain training.
- **Decision gate:** fine-tuned 3B meets/beats the site's CER on our **mixed + handwritten** buckets
  (our edge), small margin on print; field accuracy clears DMS thresholds (type ≥95%, case-no ≥90%);
  $/1k viable at scale.

## Bottom line
The gap is **mechanistic and expected**, not magic — and reachable. We already beat generic OCR
engines. The single biggest move is **detect→crop→VLM at native resolution**; the second is
**more/synthetic degraded training data**; the rest is priors, a data flywheel, and a scale router.
Prove it with the benchmark above rather than assuming.

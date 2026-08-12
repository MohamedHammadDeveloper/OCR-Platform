# أفضل نموذج OCR للخط العربي اليدوي المتدهور — ترتيب حاسم وصادق

> السياق: مسح ضوئي حقيقي متدهور (~200dpi)، مطبوع + خط يد عربي مختلط، ضوضاء (أختام، شفافية خلفية، بهتان). المطلوب لكل صفحة: `full_text` (يحافظ على التخطيط) + `document_type` (20 فئة) + `fields`. تدريب QLoRA وخدمة على نطاق 10^8–10^9 صورة.

---

## الحقيقة القاسية أولاً (قبل أي ترتيب)

**لا يوجد نموذج — مفتوح أو تجاري — يقرأ الخط العربي اليدوي المتدهور بشكل جيد جاهزاً.** كل الأرقام اللامعة مضللة لحالتنا:

| الرقم المُعلن | الواقع |
|---|---|
| sherif1313 v3 "CER 1.78%" | على شرائح اختبار نظيفة/اصطناعية، **ليست** MoJ متدهورة |
| QARI "CER 0.061" | على **مطبوع** اصطناعي غزير التشكيل — ليس خط يد |
| AIN-7B "KHATT CER 0.07" | KHATT = نماذج حديثة نظيفة أحادية الكاتب |
| Gemini-2.0-Flash "CER 0.19" (الأفضل تجارياً) | ~1 من كل 5 حروف خطأ، وعلى بيانات نظيفة نسبياً |

المؤشر الأصدق المتاح علناً هو **Muharaf** (صفحات عربية مخطوطة متدهورة حقيقية، أقرب تناظر لبياناتنا): حتى أفضل نموذج عربي (AIN-7B) يسجل **CER ~0.61 (61%)** عليها. هذه هي الفجوة الحقيقية التي نواجهها. أي ادعاء لم يُقَس على gold الخاص بنا = تسويق.

---

## 1) القائمة المختصرة المرتبة (Top 5)

### 🥇 #1 — Qwen2.5-VL-3B (قاعدتنا الحالية عبر sherif1313 v3)
- **الحجم/الرخصة:** 3B dense · Apache-2.0-family (استخدام حكومي/تجاري OK)
- **لماذا الأول:** الـ vision encoder بدقة ديناميكية أصلية (native dynamic resolution) هو العامل الحاسم الوحيد — يحافظ على الحروف العربية الرفيعة على صفحات كثيفة عند 200dpi حيث تفشل الـ encoders ثابتة الدقة. أرخص خيار على نطاق 10^8–10^9، **يقرأ بالفعل**، ويحتفظ بالـ fine-tune الموجود. كامل منظومة العربية المفتوحة (AIN, QARI, sherif) مبنية على Qwen2/2.5-VL.
- **بصراحة:** الأرقام المُعلنة عليه اصطناعية. القيمة الحقيقية تأتي من الـ fine-tune على بياناتنا + قصّ الأسطر بالدقة الأصلية، لا من القاعدة نفسها.

### 🥈 #2 — Qwen3-VL-8B / 4B (Instruct، أكتوبر 2025)
- **الحجم/الرخصة:** 4B و 8B dense · Apache-2.0-family · مدعوم بـ QLoRA/4-bit (Unsloth مؤكّد)
- **لماذا:** الجيل الأحدث، OCR أصلي محسّن صراحةً عبر 32 لغة تشمل العربية، معالجة تخطيط أفضل. أقوى مرشح **صعودي** إذا أثبت تفوقه.
- **بصراحة:** **لا يوجد أي benchmark مستقل** على الخط العربي المتدهور (جديد جداً — KITAB-Bench/AIN تسبقه). الصعود محتمل لكنه **غير مُثبت**. الترحيل يعني التخلي عن fine-tune الخط اليدوي الحالي وإعادة تعليم القراءة من الصفر.

### 🥉 #3 — Qwen2.5-VL-7B (سقف الجودة لنفس العائلة)
- **الحجم/الرخصة:** 7B dense · Apache-2.0-family · QLoRA يتّسع في 32GB
- **لماذا:** نفس المعمار، سعة أكبر للتخطيطات الفوضوية والخط الغامض. هو الاختبار الأمين لسؤال "هل يترك 3B دقة على الطاولة؟" — بلا أي عمل pipeline جديد.
- **بصراحة:** يُتبنّى فقط إذا تفوّق على 3B-المُدرَّب بهامش يبرّر ضعف تكلفة الخدمة (~2–2.5x) على نطاقنا. مرشح ممتاز لـ router من طبقتين (7B للصفحات الصعبة، 3B للباقي).

### #4 — AIN-7B (MBZUAI، مبني على Qwen2-VL-7B)
- **الحجم/الرخصة:** 7B · ⚠️ **رخصة بحثية — تحقّق قبل الإنتاج** (نماذج MBZUAI غالباً بحثية/غير تجارية = قد تمنع نشر MoJ)
- **لماذا:** أفضل VLM عربي مُوثّق على الخط اليدوي (KITAB-Bench SOTA، يتفوق على GPT-4o/Gemini على مجموعات نظيفة).
- **بصراحة:** على Muharaf المتدهورة ما زال CER ~0.61. KHATT-نظيف ≠ MoJ-متدهور. مفيد كـ **warm-start أو teacher للتقطير**، ليس بطلاً مُثبتاً على المتدهور. encoder الرؤية جيل أقدم من 2.5-VL.

### #5 — QARI-OCR (Qwen2-VL-2B) — **للمسار المطبوع فقط**
- **الحجم/الرخصة:** 2B · تحقّق من الريبو
- **لماذا:** SOTA مفتوح على العربي **المطبوع** (CER 0.061). ممتاز كـ fast-path للنصف المطبوع من الصفحات في الـ router.
- **بصراحة:** خط اليد وُصف فقط بـ"قدرات أولية واعدة". **ليس** حلاً للخط اليدوي.

---

## ❌ استبعادات صريحة

- **sherif1313 Arabic-Qwen3.5-OCR-v4 (0.8B):** لا تتبنّاه كقاعدة خط يد. اصطناعي/مطبوع-غزير، والبطاقة نفسها تعترف بـ CER 5–25% على الخط المعقّد، وسعته 0.8B أصغر من أن تحمل ICR على صفحات متدهورة. ("Qwen3.5-0.8B" ليس إصداراً رسمياً من Qwen — التسمية تسويقية.) على الأكثر: مرشح fast-path مطبوع بعد القياس.
- **dots.ocr / DeepSeek-OCR / PaddleOCR-VL / GOT-OCR2:** محلّلات مستندات **مطبوعة**/تخطيط، ليست handwriting-first. جيدة للأغلبية المطبوعة في router، خاطئة لمسار الخط اليدوي.
- **Chandra/Chandra-2 (~9B):** قوي على الخط اليدوي لكن رخصة OpenRAIL-M (لا استخدام تنافسي مع API، سقف <$2M) **تجعله غير صالح** كقاعدة إنتاج على نطاقنا. مرجع/teacher فقط.
- **Amazon Textract** (لا عربية إطلاقاً)، **ABBYY** (OCR عربي مطبوع نعم، لكن ICR عربي غير مدعوم صراحةً — أرقام فقط)، **i2OCR** (مخرجات خط يد = حروف بلا معنى + endpoint عام غير مقبول لبيانات حكومية).

---

## 2) الحكم الصريح: ابقَ على Qwen2.5-VL-3B

**ابقَ على Qwen2.5-VL-3B كافتراضي إنتاجي.** لا تنتقل الآن إلى Qwen3-VL ولا 7B ولا v4.

- **لماذا لا v4:** أصغر وأضعف، معترف بضعفه على الخط اليدوي — تراجع، ليس تقدماً.
- **لماذا لا الانتقال الأعمى لـ Qwen3-VL/7B:** صعودهما **غير مُثبت** على بياناتنا، والانتقال يكلّف التخلي عن fine-tune حالي يعمل. القرار يُبنى على قياس، لا تسويق.
- **متى تنتقل:** فقط إذا أظهر benchmark مُقسّم-طبقياً (stratified) على gold المتدهور الخاص بك أن المُتحدّي يتفوق على 3B-المُدرَّب بهامش يبرّر تكلفته على نطاق 10^8–10^9.

**الخطة العملية:** ابقَ على 3B، وأضِف ثلاثة متحدّين إلى الـ benchmark عند ميزانيات بكسل مختلفة: **Qwen2.5-VL-7B** (سقف نفس العائلة)، **Qwen3-VL-4B/8B** (OCR أحدث، صعود حقيقي لكن غير مُثبت عربياً)، **AIN-7B** (أفضل VLM عربي — بشرط الرخصة).

---

## 3) دور الـ APIs التجارية: teacher / سقف / benchmark — لا خدمة

- **الترتيب الصادق على الخط اليدوي:** Gemini 2.5 (الأفضل، CER ~0.19 على النظيف) > GPT-4o/GPT-5 (CER ~0.45) >> هاوية >> محرّكات OCR المخصّصة (Azure CER 0.83 = الأسوأ رغم "دعمه الرسمي للعربية").
- **الاقتصاد يقتلها للخدمة:** VLM APIs ~$3–5 لكل 1000 صفحة → $0.3M–$5M لـ 100M–1B صفحة. مستحيل على نطاقنا.
- **الاستخدام الصحيح:** بالضبط كما تفعل مع Claude-as-labeler — **معلّم ثانٍ + فحص تعارض (QC)**. أضِف **Gemini 2.5 Pro** لأصعب الخط اليدوي و **Flash** للحجم، كمُصنِّف ثانٍ ضد Claude. استخدم **Mistral OCR أو Google Document AI** الرخيصة (~$1–2/1000) للطبقة **المطبوعة** فقط.
- **الخصوصية الحكومية:** حاوية Azure Document Intelligence المعزولة (air-gapped) هي الأبرز للنشر الخاص — لكن الأضعف عربياً على الخط اليدوي، فتنفع المطبوع فقط. Gemini معزول فقط عبر Google Distributed Cloud (صفقة سيادية ثقيلة).

---

## 4) الأهم هنا: بيانات الـ Fine-tune، وليس القاعدة

**بيانات الـ fine-tune تفوق اختيار القاعدة — ضمن معمار كفء.** الدليل:
- Qwen2-VL-2B مُدرَّب (QARI) يتفوق على GPT-4o على المطبوع العربي.
- AIN-7B مُدرَّب يتفوق على GPT-4o/Gemini على الخط اليدوي.
- sherif 3B مُدرَّب يتفوق على Google Vision.

القاعدة تحدّد **السقف فقط** (encoder دقة أصلية + سعة لحسم الحروف الرفيعة). Qwen2.5-VL و Qwen3-VL كلاهما يؤهّل؛ نماذج <1B أو منخفضة الدقة لا تؤهّل. **اختر أي قاعدة برؤية أصلية قوية، ثم اربح على البيانات.**

**أكبر رافعة ليست تبديل القاعدة**، بل: (أ) **الدقة الفعّالة** — detect → قصّ أسطر بالدقة الأصلية (VLM يُصغّر الصفحة كاملة فتختفي الحروف الرفيعة عند 200dpi)، و(ب) **حجم بيانات خط اليد القانوني المصري الحقيقي المُعنوَن**. لا تبديل قاعدة يصلح هذا وحده.

**مجموعات البيانات:**
- خط يد حقيقي (أولوية): **Muharaf** (NeurIPS 2024، 1600+ صفحة متدهورة، 36k سطر، مراسلات قانونية — أقرب تطابق؛ للتدريب **والتقييم**) و **KHATT** (نماذج حديثة أنظف؛ baseline للتقييم).
- المطبوع + التخطيط (اصطناعي): **Cross-Lingual SynthDocs** (2.5M) و **SARD** (843k)، + رَندَرة قوالب قانونية مصرية + Augraphy لمحاكاة التدهور. تحذير: الاصطناعي يغطي المطبوع/التخطيط لا ICR الحقيقي — تلك الفجوة تُغلق فقط ببيانات خط يد MoJ حقيقية (دولاب gold الخاص بك).

---

## 5) الخطوة التالية الملموسة

1. **لا تبدّل القاعدة الآن.** ابقَ على Qwen2.5-VL-3B إنتاجياً.
2. **ابنِ مجموعة gold مُقسّمة طبقياً** من صفحات MoJ متدهورة حقيقية (مطبوع / خط يد / مختلط × مستويات ضوضاء)، معنونة عبر خطي Claude + Gemini مع فحص تعارض.
3. **شغّل benchmark مُتحكَّم** على gold هذه: `tuned-3B` مقابل `Qwen2.5-VL-7B` و `Qwen3-VL-8B` و `AIN-7B`، عند 2–3 ميزانيات بكسل. أبلِغ **CER/WER + دقة الحقول + معدل الهلوسة** — لا أرقام تسويقية.
4. **نفّذ رافعة الدقة الحقيقية بالتوازي:** خط أنابيب detect → deskew → قصّ أسطر بالدقة الأصلية → VLM مُدرَّب، + augmentation بالتدهور (Augraphy) + قوالب قانونية مصرية مُرندَرة. هذا ما يغلق فجوة الخط اليدوي، لا تبديل النموذج.
5. **رحّل القاعدة فقط** إذا تفوّق متحدٍّ بهامش يبرّر تكلفته على 10^8–10^9.

**الخلاصة الحاسمة:** اختيارك الحالي صحيح. القاعدة مسألة من الدرجة الثانية؛ الرافعة من الدرجة الأولى هي **قصّ الأسطر بالدقة الأصلية + حجم بيانات خط اليد الحقيقي المُعنوَن**. لا تطارد نموذجاً ألمع — ابنِ دولاب البيانات.

---

# مراجعة نقدية (skeptic)

Critique — skeptical read. Most severe first.

## Licensing landmines (the doc gets the #1 pick wrong)
- **Qwen2.5-VL-3B-Instruct is NOT Apache-family.** It ships under the **Qwen Research License (non-commercial)**. The doc's "Apache-2.0-family · حكومي/تجاري OK" is false for the exact model you're recommending as the production default. The **7B and 72B are Apache-2.0; the 3B is the restricted one.** This inverts the verdict: if sherif1313 v3 is really fine-tuned on 3B, your entire MoJ deployment at 10^8–10^9 scale is on a non-commercial base. Verify each model card individually — "Apache-2.0-family" is not a thing; Qwen licenses vary per size. This alone argues for **moving to 7B (genuinely Apache-2.0)**, contradicting "stay on 3B."
- AIN-7B research-license caution is correct — but same rigor wasn't applied to the base you actually ship. Inconsistent.

## Hype still taken at face value
- **"native dynamic resolution هو العامل الحاسم الوحيد"** — overstated and self-contradicted 3 sections later ("data > base"). Every current VLM (Qwen3-VL, InternVL3, dots, Gemini) has dynamic/tiled resolution now. It is table-stakes, not a moat.
- **The real mechanism is omitted:** Qwen2.5-VL downsamples because of the **`max_pixels` visual-token cap**, not because it lacks dynamic res. At 200dpi full pages you blow the token budget and it shrinks the image. The cheap first fix is **raising min/max_pixels + tiling**, tested *before* any detect→crop pipeline. The doc jumps to a heavy pipeline and skips the config knob that may recover most of the loss for free.
- **Commercial CER ranking mixes datasets.** Gemini 0.19 (clean) vs Azure 0.83 vs Muharaf-best 0.61 are different benchmarks — not a rank order. Presenting "Gemini > GPT-4o > Azure" as a handwriting ranking repeats the same sin the doc opens by condemning. Every CER needs its dataset attached or it's marketing too.
- Muharaf CER ~0.61 for AIN-7B and the KHATT numbers are stated without citation. If you're demanding gold-measured numbers from others, cite these or mark them unverified.

## Handwriting-weakness reality the ranking soft-pedals
- **QARI/QARI-family, sherif v3/v4, SynthDocs, SARD** — all printed/synthetic. Correctly flagged, but then #5 QARI is still kept "for printed path." Fine, but note QARI-OCR's own license/base (Qwen2-VL-2B, and Qwen2-VL-2B is Apache-2.0 — actually cleaner than your 3B). Recheck.
- **Muharaf is historical manuscript correspondence, not modern MoJ forms.** Script register, ink, and layout differ from degraded 200dpi government scans with stamps/bleed-through. It's the *closest public* proxy, not a matched one — don't let "أقرب تطابق" drift into "representative." Your gold set is the only real signal; the doc says this but keeps leaning on Muharaf as an anchor.

## Omissions (better/necessary options not on the board)
- **No line/layout detector is named** despite "detect→crop" being called *the* lever. Concretely evaluate **Surya** (multilingual incl. Arabic line+layout detection), **Kraken/eScriptorium** (built for Arabic manuscript segmentation, permissive), or a YOLO/DBNet text detector. "detect→crop→VLM" is hand-waving without the detector, and detector quality on stamps/bleed-through is itself a risk.
- **Detect→crop directly contradicts your scale economics.** N line-crops/page = N VLM calls. At 10^9 pages that's a 10–40× inference-cost multiplier — the same math the doc uses to kill API serving. You can't call line-crop "the cheap real lever" and ignore that it explodes self-hosted GPU cost. Reconcile: crop only the handwriting regions, or crop→lightweight HTR, not full-page-VLM-per-line.
- **Serving cost is unaddressed for self-hosting.** The doc only computes API $/page. At 10^8–10^9, throughput engineering (vLLM/SGLang, FP8/AWQ, continuous batching, page-token budget) dominates base choice and is where 3B-vs-7B actually pays or costs. That's the first-order lever the doc misses, not line-cropping.
- **Task coupling not questioned.** One VLM doing full_text + 20-class document_type + fields jointly. At this scale, decoupling classification/field-extraction from OCR (cheap classifier on the OCR output) is usually cheaper and more accurate than a single autoregressive pass. Not discussed.

## Internal contradictions to fix
- "Stay on 3B" vs "base only sets the ceiling, 7B is the honest test" vs (now) "3B is non-commercial." The honest conclusion is: **retrain/eval on 7B-Apache as the default**, keep 3B only if license is re-verified as commercial-OK.
- "العامل الحاسم الوحيد = dynamic resolution" vs "البيانات تفوق القاعدة." Pick one thesis.

## What's right (keep)
- Refusing headline CERs until measured on stratified MoJ gold — correct and the core discipline.
- APIs as teacher/QC not serving; Chandra OpenRAIL exclusion; Textract no-Arabic; data-flywheel-over-model-chasing — all sound.
- Two-tier router (printed fast-path vs handwriting) is the right shape.

**Bottom line:** the analysis is directionally strong but has one disqualifying factual error (3B license) that flips its own verdict, dresses a token-budget config issue up as an architecture moat, and calls line-cropping "cheap" while it violates the doc's own scale economics. Fix the license check, test the `max_pixels` knob before building a crop pipeline, name a detector, and cost the crop-multiplier at 10^9 before committing.
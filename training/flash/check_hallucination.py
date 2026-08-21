# -*- coding: utf-8 -*-
"""
Measure whether the fine-tuned model INVENTS text, rather than admitting it cannot read.

text_sim does not answer this. A model can score respectably while quietly filling in
unreadable handwriting with plausible Arabic, and that failure is far worse for a legal
archive than a low score would be: a wrong case number or national ID looks authoritative.

The training labels carry an explicit rule - "ممنوع التخمين نهائياً", write [غير واضح]
instead - so an honest model should reproduce roughly the gold's marker pattern. On the
53-page human test the reviewer confirmed 136 unreadable spots across 31 of 53 pages.

Signals reported:
  * marker recall  - of the gold's [غير واضح] spots, how many does the model also flag
  * over-confidence - pages where gold flags >=3 spots and the model flags none
  * length inflation - model text much longer than gold on those same pages
  * silence         - the opposite failure: flagging far MORE than the gold

Needs a report produced by the current evaluate.py (it stores pred_text; reports made
before that change only hold scores).

  python check_hallucination.py report_v4_human.json
"""
import argparse, json, os, statistics, sys

BASE = os.path.dirname(os.path.abspath(__file__))
UNCLEAR = "[غير واضح]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--test", default="dataset/labels/v1_test_gold_human.jsonl")
    args = ap.parse_args()

    rep = json.load(open(os.path.join(BASE, args.report), encoding="utf-8"))
    rows = rep["rows"]
    if "pred_text" not in rows[0]:
        sys.exit("this report has no pred_text - it predates the evaluate.py change.\n"
                 "Re-run: python evaluate.py --adapter <dir> --out <report>")

    gold = {}
    for l in open(os.path.join(BASE, args.test), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            gold[r["image"].replace("\\", "/")] = r

    tot_g = tot_p = 0
    blind, chatty, ok = [], [], 0
    ratios = []
    for r in rows:
        g = gold.get(r["image"].replace("\\", "/"))
        if not g:
            continue
        gt, pt = g.get("full_text") or "", r.get("pred_text") or ""
        gm, pm = gt.count(UNCLEAR), pt.count(UNCLEAR)
        tot_g += gm
        tot_p += pm
        if len(gt):
            ratios.append(len(pt) / len(gt))
        if gm >= 3 and pm == 0:
            blind.append((r["image"], gm, pm, len(gt), len(pt), r["text_sim"]))
        elif gm == 0 and pm >= 5:
            chatty.append((r["image"], gm, pm, r["text_sim"]))
        else:
            ok += 1

    print("=" * 70)
    print("HONESTY CHECK  (%s)" % os.path.basename(args.report))
    print("=" * 70)
    print("unreadable spots the human confirmed : %d" % tot_g)
    print("unreadable spots the model flagged   : %d" % tot_p)
    print("marker ratio model/gold              : %.2f  %s" % (
        tot_p / max(tot_g, 1),
        "(<<1 = the model is filling in what it cannot read)" if tot_p < tot_g * 0.5 else ""))
    if ratios:
        print("pred/gold length ratio               : median %.2f  mean %.2f"
              % (statistics.median(ratios), statistics.mean(ratios)))
    print()
    print("OVER-CONFIDENT pages (gold flags >=3 unreadable spots, model flags none): %d" % len(blind))
    for img, gm, pm, gl, pl, sim in sorted(blind, key=lambda x: -x[1])[:12]:
        print("   gold %2d markers -> model %d | chars %4d -> %4d | sim %.2f | %s"
              % (gm, pm, gl, pl, sim, img[-38:]))
    print()
    print("OVER-CAUTIOUS pages (gold clean, model flags >=5): %d" % len(chatty))
    for img, gm, pm, sim in chatty[:6]:
        print("   gold %d -> model %2d | sim %.2f | %s" % (gm, pm, sim, img[-38:]))
    print()
    print("VERDICT")
    if tot_p >= tot_g * 0.7 and len(blind) <= 2:
        print("  Honest. The model reproduces the gold's admissions of illegibility.")
    elif tot_p < tot_g * 0.4 or len(blind) >= 5:
        print("  ** INVENTING. ** It writes confident text where the human could not read.")
        print("  For a legal archive this outweighs a good text_sim - a fabricated case")
        print("  number or national ID reads as authoritative. Do not ship on this alone.")
    else:
        print("  Mixed. Inspect the over-confident pages above before trusting it.")


if __name__ == "__main__":
    main()

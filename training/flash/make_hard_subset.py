# -*- coding: utf-8 -*-
"""
Carve a small HARD subset out of the 53-page human test, for fast iteration.

A full eval is ~20-35 min, which is slow when you are only trying to see whether a change
helped the pages that are actually broken. This picks the pages worth watching:
  * every page where v4 scores below --thr (default 0.6)
  * every page where a comparison run collapsed (text_sim < 0.1) even though v4 was fine
  * every خط يد page, since that is the weakest category and only 4 pages carry it

!! The score on this subset is NOT comparable to the headline number. It is deliberately
!! biased toward failures, so it reads far lower than the full-set score. Use it to see
!! MOVEMENT between runs; quote only the full 53-page number as the model's accuracy.

  python make_hard_subset.py                       # from report_v4_matched.json
  python make_hard_subset.py --also report_qwen3vl_human.json
"""
import argparse, json, os

BASE = os.path.dirname(os.path.abspath(__file__))


def rows(path):
    return {r["image"]: r for r in json.load(open(path, encoding="utf-8"))["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-report", default="report_v4_matched.json")
    ap.add_argument("--also", nargs="*", default=[], help="other reports; their collapses are added")
    ap.add_argument("--test", default="dataset/labels/v1_test_gold_human.jsonl")
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--out", default="dataset/labels/test_hard.jsonl")
    args = ap.parse_args()

    gold = {}
    for l in open(os.path.join(BASE, args.test), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            gold[r["image"].replace("\\", "/")] = r

    base = rows(os.path.join(BASE, args.base_report))
    picked, why = {}, {}

    for k, r in base.items():
        if r["text_sim"] < args.thr:
            picked[k] = True
            why[k] = "v4 weak (%.3f)" % r["text_sim"]

    for other in args.also:
        for k, r in rows(os.path.join(BASE, other)).items():
            if r["text_sim"] < 0.1 and base.get(k, {}).get("text_sim", 0) >= 0.1:
                picked[k] = True
                why.setdefault(k, "collapsed in %s (%.3f)" % (os.path.basename(other), r["text_sim"]))

    for k, g in gold.items():
        if g.get("content_kind") == "خط يد":
            picked[k] = True
            why.setdefault(k, "handwriting (weakest category)")

    keep = [gold[k] for k in gold if k in picked]
    out = os.path.join(BASE, args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("hard subset: %d of %d pages -> %s\n" % (len(keep), len(gold), args.out))
    for k in sorted(picked, key=lambda x: base.get(x, {}).get("text_sim", 0)):
        if k in gold:
            print("  v4 %.3f  %-8s  %-34s  %s"
                  % (base.get(k, {}).get("text_sim", 0), gold[k].get("content_kind", "?"),
                     k[-34:], why[k]))
    print("\nrun it with:  python evaluate.py --adapter <dir> --test %s --out <report>" % args.out)
    print("REMINDER: subset scores are not comparable to the 53-page number.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Build an IN-DOMAIN test set from the validation split.

Why: the 53-page human test is drawn entirely from OCR1/Alex2017 (81%) and OCR2/Giza,
and ZERO training pages come from those branches - they are the deliberately held-out
"unseen governorate/year" split. So `text_sim 0.710` is a GENERALISATION score, not an
accuracy score. We have never measured how the model does on documents that look like
what it trained on.

flash_val.json answers that: 150 pages from the training distribution that the model was
evaluated on but never trained on. Labels come from labels_resolved.jsonl (Gemini), so the
number is student-vs-teacher rather than student-vs-human - read it alongside the fact that
Gemini itself scores 0.929 against the human gold.

  python make_indomain_test.py            # -> dataset/labels/test_indomain.jsonl
"""
import argparse, json, os

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/flash_val.json")
    ap.add_argument("--gold", default="dataset/labels/labels_resolved.jsonl")
    ap.add_argument("--out", default="dataset/labels/test_indomain.jsonl")
    ap.add_argument("--limit", type=int, default=60, help="keep it comparable in size to the human test")
    args = ap.parse_args()

    gold = {}
    for l in open(os.path.join(BASE, args.gold), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            gold[r["image"].replace("\\", "/")] = r

    val = json.load(open(os.path.join(BASE, args.val), encoding="utf-8"))
    rels = [x["images"][0].replace("dataset/images/", "") for x in val]

    keep, missing = [], 0
    for rel in rels:
        if rel in gold:
            keep.append(gold[rel])
        else:
            missing += 1
    keep = keep[: args.limit]

    out = os.path.join(BASE, args.out)
    with open(out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import collections
    print("val pages: %d | labelled: %d | missing: %d" % (len(rels), len(rels) - missing, missing))
    print("wrote %d pages -> %s" % (len(keep), args.out))
    br = collections.Counter("/".join(r["image"].replace("\\", "/").split("/")[:2]) for r in keep)
    kd = collections.Counter(r.get("content_kind") for r in keep)
    print("branches: %s" % dict(br))
    print("kinds   : %s" % dict(kd))
    print("\nrun:  python evaluate.py --adapter <dir> --test %s --out <report>" % args.out)
    print("This measures student-vs-TEACHER in-domain. The 53-page human test stays the")
    print("headline number; this one answers a different question.")


if __name__ == "__main__":
    main()

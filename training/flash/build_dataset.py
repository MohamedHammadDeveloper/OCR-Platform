"""
Convert Flash gold JSONL -> LLaMA-Factory multimodal SFT format.

Reads v1_gold.jsonl (299 train), splits into train/val, and writes:
  - flash_train.json  (list of {messages, images})
  - flash_val.json
  - dataset_info.json  (registration LLaMA-Factory reads)

Image paths in the gold are relative to the images root; we prepend --images-root
so LLaMA-Factory gets absolute paths that exist on the training box.

SPLIT HYGIENE: OCR1/Alex2017 + OCR2/Giza are the held-out TEST branches (unseen
governorate/year). v1 trained on 0 pages from them; v2 batch-1 sampling accidentally
pulled in 15 (8 share a PDF with an actual test page). Training on those inflates the
test metrics and makes v1-vs-v2 unreadable, so they are dropped from train/val by
DEFAULT. The labels stay on disk — pass --include-heldout for the FINAL production
model, once you are no longer comparing runs.

Usage (on the training box):
  python build_dataset.py \
      --gold /data/flash/labels/v1_gold.jsonl \
      --images-root /data/flash/images \
      --out-dir ./data --val 30
"""
import argparse, hashlib, json, os, random

from prompt import build_target, user_content_llamafactory


def dedup_by_bytes(recs, images_root):
    """Drop byte-identical duplicate scans (same image under OCR1/alex2020 AND
    OCR2/Alex/2020 trees, etc.). Keeps ONE record per md5 group — the one whose
    image path sorts first — so the choice is deterministic and reproducible on
    any box. This matters because the manifest lists the duplicated scans as
    separate rows, so the sampler labeled some pages twice (occasionally with
    conflicting types); training on both wastes capacity and injects noise."""
    groups = {}
    missing = 0
    for r in recs:
        p = images_root.rstrip("/\\") + "/" + r["image"].replace("\\", "/")
        try:
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except OSError:
            missing += 1
            h = "MISSING:" + r["image"]  # never merge un-hashable rows together
        groups.setdefault(h, []).append(r)
    keep, dropped = [], []
    for h, rs in groups.items():
        rs_sorted = sorted(rs, key=lambda r: r["image"].replace("\\", "/"))
        keep.append(rs_sorted[0])
        dropped.extend(rs_sorted[1:])
    if missing:
        print(f"dedup: WARNING {missing} rows could not be hashed (image not found) — kept as-is")
    if dropped:
        print(f"dedup: dropped {len(dropped)} byte-identical duplicate scans "
              f"({len(keep)} unique kept)")
    return keep, dropped


def sample(record, images_root):
    # forward slashes only — training runs on Linux (vast.ai)
    img = images_root.rstrip("/\\") + "/" + record["image"].replace("\\", "/")
    # LLaMA-Factory CANONICAL sharegpt: conversations + from/value + human/gpt.
    # (The role/content/user/assistant variant left labels unmasked -> grad_norm 0.)
    return {
        "conversations": [
            {"from": "human", "value": user_content_llamafactory()},
            {"from": "gpt", "value": build_target(record)},
        ],
        "images": [img],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="dataset/labels/v1_gold.jsonl")
    ap.add_argument("--images-root", default="dataset/images")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--val", type=int, default=30, help="val examples held out from train gold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heldout-branches", default="OCR1/Alex2017,OCR2/Giza",
                    help="comma-separated path prefixes that make up the held-out TEST split; "
                         "rows under them are dropped from train/val. Pass '' to disable.")
    ap.add_argument("--include-heldout", action="store_true",
                    help="TRAIN on the held-out branches too. Only for the final production "
                         "model — it makes test metrics incomparable with earlier runs.")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip byte-identical duplicate-scan removal (dedup is ON by default)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recs = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    # drop fully-blank transcriptions from training (low value)
    recs = [r for r in recs if r.get("full_text", "").strip()
            and r["full_text"].strip() != "[صفحة فارغة - لا يوجد نص مقروء]"]

    # drop byte-identical duplicate scans (same page under OCR1/alex2020 & OCR2/Alex trees)
    if not args.no_dedup:
        recs, dup_dropped = dedup_by_bytes(recs, args.images_root)
        if dup_dropped:
            path = os.path.join(args.out_dir, "excluded_dups.json")
            json.dump([r["image"] for r in dup_dropped], open(path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"  dup list -> {path} (gold on disk is untouched)")

    prefixes = tuple(p.strip().rstrip("/") + "/"
                     for p in args.heldout_branches.split(",") if p.strip())
    if args.include_heldout:
        print(f"!! --include-heldout: training on {list(prefixes)} as well — "
              f"test metrics will NOT be comparable to v1/v2 runs")
    elif prefixes:
        keep, dropped = [], []
        for r in recs:
            img = r["image"].replace("\\", "/")
            (dropped if img.startswith(prefixes) else keep).append(r)
        recs = keep
        by_branch = {}
        for r in dropped:
            b = "/".join(r["image"].replace("\\", "/").split("/")[:2])
            by_branch[b] = by_branch.get(b, 0) + 1
        print(f"held-out excluded from train/val: {len(dropped)} rows {by_branch}")
        if dropped:
            path = os.path.join(args.out_dir, "excluded_heldout.json")
            json.dump([r["image"] for r in dropped], open(path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"  list -> {path} (gold on disk is untouched)")

    rng = random.Random(args.seed)
    rng.shuffle(recs)

    val = recs[: args.val]
    train = recs[args.val:]
    for name, part in [("flash_train", train), ("flash_val", val)]:
        data = [sample(r, args.images_root) for r in part]
        path = os.path.join(args.out_dir, name + ".json")
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {path}: {len(data)} samples")

    def entry(fn):
        return {
            "file_name": fn, "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt"},
        }
    info = {"flash_train": entry("flash_train.json"),
            "flash_val": entry("flash_val.json")}
    json.dump(info, open(os.path.join(args.out_dir, "dataset_info.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", os.path.join(args.out_dir, "dataset_info.json"))


if __name__ == "__main__":
    main()

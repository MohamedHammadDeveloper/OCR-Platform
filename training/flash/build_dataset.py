"""
Convert Flash gold JSONL -> LLaMA-Factory multimodal SFT format.

Reads v1_gold.jsonl (299 train), splits into train/val, and writes:
  - flash_train.json  (list of {messages, images})
  - flash_val.json
  - dataset_info.json  (registration LLaMA-Factory reads)

Image paths in the gold are relative to the images root; we prepend --images-root
so LLaMA-Factory gets absolute paths that exist on the training box.

Usage (on the training box):
  python build_dataset.py \
      --gold /data/flash/labels/v1_gold.jsonl \
      --images-root /data/flash/images \
      --out-dir ./data --val 30
"""
import argparse, json, os, random

from prompt import build_target, user_content_llamafactory


def sample(record, images_root):
    # forward slashes only — training runs on Linux (vast.ai)
    img = images_root.rstrip("/\\") + "/" + record["image"].replace("\\", "/")
    return {
        "messages": [
            {"role": "user", "content": user_content_llamafactory()},
            {"role": "assistant", "content": build_target(record)},
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recs = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    # drop fully-blank transcriptions from training (low value)
    recs = [r for r in recs if r.get("full_text", "").strip()
            and r["full_text"].strip() != "[صفحة فارغة - لا يوجد نص مقروء]"]
    rng = random.Random(args.seed)
    rng.shuffle(recs)

    val = recs[: args.val]
    train = recs[args.val:]
    for name, part in [("flash_train", train), ("flash_val", val)]:
        data = [sample(r, args.images_root) for r in part]
        path = os.path.join(args.out_dir, name + ".json")
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {path}: {len(data)} samples")

    info = {
        "flash_train": {
            "file_name": "flash_train.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        },
        "flash_val": {
            "file_name": "flash_val.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        },
    }
    json.dump(info, open(os.path.join(args.out_dir, "dataset_info.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", os.path.join(args.out_dir, "dataset_info.json"))


if __name__ == "__main__":
    main()

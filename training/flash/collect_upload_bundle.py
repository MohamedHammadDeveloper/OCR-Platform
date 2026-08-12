"""
Collect exactly what needs to be uploaded to the GPU box for v1 training:
only the images referenced by the train + test gold (not all 7,852), plus the
two gold JSONL files, preserving the relative folder structure.

Usage:
  python collect_upload_bundle.py \
      --labels-dir E:/Work/Namaa/Flash/labels \
      --images-root E:/Work/Namaa/Flash/images \
      --out E:/Work/Namaa/Flash/upload_bundle

Then zip `out/` and upload; on the box use --images-root <bundle>/images.
"""
import argparse, json, os, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    golds = ["v1_gold.jsonl", "v1_test_gold.jsonl"]
    imgs_out = os.path.join(args.out, "images")
    lbl_out = os.path.join(args.out, "labels")
    os.makedirs(imgs_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    copied, missing, total_bytes = 0, [], 0
    for g in golds:
        gp = os.path.join(args.labels_dir, g)
        shutil.copy2(gp, os.path.join(lbl_out, g))
        for line in open(gp, encoding="utf-8"):
            if not line.strip():
                continue
            rel = json.loads(line)["image"].replace("\\", "/")
            src = os.path.join(args.images_root, rel)
            dst = os.path.join(imgs_out, rel)
            if not os.path.exists(src):
                missing.append(rel); continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            total_bytes += os.path.getsize(src)
            copied += 1

    print(f"copied {copied} images | {total_bytes/1e6:.1f} MB")
    print(f"labels -> {lbl_out}")
    if missing:
        print(f"WARNING: {len(missing)} images missing, e.g. {missing[:3]}")
    print(f"bundle ready at: {args.out}  (zip and upload)")


if __name__ == "__main__":
    main()

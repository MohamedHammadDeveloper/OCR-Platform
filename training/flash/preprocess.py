"""
Gentle image enhancement for faded / low-contrast legal scans.

Design goal: make faded print + ink readable WITHOUT destroying thin handwriting
strokes. So: local contrast (CLAHE) + light denoise + upscale-if-small. NO hard
binarization by default (that kills faint handwriting) — it's opt-in for
printed-only pages via --binarize.

The SAME enhancement must be applied to training images and at inference (consistency).

As a module:
    from preprocess import enhance_image
    pil = enhance_image("page.png")            # -> PIL.Image (RGB)

As a CLI (single or whole tree, mirroring structure):
    python preprocess.py --in page.png --out page_enh.png
    python preprocess.py --dir  E:/.../images  --out-dir E:/.../images_enh
"""
import argparse, os, glob
import cv2
import numpy as np
from PIL import Image


def enhance_image(path_or_pil, min_long_side=2200, clip=2.0, denoise_h=6, binarize=False):
    if isinstance(path_or_pil, str):
        img = cv2.imread(path_or_pil, cv2.IMREAD_GRAYSCALE)
    else:
        img = np.array(path_or_pil.convert("L"))
    if img is None:
        raise ValueError(f"could not read image: {path_or_pil}")

    # 1) upscale small scans (thin faded text needs pixels) — never downscale here
    h, w = img.shape
    long_side = max(h, w)
    if long_side < min_long_side:
        s = min_long_side / long_side
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)

    # 2) light denoise (removes scan speckle; small h keeps strokes)
    if denoise_h:
        img = cv2.fastNlMeansDenoising(img, None, h=denoise_h, templateWindowSize=7,
                                       searchWindowSize=21)

    # 3) CLAHE — local contrast; rescues unevenly faded regions, gentle on strokes
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 4) optional adaptive binarization — PRINTED pages only (kills faint handwriting)
    if binarize:
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)

    return Image.fromarray(img).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--min-long-side", type=int, default=2200)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--denoise-h", type=int, default=6)
    ap.add_argument("--binarize", action="store_true")
    args = ap.parse_args()

    kw = dict(min_long_side=args.min_long_side, clip=args.clip,
              denoise_h=args.denoise_h, binarize=args.binarize)

    if args.inp:
        enhance_image(args.inp, **kw).save(args.out)
        print("wrote", args.out)
    elif args.dir:
        files = glob.glob(os.path.join(args.dir, "**", "*.png"), recursive=True)
        for i, f in enumerate(files, 1):
            rel = os.path.relpath(f, args.dir)
            dst = os.path.join(args.out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            enhance_image(f, **kw).save(dst)
            if i % 100 == 0:
                print(f"  ...{i}/{len(files)}")
        print(f"done: {len(files)} images -> {args.out_dir}")
    else:
        ap.error("give --in/--out or --dir/--out-dir")


if __name__ == "__main__":
    main()

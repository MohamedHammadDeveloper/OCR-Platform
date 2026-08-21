# -*- coding: utf-8 -*-
"""
Mirror training checkpoints to the Hub as they appear, without restarting the run.

LLaMA-Factory can do this itself (`push_to_hub` + `hub_strategy: checkpoint`), but only if
it was set BEFORE training started - turning it on mid-run means relaunching and losing the
progress so far. This watches the output dir instead, so it can be started at any point.

It also covers a second hazard: `save_total_limit: 3` deletes older checkpoints, so a box
that dies can leave nothing useful behind even though checkpoints were "being saved".

Run it in a SECOND terminal (needs `hf auth login` first):
  python watch_and_push.py --out-dir saves/qwen25vl-7b-lora-vision \
      --repo m-hammad/flash-vision-ckpt

Note: with freeze_vision_tower=false the checkpoints carry vision-tower weights too, so
they are GBs, not the ~100MB a pure-LoRA checkpoint would be. --keep-latest-only uploads
each new checkpoint to the same folder to save bandwidth.
"""
import argparse, glob, os, re, sys, time


def step_of(path):
    m = re.search(r"checkpoint-(\d+)$", path.replace("\\", "/"))
    return int(m.group(1)) if m else -1


def size_gb(path):
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", required=True, help="e.g. m-hammad/flash-vision-ckpt")
    ap.add_argument("--every", type=int, default=60, help="seconds between scans")
    ap.add_argument("--keep-latest-only", action="store_true",
                    help="always upload into 'latest/' instead of one folder per step")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    try:
        create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
        print("repo ready: %s (private)" % args.repo)
    except Exception as e:
        sys.exit("could not create/access %s: %s\nDid you run `hf auth login`?" % (args.repo, e))

    done = set()
    print("watching %s every %ds - Ctrl+C to stop\n" % (args.out_dir, args.every))
    while True:
        cks = sorted(glob.glob(os.path.join(args.out_dir, "checkpoint-*")), key=step_of)
        for ck in cks:
            s = step_of(ck)
            if s in done or not os.path.exists(os.path.join(ck, "trainer_state.json")):
                continue                       # skip half-written checkpoints
            target = "latest" if args.keep_latest_only else os.path.basename(ck)
            gb = size_gb(ck)
            print("uploading %s (%.2f GB) -> %s/%s ..." % (os.path.basename(ck), gb, args.repo, target))
            try:
                api.upload_folder(folder_path=ck, repo_id=args.repo, path_in_repo=target,
                                  commit_message="checkpoint-%d" % s)
                done.add(s)
                print("  done (steps mirrored: %s)" % sorted(done))
            except Exception as e:
                print("  FAILED: %s - will retry next scan" % e)
        time.sleep(args.every)


if __name__ == "__main__":
    main()

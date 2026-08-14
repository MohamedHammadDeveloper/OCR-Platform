"""
Print a side-by-side table of one or more evaluate.py report JSONs.
Use it to read a resolution sweep (or any A/B of adapters) at a glance.

  python compare_reports.py report_v3.json report_v3_res1M.json report_v3_res2M.json ...
"""
import json, sys

COLS = [("json_parse_rate", "json"), ("type_accuracy", "type"),
        ("text_similarity_mean", "text_sim"), ("keyword_recall_mean", "kw_rec")]


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python compare_reports.py <report1.json> [report2.json ...]")
        return
    reps = [(p, json.load(open(p, encoding="utf-8"))["summary"]) for p in paths]
    name_w = max(len(p) for p in paths) + 2
    print(f"{'report':<{name_w}}{'n':>5}" + "".join(f"{lbl:>11}" for _, lbl in COLS))
    for p, s in reps:
        row = f"{p:<{name_w}}{s.get('n',''):>5}"
        for key, _ in COLS:
            v = s.get(key)
            row += f"{(f'{v:.3f}' if isinstance(v,(int,float)) else '-'):>11}"
        print(row)
    # deltas vs the first report (the baseline)
    if len(reps) > 1:
        base = reps[0][1]
        print(f"\ndeltas vs {paths[0]}:")
        for p, s in reps[1:]:
            d = []
            for key, lbl in COLS:
                if isinstance(s.get(key), (int, float)) and isinstance(base.get(key), (int, float)):
                    d.append(f"{lbl} {s[key]-base[key]:+.3f}")
            print(f"  {p}: " + "  ".join(d))


if __name__ == "__main__":
    main()

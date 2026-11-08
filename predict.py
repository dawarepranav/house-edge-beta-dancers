"""Recreate the House Edge forecast CSV: python predict.py --data /data --out /output/submission.csv"""
import argparse
import csv
from pathlib import Path

import he_model as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    P = H.load_params(Path(__file__).resolve().parent / "params.json")
    queries, history, episodes = H.load_split(Path(a.data) / a.split)
    preds, _ = H.predict_all(episodes, history, P)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for q in queries:
            w.writerow([q["query_id"]] + [repr(v) for v in preds[q["query_id"]]])


if __name__ == "__main__":
    main()

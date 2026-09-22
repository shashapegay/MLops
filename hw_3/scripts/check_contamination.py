#!/usr/bin/env python3
"""Separate train/test contamination gate."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_params
from src.contamination import is_clean, report
from src.schema import iter_examples

def main() -> int:
    params = load_params()
    paths = params["paths"]
    nd = params["clean"]["near_dup"]
    train = list(iter_examples(paths["train"]))
    test = list(iter_examples(paths["test"]))
    rep = report(train, test, nd["shingle_words"], nd["num_perm"], params["contamination"]["threshold"])
    payload = {
        "version": params["collect"]["version"],
        "train_rows": len(train),
        "test_rows": len(test),
        **rep,
        "passed": is_clean(rep),
    }
    Path(paths["metrics_contamination"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["metrics_contamination"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train: {len(train)} строк, test: {len(test)} строк")
    print(f"  пересечение по id:        {rep['id_overlap']}")
    print(f"  пересечение по тексту:    {rep['text_overlap']}")
    print(f"  пересечение по группам:   {rep['group_overlap']}")
    print(f"  near-dup train-test pairs: {rep['near_dup_pairs']}")
    if is_clean(rep):
        print("no contamination")
        return 0
    print("CONTAMINATION: train and test overlap")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())

"""Collect a reproducible slice of Open-Orca from the Hugging Face Dataset Viewer API.

The source is intentionally treated as raw material.  We:
* select a deterministic, configurable slice;
* convert Open-Orca's system_prompt/question/response schema to the course chat schema;
* diversify the system instruction;
* derive a stable group key used only for honest splitting.

No generated copy of Open-Orca is committed to git; the raw JSONL is a DVC output.
"""
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.config import load_params

API_BASE = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "as", "by", "with", "from", "that", "this", "it", "what",
    "which", "who", "how", "why", "when", "where", "do", "does", "did", "can",
    "could", "would", "should", "please", "following", "given", "answer", "question",
    "following", "select", "choose", "one",
}

def pick_prompt(example_id: str, variants: list[str]) -> str:
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]

def normalize_for_group(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def derive_topic(example_id: str, question: str, bucket_count: int) -> str:
    """Stable, transparent grouping key, not a semantic ground-truth label.

    Open-Orca does not provide a topic column.  We therefore form reproducible
    lexical buckets from the question.  The source family plus a 128-way bucket
    keeps groups sufficiently fine-grained for a group-wise split while avoiding
    inventing semantic labels.
    """
    family = example_id.split(".", 1)[0].lower() or "unknown"
    words = [w for w in normalize_for_group(question).split() if w not in STOPWORDS]
    signature = " ".join(words[:8])
    digest = hashlib.sha1(f"{family}|{signature}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % bucket_count
    return f"{family}:bucket-{bucket:03d}"

def fetch_rows(offset: int, length: int, source: dict) -> list[dict]:
    params = {
        "dataset": source["dataset"],
        "config": source["config"],
        "split": source["split"],
        "offset": offset,
        "length": length,
    }
    req = Request(f"{API_BASE}?{urlencode(params)}", headers={"User-Agent": "mlops26-hw3/1.0"})
    with urlopen(req, timeout=60) as response:
        payload = json.load(response)
    return [item["row"] for item in payload.get("rows", [])]

def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    variants = cfg["system_prompts"]
    if not variants:
        raise SystemExit("collect.system_prompts пуст")

    target = int(cfg["n_rows"])
    source = cfg["source"]
    offset = int(cfg.get("offsets", {}).get(cfg["version"], 0))
    bucket_count = int(cfg.get("topic_buckets", 128))
    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    scanned = written = dropped_short = 0
    prompts_used: set[str] = set()
    seen_ids: set[str] = set()

    with out.open("w", encoding="utf-8") as fh:
        cursor = offset
        while written < target:
            rows = fetch_rows(cursor, PAGE_SIZE, source)
            if not rows:
                break
            cursor += len(rows)
            for row in rows:
                scanned += 1
                ex_id = str(row.get("id", "")).strip()
                question = str(row.get("question", "")).strip()
                response = str(row.get("response", "")).strip()
                if not ex_id or ex_id in seen_ids or len(question) < cfg["min_source_question_chars"] or len(response) < cfg["min_source_response_chars"]:
                    dropped_short += 1
                    continue
                seen_ids.add(ex_id)
                prompt = pick_prompt(ex_id, variants)
                prompts_used.add(prompt)
                record = {
                    "id": ex_id,
                    "topic": derive_topic(ex_id, question, bucket_count),
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= target:
                    break

    if written < target:
        raise SystemExit(f"Open-Orca: удалось собрать только {written} из {target} строк")

    metrics = {
        "version": cfg["version"],
        "source": source,
        "offset": offset,
        "rows_scanned": scanned,
        "rows_written": written,
        "dropped_short_or_duplicate_id": dropped_short,
        "system_prompt_variants": len(prompts_used),
        "topic_buckets": bucket_count,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"collect: {cfg['version']}, scanned={scanned}, written={written}, prompts={len(prompts_used)}, {metrics['seconds']} s")

if __name__ == "__main__":
    main()

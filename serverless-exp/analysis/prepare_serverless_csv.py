#!/usr/bin/env python3
import json
import csv
from pathlib import Path

# ----------------------------
# USER SETTINGS (EDIT HERE)
# ----------------------------

# Option A: Process a single file (recommended for your case)
INPUT_FILES = [
    "histgbdtenable_cpuintensive1.json"
]

# Option B: Process multiple files (uncomment and list them)
# INPUT_FILES = [
#     "file1.json",
#     "file2.json",
#     "file3.json"
# ]

# Option C: Process ALL json/jsonl files in a folder (uncomment)
# INPUT_FOLDER = "raw_logs"
# INPUT_FILES = None

OUTPUT_CSV = "prepared.csv"

# ----------------------------
# Expected keys in each record
# ----------------------------
REQUIRED_KEYS = [
    "platform", "workload", "run_id", "invocation_id",
    "mem_limit_mb", "cold_start", "concurrency", "queue_delay_ms",
    "duration_ms", "cpu_time_ms", "rss_mb", "peak_rss_mb",
    "io_read_bytes", "io_write_bytes", "energy_joules"
]


def read_json_records(file_path: Path):
    """
    Supports:
    - JSONL (one JSON object per line)
    - Single JSON object
    - JSON array
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return

    # JSON array
    if text.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec
        return

    # If it's a single JSON object in one line
    if "\n" not in text and text.startswith("{") and text.endswith("}"):
        obj = json.loads(text)
        if isinstance(obj, dict):
            yield obj
        return

    # JSONL
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1]
        yield json.loads(s)


def collect_files():
    """
    Uses INPUT_FILES if set, otherwise uses INPUT_FOLDER.
    """
    # Folder mode
    if "INPUT_FOLDER" in globals() and globals().get("INPUT_FOLDER") and globals().get("INPUT_FILES") is None:
        folder = Path(globals()["INPUT_FOLDER"])
        files = list(folder.glob("*.json")) + list(folder.glob("*.jsonl"))
        return sorted(files)

    # File list mode
    files = []
    for name in INPUT_FILES:
        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p.resolve()}")
        files.append(p)
    return files


def main():
    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    if not files:
        raise RuntimeError("No input files found. Check INPUT_FILES / INPUT_FOLDER.")

    print("Processing:")
    for f in files:
        print("  -", f)

    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=REQUIRED_KEYS)
        writer.writeheader()

        for file in files:
            for rec in read_json_records(file):
                row = {k: rec.get(k, None) for k in REQUIRED_KEYS}
                writer.writerow(row)
                rows_written += 1

    print(f"[OK] Wrote {rows_written} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
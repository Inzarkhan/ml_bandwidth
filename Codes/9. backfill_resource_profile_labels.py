#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOADS_DIR = (SCRIPT_DIR.parent / "serverless-exp" / "workloads").resolve()
if str(WORKLOADS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKLOADS_DIR))

from sebs_common import RESOURCE_PROFILE_MAP, RESOURCE_PROFILE_PREFIX, SEBS_WORKLOADS  # noqa: E402


LABEL_COLUMNS = [
    "resource_profile",
    "resource_profile_index",
    "resource_profile_label",
    "plot_workload_name",
]


def build_metadata_maps():
    by_workload = {}
    by_benchmark_partition = {}

    for workload_key, (resource_profile, resource_profile_index) in RESOURCE_PROFILE_MAP.items():
        spec = SEBS_WORKLOADS[workload_key]
        profile_prefix = RESOURCE_PROFILE_PREFIX[resource_profile]
        resource_profile_label = f"{profile_prefix}{resource_profile_index}"
        metadata = {
            "resource_profile": resource_profile,
            "resource_profile_index": resource_profile_index,
            "resource_profile_label": resource_profile_label,
            "plot_workload_name": f"{spec['benchmark_name']}_{resource_profile_label}",
        }
        by_workload[workload_key] = metadata
        by_benchmark_partition[(spec["benchmark_name"], spec["partition"])] = metadata

    return by_workload, by_benchmark_partition


BY_WORKLOAD, BY_BENCHMARK_PARTITION = build_metadata_maps()


def normalize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def lookup_metadata(record):
    workload_key = normalize_text(record.get("workload")) or normalize_text(record.get("workload_key"))
    if workload_key and workload_key in BY_WORKLOAD:
        return BY_WORKLOAD[workload_key]

    benchmark_name = normalize_text(record.get("benchmark_name"))
    partition = normalize_text(record.get("partition"))
    if benchmark_name and partition:
        return BY_BENCHMARK_PARTITION.get((benchmark_name, partition))

    return None


def backfill_record(record):
    metadata = lookup_metadata(record)
    if metadata is None:
        return record, False

    updated = dict(record)
    changed = False
    for col_name, value in metadata.items():
        if updated.get(col_name) != value:
            updated[col_name] = value
            changed = True
    return updated, changed


def write_jsonl(path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    rows_seen = 0
    rows_changed = 0
    rows_missing = 0

    with open(path, "r", encoding="utf-8") as src, open(temp_path, "w", encoding="utf-8") as dst:
        for line in src:
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            updated, changed = backfill_record(record)
            rows_seen += 1
            if changed:
                rows_changed += 1
            if lookup_metadata(updated) is None:
                rows_missing += 1
            dst.write(json.dumps(updated) + "\n")

    os.replace(temp_path, path)
    return rows_seen, rows_changed, rows_missing


def write_csv(path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    rows_seen = 0
    rows_changed = 0
    rows_missing = 0

    with open(path, "r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        for col_name in LABEL_COLUMNS:
            if col_name not in fieldnames:
                fieldnames.append(col_name)

        with open(temp_path, "w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for record in reader:
                updated, changed = backfill_record(record)
                rows_seen += 1
                if changed:
                    rows_changed += 1
                if lookup_metadata(updated) is None:
                    rows_missing += 1
                writer.writerow(updated)

    os.replace(temp_path, path)
    return rows_seen, rows_changed, rows_missing


def process_file(path_str):
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        rows_seen, rows_changed, rows_missing = write_csv(path)
    else:
        rows_seen, rows_changed, rows_missing = write_jsonl(path)

    print(
        f"[OK] Backfilled {path} | rows={rows_seen} | changed={rows_changed} | "
        f"unmapped={rows_missing}"
    )


def main():
    raw_inputs = os.environ.get("BACKFILL_INPUT_FILES", "").strip()
    if not raw_inputs:
        raise RuntimeError(
            "Set BACKFILL_INPUT_FILES to a comma-separated list of CSV/JSONL files, "
            "for example raw_known_full_plusfb_windows.jsonl,prepared_known_full_plusfb.csv."
        )

    input_files = [item.strip() for item in raw_inputs.split(",") if item.strip()]
    for path_str in input_files:
        process_file(path_str)


if __name__ == "__main__":
    main()

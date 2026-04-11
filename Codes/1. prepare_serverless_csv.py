#!/usr/bin/env python3
import json
import csv
import os
from pathlib import Path

# ----------------------------
# USER SETTINGS (EDIT HERE)
# ----------------------------

# Option A: Process the final combined known benchmark windows used for training
INPUT_FILES = [
    "raw_known_full_plusfb_windows.jsonl",
]

# Final unseen FunctionBench validation preset
# INPUT_FILES = [
#     "raw_functionbench_download_upload_unseen_windows.jsonl",
# ]

# Option B: Process multiple files (uncomment and list them)
# INPUT_FILES = [
#     "file1.json",
#     "file2.json",
#     "file3.json"
# ]

# Option C: Process ALL json/jsonl files in a folder (uncomment)
# INPUT_FOLDER = "raw_logs"
# INPUT_FILES = None

OUTPUT_CSV = "prepared_known_full_plusfb.csv"

# ----------------------------
# Expected keys in each record
# ----------------------------
REQUIRED_KEYS = [
    "platform", "workload", "run_id", "invocation_id",
    "mem_limit_mb", "cold_start", "concurrency", "queue_delay_ms",
    "duration_ms", "cpu_time_ms", "rss_mb", "peak_rss_mb",
    "io_read_bytes", "io_write_bytes", "energy_joules"
]

OPTIONAL_KEYS = [
    "display_name",
    "suite",
    "workload_type",
    "partition",
    "resource_profile",
    "resource_profile_index",
    "resource_profile_label",
    "plot_workload_name",
    "omp_threads",
    "benchmark_name",
    "input_profile",
    "package_name",
    "build_config",
    "target_seconds",
    "target_iterations",
    "work_mode",
    "elapsed_seconds",
    "iterations_completed",
    "command_runs",
    "cpu_limit",
    "idle_gap_ms",
    "run_index",
    "launch_overhead_ms",
    "cpu_user_time_ms",
    "cpu_system_time_ms",
    "cpu_util_pct",
    "cpu_nr_periods",
    "cpu_nr_throttled",
    "cpu_throttled_ms",
    "cpu_throttled_pct",
    "memory_current_mb",
    "memory_peak_mb",
    "memory_avg_mb",
    "memory_util_pct",
    "memory_peak_util_pct",
    "memory_avg_util_pct",
    "memory_max_events",
    "memory_high_events",
    "memory_oom_events",
    "memory_oom_kill_events",
    "cpu_pressure_some_avg10",
    "cpu_pressure_full_avg10",
    "memory_pressure_some_avg10",
    "memory_pressure_full_avg10",
    "queue_length",
    "queue_signal_available",
    "monitor_sample_count",
    "instrumentation_mode",
    "rapl_wrapped",
    "window_index",
    "window_start_ms",
    "window_end_ms",
    "sample_kind",
    "service_time_ms",
    "throughput_ops_per_s",
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
    raw_env_input = os.environ.get("PREPARE_INPUT_FILES")
    if raw_env_input:
        input_files = [item.strip() for item in raw_env_input.split(",") if item.strip()]
    else:
        input_files = INPUT_FILES

    files = []
    for name in input_files:
        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p.resolve()}")
        files.append(p)
    return files


def main():
    output_csv = os.environ.get("PREPARE_OUTPUT_CSV", OUTPUT_CSV)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    if not files:
        raise RuntimeError("No input files found. Check INPUT_FILES / INPUT_FOLDER.")

    print("Processing:")
    for f in files:
        print("  -", f)

    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = REQUIRED_KEYS + OPTIONAL_KEYS
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file in files:
            for rec in read_json_records(file):
                row = {k: rec.get(k, None) for k in fieldnames}
                writer.writerow(row)
                rows_written += 1

    print(f"[OK] Wrote {rows_written} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()

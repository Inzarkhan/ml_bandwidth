import json
import random
import time
import os

from sebs_container_runner import run_instrumented_container, validate_energy_access

# Path to workloads
HOST_WORKLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../serverless-exp/workloads"))
HOST_SEBS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "serverless-benchmarks"))
IMAGE_NAME = "serverless-runner:ubuntu22" 
OUTPUT_FILE = os.environ.get(
    "SEBS_UNSEEN_OUTPUT",
    os.environ.get(
        "BENCH_UNSEEN_OUTPUT",
        "/home/said/Pictures/Paper_Conference/Paper_Micro/Codes/raw_sebs_unseen.jsonl",
    ),
)
WINDOWS_OUTPUT = os.environ.get(
    "SEBS_UNSEEN_WINDOWS_OUTPUT",
    "/home/said/Pictures/Paper_Conference/Paper_Micro/Codes/raw_sebs_unseen_windows.jsonl",
)
MEMORY_SIZES = [128, 256, 384, 512, 768, 1024]
# Default to the longer steady-state run length used for analysis.
TARGET_SECONDS = 120
WORKLOAD_SPECS = {
    "sebs_compression_unseen": {
        "script": "sebs_compression_unseen.py",
        "display_name": "Utility (compression)",
        "suite": "SeBS",
        "workload_type": "utility",
        "partition": "unseen",
        "benchmark_name": "compression",
    },
    "sebs_graph_pagerank_unseen": {
        "script": "sebs_graph_pagerank_unseen.py",
        "display_name": "Scientific (graph-pagerank)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "unseen",
        "benchmark_name": "graph-pagerank",
    },
    "sebs_graph_mst_unseen": {
        "script": "sebs_graph_mst_unseen.py",
        "display_name": "Scientific (graph-mst)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "unseen",
        "benchmark_name": "graph-mst",
    },
    "sebs_uploader_unseen": {
        "script": "sebs_uploader_unseen.py",
        "display_name": "Web (uploader)",
        "suite": "SeBS",
        "workload_type": "web",
        "partition": "unseen",
        "benchmark_name": "uploader",
    },
    "sebs_video_processing_unseen": {
        "script": "sebs_video_processing_unseen.py",
        "display_name": "Multimedia (video-processing)",
        "suite": "SeBS",
        "workload_type": "multimedia",
        "partition": "unseen",
        "benchmark_name": "video-processing",
    },
    "sebs_dna_visualisation_unseen": {
        "script": "sebs_dna_visualisation_unseen.py",
        "display_name": "Scientific (dna-visualisation)",
        "suite": "SeBS",
        "workload_type": "scientific",
        "partition": "unseen",
        "benchmark_name": "dna-visualisation",
    },
}

RESOURCE_PROFILE_MAP = {
    "sebs_compression_unseen": ("cpu", 1),
    "sebs_video_processing_unseen": ("cpu", 2),
    "sebs_graph_pagerank_unseen": ("memory", 1),
    "sebs_graph_mst_unseen": ("memory", 3),
    "sebs_uploader_unseen": ("mixed", 5),
    "sebs_dna_visualisation_unseen": ("mixed", 6),
}

RESOURCE_PROFILE_PREFIX = {
    "cpu": "cpu",
    "memory": "mem",
    "mixed": "mix",
}

for workload_key, (resource_profile, resource_profile_index) in RESOURCE_PROFILE_MAP.items():
    spec = WORKLOAD_SPECS[workload_key]
    profile_prefix = RESOURCE_PROFILE_PREFIX[resource_profile]
    resource_profile_label = f"{profile_prefix}{resource_profile_index}"
    spec["resource_profile"] = resource_profile
    spec["resource_profile_index"] = resource_profile_index
    spec["resource_profile_label"] = resource_profile_label
    spec["plot_workload_name"] = f"{spec['benchmark_name']}_{resource_profile_label}"

NEW_WORKLOADS = [
    "sebs_graph_mst_unseen",
    "sebs_uploader_unseen",
    "sebs_video_processing_unseen",
    "sebs_dna_visualisation_unseen",
]
ITERATIONS = 1
CPU_LIMITS = [1.0]
SAMPLE_INTERVAL_MS = 250
TARGET_ITERATIONS = 0
RANDOMIZE_MEMORY_ORDER = True
MEMORY_ORDER_SEED = 42
HOST_BUDGET_SECONDS = 0.0


def parse_int_env(names, default_value):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        return int(raw)
    return default_value


def parse_float_env(names, default_value):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        return float(raw)
    return default_value


def parse_list_env(names, default_values):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        return [item.strip() for item in raw.split(",") if item.strip()]
    return default_values


def parse_float_list_env(names, default_values):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    return default_values


def parse_int_list_env(names, default_values):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return default_values


def parse_bool_env(names, default_value=False):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default_value


def configured_memory_sizes():
    return parse_int_list_env(["SEBS_MEMORY_SIZES"], MEMORY_SIZES)


def ordered_configurations(workload, run_index):
    base_configs = [
        (mem_mb, float(cpu_limit))
        for cpu_limit in parse_float_list_env(["SEBS_CPU_LIMITS"], CPU_LIMITS)
        for mem_mb in configured_memory_sizes()
    ]
    randomize = parse_bool_env(["SEBS_RANDOMIZE_MEMORY_ORDER"], RANDOMIZE_MEMORY_ORDER)
    if not randomize:
        return base_configs

    seed_value = os.environ.get("SEBS_MEMORY_ORDER_SEED", str(MEMORY_ORDER_SEED))
    rng = random.Random(f"{seed_value}:{workload}:{run_index}")
    rng.shuffle(base_configs)
    return base_configs


def initialize_output_file(path, mode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mode == "w":
        with open(path, "w", encoding="utf-8"):
            pass


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()

def run_container(workload, mem_mb, cpu_limit, target_seconds, run_index=1, idle_gap_ms=0.0, target_iterations=None):
    spec = WORKLOAD_SPECS[workload]
    return run_instrumented_container(
        workload=workload,
        script_name=spec["script"],
        spec=spec,
        mem_mb=mem_mb,
        target_seconds=target_seconds,
        image_name=IMAGE_NAME,
        host_workload_dir=HOST_WORKLOAD_DIR,
        host_sebs_dir=HOST_SEBS_DIR,
        cpu_limit=cpu_limit,
        run_index=run_index,
        idle_gap_ms=idle_gap_ms,
        sample_interval_ms=parse_int_env(["SEBS_SAMPLE_INTERVAL_MS"], SAMPLE_INTERVAL_MS),
        target_iterations=target_iterations,
    )

def main():
    target_seconds = parse_int_env(["SEBS_TARGET_SECONDS", "MIBENCH_TARGET_SECONDS"], TARGET_SECONDS)
    target_iterations = parse_int_env(["SEBS_TARGET_ITERATIONS"], TARGET_ITERATIONS)
    target_iterations = target_iterations if target_iterations > 0 else None
    iterations = parse_int_env(["SEBS_RUNS", "MIBENCH_RUNS"], ITERATIONS)
    selected_workloads = parse_list_env(
        ["SEBS_UNSEEN_WORKLOADS", "MIBENCH_UNSEEN_WORKLOADS"],
        NEW_WORKLOADS,
    )
    append_mode = parse_bool_env(["SEBS_APPEND", "MIBENCH_APPEND"], False)
    host_budget_seconds = parse_float_env(["SEBS_HOST_BUDGET_SECONDS"], HOST_BUDGET_SECONDS)

    invalid = [wl for wl in selected_workloads if wl not in WORKLOAD_SPECS]
    if invalid:
        print(f"Error: Unknown unseen workloads requested: {invalid}")
        return

    if target_iterations is not None:
        print(f"--- COLLECTING SEBS UNSEEN SERVERLESS DATA ({target_iterations} Iterations / Fixed-Work Mode) ---")
    else:
        print(f"--- COLLECTING SEBS UNSEEN SERVERLESS DATA ({target_seconds}s Duration) ---")
    if not os.path.exists(HOST_WORKLOAD_DIR):
        print(f"Error: {HOST_WORKLOAD_DIR} not found.")
        return
    if not os.path.exists(HOST_SEBS_DIR):
        print(f"Error: {HOST_SEBS_DIR} not found.")
        return

    try:
        validate_energy_access()
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    results = []
    window_results = []
    previous_end_time = None
    mode = "a" if append_mode else "w"
    initialize_output_file(OUTPUT_FILE, mode)
    initialize_output_file(WINDOWS_OUTPUT, mode)
    collection_start_time = time.time()
    expected_run_seconds = (
        parse_float_env(["SEBS_ESTIMATED_RUN_SECONDS"], float(target_seconds) + 2.0)
        if target_iterations is None
        else parse_float_env(["SEBS_ESTIMATED_RUN_SECONDS"], 10.0)
    )
    budget_exhausted = False
    
    for wl in selected_workloads:
        if budget_exhausted:
            break
        spec = WORKLOAD_SPECS[wl]
        for i in range(iterations):
            if budget_exhausted:
                break
            config_order = ordered_configurations(wl, i + 1)
            print(
                f"\nConfiguration order for {spec['display_name']} (run {i+1}): "
                f"{[(mem, cpu) for mem, cpu in config_order]}"
            )
            for mem, cpu_limit in config_order:
                elapsed_budget = time.time() - collection_start_time
                if host_budget_seconds > 0 and (elapsed_budget + expected_run_seconds) > host_budget_seconds:
                    print(
                        f"\n[STOP] Host budget reached for unseen collection "
                        f"({elapsed_budget:.1f}s elapsed, budget {host_budget_seconds:.1f}s)."
                    )
                    budget_exhausted = True
                    break
                print(f"\nBenchmarking {spec['display_name']} @ {mem}MB / CPU {cpu_limit:.2f} ...")
                now = time.time()
                idle_gap_ms = 0.0 if previous_end_time is None else max((now - previous_end_time) * 1000.0, 0.0)
                data = run_container(
                    wl,
                    mem,
                    cpu_limit,
                    target_seconds,
                    run_index=i + 1,
                    idle_gap_ms=idle_gap_ms,
                    target_iterations=target_iterations,
                )
                if data:
                    aggregate_record = data["aggregate_record"]
                    print(
                        f"  Run {i+1}: {aggregate_record['energy_joules']:.2f} J | "
                        f"{aggregate_record['duration_ms']:.0f} ms | CPU {aggregate_record['cpu_limit']:.2f}"
                    )
                    results.append(aggregate_record)
                    window_results.extend(data.get("window_records", []))
                    append_jsonl(OUTPUT_FILE, aggregate_record)
                    for row in data.get("window_records", []):
                        append_jsonl(WINDOWS_OUTPUT, row)
                    previous_end_time = time.time()
                time.sleep(1) 
    print(f"\n[DONE] Saved to {OUTPUT_FILE}")
    if window_results:
        print(f"[DONE] Window telemetry saved to {WINDOWS_OUTPUT}")

if __name__ == "__main__":
    main()

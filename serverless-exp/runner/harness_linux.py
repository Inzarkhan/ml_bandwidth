import argparse, json, os, subprocess, time, uuid, re

# Parse /usr/bin/time -v output from stderr
def parse_time_v(stderr_text: str):
    out = {}
    # Example lines:
    # "User time (seconds): 0.12"
    # "System time (seconds): 0.03"
    # "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.20"
    # "Maximum resident set size (kbytes): 12345"
    # "File system inputs: 0"
    # "File system outputs: 16"
    patterns = {
        "user_s": r"User time \(seconds\):\s*([0-9.]+)",
        "sys_s": r"System time \(seconds\):\s*([0-9.]+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*(\d+)",
        "fs_in": r"File system inputs:\s*(\d+)",
        "fs_out": r"File system outputs:\s*(\d+)",
        "elapsed": r"Elapsed \(wall clock\) time.*:\s*([0-9:.]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, stderr_text)
        if m:
            out[k] = m.group(1)

    # Convert elapsed like "0:00.20" or "1:02.03" or "0:01"
    elapsed_s = None
    if "elapsed" in out:
        s = out["elapsed"].strip()
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h = int(parts[0]); m = int(parts[1]); sec = float(parts[2])
                elapsed_s = h*3600 + m*60 + sec
            elif len(parts) == 2:
                m = int(parts[0]); sec = float(parts[1])
                elapsed_s = m*60 + sec
            else:
                elapsed_s = float(parts[0])
        except Exception:
            elapsed_s = None

    result = {
        "cpu_time_ms": None,
        "duration_ms": None,
        "peak_rss_mb": None,
        "io_read_bytes": None,
        "io_write_bytes": None,
    }

    if "user_s" in out or "sys_s" in out:
        user = float(out.get("user_s", 0.0))
        sys = float(out.get("sys_s", 0.0))
        result["cpu_time_ms"] = (user + sys) * 1000.0

    if elapsed_s is not None:
        result["duration_ms"] = elapsed_s * 1000.0

    if "max_rss_kb" in out:
        result["peak_rss_mb"] = int(out["max_rss_kb"]) / 1024.0

    # time -v reports filesystem inputs/outputs in blocks (usually 1K blocks), not bytes.
    # We'll store as "io_*_blocks" to avoid lying about units.
    if "fs_in" in out:
        result["io_read_blocks"] = int(out["fs_in"])
    if "fs_out" in out:
        result["io_write_blocks"] = int(out["fs_out"])

    return result

def run_once(workload_cmd, meta):
    invocation_id = str(uuid.uuid4())

    # We measure wall time ourselves too (backup)
    t0 = time.time()

    # Wrap workload with /usr/bin/time -v
    cmd = ["/usr/bin/time", "-v"] + workload_cmd
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()

    t1 = time.time()
    wall_ms_backup = (t1 - t0) * 1000.0

    stats = parse_time_v(err)

    record = {
        **meta,
        "invocation_id": invocation_id,
        "exit_code": p.returncode,
        "duration_ms": stats["duration_ms"] if stats["duration_ms"] is not None else wall_ms_backup,
        "cpu_time_ms": stats["cpu_time_ms"],
        "peak_rss_mb": stats["peak_rss_mb"],
        "rss_mb": stats["peak_rss_mb"],
        "io_read_bytes": 0,
        "io_write_bytes": 0,
        "energy_joules": 0.0,
        "io_read_blocks": stats.get("io_read_blocks"),
        "io_write_blocks": stats.get("io_write_blocks"),
    }

    return record, out, err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="python file name without .py, e.g. cpuintensive")
    ap.add_argument("--mem_limit_mb", type=int, required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--cold_every", type=int, default=1, help="mark cold_start every N runs")
    ap.add_argument("--log", default="/logs/raw.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    run_id = str(uuid.uuid4())
    platform = "docker_ubuntu22_on_ubuntu20"

    workload_cmd = ["python3", f"/app/workloads/{args.workload}.py"]

    with open(args.log, "a") as f:
        for i in range(args.runs):
            cold = (i % args.cold_every == 0)
            meta = {
                "platform": platform,
                "workload": args.workload,
                "run_id": run_id,
                "mem_limit_mb": args.mem_limit_mb,
                "cold_start": bool(cold),
                "concurrency": 1,
                "queue_delay_ms": 0,
            }
            rec, out, err = run_once(workload_cmd, meta)
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()

import argparse, json, os, subprocess, time, uuid, re

WORKLOADS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_rapl_energy_mj():
    rapl_path = "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj"
    try:
        if os.path.exists(rapl_path):
            with open(rapl_path, 'r') as f:
                return int(f.read().strip())
    except: pass
    return None

def parse_time_v(stderr_text):
    out = {}
    patterns = {
        "user_s": r"User time \(seconds\):\s*([0-9.]+)",
        "sys_s": r"System time \(seconds\):\s*([0-9.]+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*(\d+)",
        "elapsed": r"Elapsed \(wall clock\) time.*:\s*([0-9:.]+)",
        "fs_in": r"File system inputs:\s*(\d+)",
        "fs_out": r"File system outputs:\s*(\d+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, stderr_text)
        if m: out[k] = m.group(1)
        
    elapsed_s = None
    if "elapsed" in out:
        parts = out["elapsed"].strip().split(":")
        try:
            if len(parts) == 3: elapsed_s = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
            elif len(parts) == 2: elapsed_s = int(parts[0])*60 + float(parts[1])
            else: elapsed_s = float(parts[0])
        except: pass

    return {
        "cpu_time_ms": (float(out.get("user_s", 0)) + float(out.get("sys_s", 0))) * 1000,
        "duration_ms": elapsed_s * 1000 if elapsed_s else None,
        "peak_rss_mb": int(out.get("max_rss_kb", 0)) / 1024.0,
        "io_read_blocks": int(out.get("fs_in", 0)),
        "io_write_blocks": int(out.get("fs_out", 0))
    }

def run_once(workload_cmd, meta):
    t0 = time.time()
    e_start = get_rapl_energy_mj()
    
    cmd = ["/usr/bin/time", "-v"] + workload_cmd
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    
    e_end = get_rapl_energy_mj()
    t1 = time.time()
    
    stats = parse_time_v(err)
    
    energy_joules = 0.0
    if e_start is not None and e_end is not None and e_end > e_start:
         energy_joules = (e_end - e_start) / 1e6
    
    result = {
        "platform": meta.get("platform", "docker_local"),
        "workload": meta.get("workload", "unknown"),
        "run_id": meta.get("run_id", "unknown"),
        "invocation_id": str(uuid.uuid4()),
        "mem_limit_mb": meta.get("mem_limit_mb", 0),
        "cold_start": meta.get("cold_start", False),
        "concurrency": 1,
        "queue_delay_ms": 0,
        "duration_ms": stats["duration_ms"] or (t1-t0)*1000,
        "cpu_time_ms": stats["cpu_time_ms"],
        "peak_rss_mb": stats["peak_rss_mb"] or 0.0,
        "rss_mb": stats["peak_rss_mb"] or 0.0,
        "io_read_bytes": stats.get("io_read_blocks", 0) * 512,
        "io_write_bytes": stats.get("io_write_blocks", 0) * 512,
        "energy_joules": energy_joules
    }
    for optional_key in ("display_name", "suite", "workload_type", "partition"):
        if meta.get(optional_key) is not None:
            result[optional_key] = meta[optional_key]
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--mem_limit_mb", type=int, required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--cold_every", type=int, default=1)
    ap.add_argument("--log", default="/logs/raw.jsonl")
    ap.add_argument("--workload_event_json", default="")
    ap.add_argument("--display_name", default=None)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--workload_type", default=None)
    ap.add_argument("--partition", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    workload_script = os.path.join(WORKLOADS_DIR, f"{args.workload}.py")
    if not os.path.exists(workload_script):
        raise FileNotFoundError(f"Workload script not found: {workload_script}")

    cmd = ["python3", workload_script]
    if args.workload_event_json:
        cmd.append(args.workload_event_json)
    run_id = str(uuid.uuid4())

    print(f"Starting {args.runs} runs of {args.workload}...")

    with open(args.log, "a") as f:
        for i in range(args.runs):
            meta = {
                "platform": "docker_ubuntu22_on_ubuntu20",
                "workload": args.workload,
                "run_id": run_id,
                "mem_limit_mb": args.mem_limit_mb,
                "cold_start": (i % args.cold_every == 0),
                "display_name": args.display_name,
                "suite": args.suite,
                "workload_type": args.workload_type,
                "partition": args.partition,
            }
            res = run_once(cmd, meta)
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"Run {i+1}: {res['energy_joules']:.4f} J")

if __name__ == "__main__":
    main()

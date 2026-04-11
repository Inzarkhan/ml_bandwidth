import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid


RAPL_FILE = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
RAPL_MAX_RANGE_FILE = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"
CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PAGE_SIZE = os.sysconf(os.sysconf_names["SC_PAGE_SIZE"])
MAX_REASONABLE_POWER_W = float(os.environ.get("SEBS_RAPL_MAX_POWER_W", "400.0"))
ENERGY_SANITY_MULTIPLIER = float(os.environ.get("SEBS_RAPL_SANITY_MULTIPLIER", "3.0"))


def read_energy():
    with open(RAPL_FILE, "r", encoding="utf-8") as f:
        return int(f.read().strip())


def read_energy_max_range():
    with open(RAPL_MAX_RANGE_FILE, "r", encoding="utf-8") as f:
        return int(f.read().strip())


def validate_energy_access():
    try:
        _ = read_energy()
        _ = read_energy_max_range()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read RAPL energy counter at {RAPL_FILE} or range file at {RAPL_MAX_RANGE_FILE}. "
            "Run this script with sufficient privileges so energy_joules is measured correctly."
        ) from exc


def parse_summary(stdout_text):
    if not stdout_text:
        return {}

    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _safe_read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _safe_read_int(path):
    text = _safe_read_text(path)
    if text is None or text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_kv_lines(text):
    values = {}
    if not text:
        return values
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            raw_value = parts[1]
            try:
                values[key] = int(raw_value)
            except ValueError:
                try:
                    values[key] = float(raw_value)
                except ValueError:
                    values[key] = raw_value
    return values


def _parse_pressure(text):
    values = {}
    if not text:
        return values
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        prefix = parts[0]
        for item in parts[1:]:
            if "=" not in item:
                continue
            key, raw_value = item.split("=", 1)
            metric_key = f"{prefix}_{key}"
            try:
                values[metric_key] = float(raw_value)
            except ValueError:
                continue
    return values


def _parse_io_v2(text):
    totals = {"rbytes": 0, "wbytes": 0}
    if not text:
        return totals
    for line in text.splitlines():
        for item in line.split()[1:]:
            if "=" not in item:
                continue
            key, raw_value = item.split("=", 1)
            if key not in totals:
                continue
            try:
                totals[key] += int(raw_value)
            except ValueError:
                continue
    return totals


def _parse_io_v1(text):
    totals = {"Read": 0, "Write": 0}
    if not text:
        return totals
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        _, op_name, raw_value = parts
        if op_name not in totals:
            continue
        try:
            totals[op_name] += int(raw_value)
        except ValueError:
            continue
    return totals


def _docker_inspect_value(container_id, template):
    result = subprocess.run(
        ["docker", "inspect", "-f", template, container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _parse_size_to_bytes(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text == "" or text.lower() == "n/a":
        return None
    match = re.match(r"^\s*([0-9.]+)\s*([kmgtp]?i?b)?\s*$", text, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
        "tb": 1000 ** 4,
        "pb": 1000 ** 5,
        "kib": 1024,
        "mib": 1024 ** 2,
        "gib": 1024 ** 3,
        "tib": 1024 ** 4,
        "pib": 1024 ** 5,
    }
    return int(value * multipliers.get(unit, 1))


def _docker_stats_snapshot(container_id):
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    text = result.stdout.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    cpu_raw = str(parsed.get("CPUPerc", "0")).strip().rstrip("%")
    try:
        cpu_pct = float(cpu_raw)
    except ValueError:
        cpu_pct = 0.0

    mem_usage = str(parsed.get("MemUsage", "")).split("/")
    mem_current_bytes = _parse_size_to_bytes(mem_usage[0]) if mem_usage else None

    block_io = str(parsed.get("BlockIO", "")).split("/")
    io_read_bytes = _parse_size_to_bytes(block_io[0]) if block_io else None
    io_write_bytes = _parse_size_to_bytes(block_io[1]) if len(block_io) > 1 else None

    return {
        "docker_cpu_util_pct": cpu_pct,
        "docker_memory_current_bytes": mem_current_bytes,
        "docker_io_read_bytes": io_read_bytes,
        "docker_io_write_bytes": io_write_bytes,
    }


def _safe_read_proc_status_value_kb(pid, key_name):
    text = _safe_read_text(f"/proc/{pid}/status")
    if not text:
        return None
    prefix = f"{key_name}:"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _read_proc_snapshot(pid):
    if not pid:
        return {}

    text = _safe_read_text(f"/proc/{pid}/stat")
    if not text:
        return {}

    right_paren = text.rfind(")")
    if right_paren < 0:
        return {}
    tail = text[right_paren + 2 :].split()
    if len(tail) < 22:
        return {}

    try:
        utime_ticks = int(tail[11])
        stime_ticks = int(tail[12])
        rss_pages = int(tail[21])
    except ValueError:
        return {}

    current_rss_bytes = max(rss_pages, 0) * PAGE_SIZE
    peak_rss_kb = _safe_read_proc_status_value_kb(pid, "VmHWM")
    current_rss_kb = _safe_read_proc_status_value_kb(pid, "VmRSS")
    if current_rss_kb is not None:
        current_rss_bytes = current_rss_kb * 1024

    peak_rss_bytes = current_rss_bytes
    if peak_rss_kb is not None:
        peak_rss_bytes = max(peak_rss_bytes, peak_rss_kb * 1024)

    return {
        "proc_cpu_user_usec": int((utime_ticks * 1_000_000) / CLOCK_TICKS),
        "proc_cpu_system_usec": int((stime_ticks * 1_000_000) / CLOCK_TICKS),
        "proc_cpu_usage_usec": int(((utime_ticks + stime_ticks) * 1_000_000) / CLOCK_TICKS),
        "proc_memory_current_bytes": int(current_rss_bytes),
        "proc_memory_peak_bytes": int(peak_rss_bytes),
    }


def _plausible_energy_upper_bound_uj(duration_ms):
    duration_seconds = max(float(duration_ms), 1.0) / 1000.0
    return duration_seconds * MAX_REASONABLE_POWER_W * 1_000_000.0 * ENERGY_SANITY_MULTIPLIER


def _sanitize_window_energy_deltas_uj(raw_deltas_uj, durations_ms, total_energy_uj, rapl_max_range_uj):
    if not raw_deltas_uj:
        return []

    safe_durations = [max(float(duration_ms), 1.0) for duration_ms in durations_ms]
    total_duration_ms = max(sum(safe_durations), 1.0)

    normalized = []
    for raw_delta_uj, duration_ms in zip(raw_deltas_uj, safe_durations):
        plausible_upper_uj = _plausible_energy_upper_bound_uj(duration_ms)
        energy_delta_uj = None

        if 0.0 <= raw_delta_uj <= plausible_upper_uj:
            energy_delta_uj = raw_delta_uj
        elif raw_delta_uj < 0.0 and rapl_max_range_uj > 0.0:
            wrapped_delta_uj = raw_delta_uj + rapl_max_range_uj
            if (
                0.0 <= wrapped_delta_uj <= plausible_upper_uj
                and rapl_max_range_uj <= (plausible_upper_uj * 2.0)
            ):
                energy_delta_uj = wrapped_delta_uj

        if energy_delta_uj is None:
            energy_share = duration_ms / total_duration_ms
            energy_delta_uj = max(float(total_energy_uj), 0.0) * energy_share

        normalized.append(max(float(energy_delta_uj), 0.0))

    normalized_sum_uj = sum(normalized)
    if total_energy_uj > 0.0 and normalized_sum_uj > 0.0:
        scale = float(total_energy_uj) / normalized_sum_uj
        normalized = [max(value * scale, 0.0) for value in normalized]

    return normalized


def _detect_cgroup_context(pid):
    proc_cgroup = Path(f"/proc/{pid}/cgroup")
    if not proc_cgroup.exists():
        return None

    unified_path = None
    controller_paths = {}
    for line in proc_cgroup.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue
        _, controllers, rel_path = parts
        rel_path = rel_path.lstrip("/")
        if controllers == "":
            unified_path = rel_path
            continue
        for controller in controllers.split(","):
            controller_paths[controller] = rel_path

    cgroup_root = Path("/sys/fs/cgroup")
    if unified_path is not None:
        base = cgroup_root / unified_path
        if base.exists():
            return {"mode": "v2", "base": base}

    context = {"mode": "v1"}
    for controller in ("cpuacct", "cpu", "memory", "blkio"):
        rel_path = controller_paths.get(controller)
        if rel_path is None:
            continue
        base = cgroup_root / controller / rel_path
        if base.exists():
            context[controller] = base

    if any(key in context for key in ("cpuacct", "cpu", "memory", "blkio")):
        return context
    return None


def _read_snapshot(context):
    if context is None:
        return {}

    if context["mode"] == "v2":
        base = context["base"]
        cpu_stat = _parse_kv_lines(_safe_read_text(base / "cpu.stat"))
        memory_events = _parse_kv_lines(_safe_read_text(base / "memory.events"))
        cpu_pressure = _parse_pressure(_safe_read_text(base / "cpu.pressure"))
        memory_pressure = _parse_pressure(_safe_read_text(base / "memory.pressure"))
        io_totals = _parse_io_v2(_safe_read_text(base / "io.stat"))
        return {
            "cpu_usage_usec": int(cpu_stat.get("usage_usec", 0)),
            "cpu_user_usec": int(cpu_stat.get("user_usec", 0)),
            "cpu_system_usec": int(cpu_stat.get("system_usec", 0)),
            "cpu_nr_periods": int(cpu_stat.get("nr_periods", 0)),
            "cpu_nr_throttled": int(cpu_stat.get("nr_throttled", 0)),
            "cpu_throttled_usec": int(cpu_stat.get("throttled_usec", 0)),
            "memory_current_bytes": int(_safe_read_int(base / "memory.current") or 0),
            "memory_peak_bytes": int(_safe_read_int(base / "memory.peak") or 0),
            "memory_high_events": int(memory_events.get("high", 0)),
            "memory_max_events": int(memory_events.get("max", 0)),
            "memory_oom_events": int(memory_events.get("oom", 0)),
            "memory_oom_kill_events": int(memory_events.get("oom_kill", 0)),
            "cpu_pressure_some_avg10": float(cpu_pressure.get("some_avg10", 0.0)),
            "cpu_pressure_full_avg10": float(cpu_pressure.get("full_avg10", 0.0)),
            "memory_pressure_some_avg10": float(memory_pressure.get("some_avg10", 0.0)),
            "memory_pressure_full_avg10": float(memory_pressure.get("full_avg10", 0.0)),
            "io_read_bytes": int(io_totals.get("rbytes", 0)),
            "io_write_bytes": int(io_totals.get("wbytes", 0)),
        }

    cpuacct_base = context.get("cpuacct")
    cpu_base = context.get("cpu")
    memory_base = context.get("memory")
    blkio_base = context.get("blkio")

    cpu_usage_ns = _safe_read_int(cpuacct_base / "cpuacct.usage") if cpuacct_base else 0
    cpuacct_stat = _parse_kv_lines(_safe_read_text(cpuacct_base / "cpuacct.stat")) if cpuacct_base else {}
    cpu_stat = _parse_kv_lines(_safe_read_text(cpu_base / "cpu.stat")) if cpu_base else {}
    memory_events = {
        "max": _safe_read_int(memory_base / "memory.failcnt") if memory_base else 0,
    }
    io_totals = _parse_io_v1(
        _safe_read_text(blkio_base / "blkio.throttle.io_service_bytes_recursive")
    ) if blkio_base else {"Read": 0, "Write": 0}

    return {
        "cpu_usage_usec": int((cpu_usage_ns or 0) / 1000),
        "cpu_user_usec": int((int(cpuacct_stat.get("user", 0)) * 1_000_000) / CLOCK_TICKS),
        "cpu_system_usec": int((int(cpuacct_stat.get("system", 0)) * 1_000_000) / CLOCK_TICKS),
        "cpu_nr_periods": int(cpu_stat.get("nr_periods", 0)),
        "cpu_nr_throttled": int(cpu_stat.get("nr_throttled", 0)),
        "cpu_throttled_usec": int(int(cpu_stat.get("throttled_time", 0)) / 1000),
        "memory_current_bytes": int(_safe_read_int(memory_base / "memory.usage_in_bytes") or 0),
        "memory_peak_bytes": int(_safe_read_int(memory_base / "memory.max_usage_in_bytes") or 0),
        "memory_high_events": 0,
        "memory_max_events": int(memory_events["max"] or 0),
        "memory_oom_events": 0,
        "memory_oom_kill_events": 0,
        "cpu_pressure_some_avg10": 0.0,
        "cpu_pressure_full_avg10": 0.0,
        "memory_pressure_some_avg10": 0.0,
        "memory_pressure_full_avg10": 0.0,
        "io_read_bytes": int(io_totals.get("Read", 0)),
        "io_write_bytes": int(io_totals.get("Write", 0)),
    }


class ContainerMetricsMonitor:
    def __init__(self, container_id, mem_limit_mb, cpu_limit, sample_interval_s=0.25):
        self.container_id = container_id
        self.mem_limit_mb = float(mem_limit_mb)
        self.cpu_limit = float(cpu_limit)
        self.sample_interval_s = sample_interval_s
        self.pid = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.context = None
        self.use_docker_stats = os.environ.get("SEBS_USE_DOCKER_STATS", "").strip().lower() in {"1", "true", "yes", "on"}
        self.initial_snapshot = {}
        self.final_snapshot = {}
        self.last_snapshot = {}
        self.last_memory_bytes = 0
        self.max_sampled_memory_bytes = 0
        self.memory_sum_bytes = 0.0
        self.sample_count = 0
        self.cpu_pressure_some_avg10 = 0.0
        self.cpu_pressure_full_avg10 = 0.0
        self.memory_pressure_some_avg10 = 0.0
        self.memory_pressure_full_avg10 = 0.0
        self.cpu_util_samples = []
        self.io_read_samples = []
        self.io_write_samples = []
        self.samples = []
        self.start_monotonic = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self):
        self.start_monotonic = time.monotonic()
        deadline = time.time() + 5.0
        while not self.stop_event.is_set() and time.time() < deadline:
            pid_raw = _docker_inspect_value(self.container_id, "{{.State.Pid}}")
            try:
                pid = int(pid_raw) if pid_raw else 0
            except ValueError:
                pid = 0
            if pid > 0:
                self.pid = pid
                self.context = _detect_cgroup_context(pid)
                if self.context is not None or self.pid:
                    self.initial_snapshot = _read_snapshot(self.context)
                    proc_snapshot = _read_proc_snapshot(self.pid)
                    if proc_snapshot:
                        self.initial_snapshot = {**self.initial_snapshot, **proc_snapshot}
                    if self.use_docker_stats:
                        initial_stats = _docker_stats_snapshot(self.container_id)
                        if initial_stats:
                            self.initial_snapshot = {**self.initial_snapshot, **initial_stats}
                    self.initial_snapshot["host_energy_uj"] = read_energy()
                    self.initial_snapshot["timestamp_ms"] = 0.0
                    self.last_snapshot = dict(self.initial_snapshot)
                    self.samples.append(dict(self.initial_snapshot))
                    break
            time.sleep(0.05)

        while not self.stop_event.is_set():
            snapshot = _read_snapshot(self.context)
            proc_snapshot = _read_proc_snapshot(self.pid)
            if proc_snapshot:
                snapshot = {**snapshot, **proc_snapshot}
            if self.use_docker_stats:
                stats_snapshot = _docker_stats_snapshot(self.container_id)
                if stats_snapshot:
                    snapshot = {**snapshot, **stats_snapshot}
            snapshot["host_energy_uj"] = read_energy()
            snapshot["timestamp_ms"] = max((time.monotonic() - self.start_monotonic) * 1000.0, 0.0)
            self._consume_snapshot(snapshot)
            time.sleep(self.sample_interval_s)

        self.final_snapshot = _read_snapshot(self.context)
        proc_snapshot = _read_proc_snapshot(self.pid)
        if proc_snapshot:
            self.final_snapshot = {**self.final_snapshot, **proc_snapshot}
        if self.use_docker_stats:
            stats_snapshot = _docker_stats_snapshot(self.container_id)
            if stats_snapshot:
                self.final_snapshot = {**self.final_snapshot, **stats_snapshot}
        self.final_snapshot["host_energy_uj"] = read_energy()
        self.final_snapshot["timestamp_ms"] = max((time.monotonic() - self.start_monotonic) * 1000.0, 0.0)
        self._consume_snapshot(self.final_snapshot)

    def _consume_snapshot(self, snapshot):
        if not snapshot:
            return
        self.last_snapshot = dict(snapshot)
        self.samples.append(dict(snapshot))
        memory_bytes = int(
            snapshot.get("memory_current_bytes", 0)
            or snapshot.get("proc_memory_current_bytes", 0)
            or snapshot.get("docker_memory_current_bytes", 0)
            or 0
        )
        self.last_memory_bytes = memory_bytes
        self.max_sampled_memory_bytes = max(self.max_sampled_memory_bytes, memory_bytes)
        self.memory_sum_bytes += memory_bytes
        self.sample_count += 1
        self.cpu_pressure_some_avg10 = max(
            self.cpu_pressure_some_avg10,
            float(snapshot.get("cpu_pressure_some_avg10", 0.0) or 0.0),
        )
        self.cpu_pressure_full_avg10 = max(
            self.cpu_pressure_full_avg10,
            float(snapshot.get("cpu_pressure_full_avg10", 0.0) or 0.0),
        )
        self.memory_pressure_some_avg10 = max(
            self.memory_pressure_some_avg10,
            float(snapshot.get("memory_pressure_some_avg10", 0.0) or 0.0),
        )
        self.memory_pressure_full_avg10 = max(
            self.memory_pressure_full_avg10,
            float(snapshot.get("memory_pressure_full_avg10", 0.0) or 0.0),
        )
        cpu_pct = snapshot.get("docker_cpu_util_pct")
        if cpu_pct is not None:
            try:
                self.cpu_util_samples.append(float(cpu_pct))
            except (TypeError, ValueError):
                pass
        io_read_bytes = snapshot.get("docker_io_read_bytes")
        if io_read_bytes is not None:
            try:
                self.io_read_samples.append(int(io_read_bytes))
            except (TypeError, ValueError):
                pass
        io_write_bytes = snapshot.get("docker_io_write_bytes")
        if io_write_bytes is not None:
            try:
                self.io_write_samples.append(int(io_write_bytes))
            except (TypeError, ValueError):
                pass

    def metrics(self, duration_ms):
        duration_ms = max(float(duration_ms), 1.0)

        initial = self.initial_snapshot or {}
        final = self.final_snapshot or self.last_snapshot or initial or {}

        def delta(key):
            return max(float(final.get(key, 0.0) or 0.0) - float(initial.get(key, 0.0) or 0.0), 0.0)

        cgroup_cpu_time_ms = delta("cpu_usage_usec") / 1000.0
        proc_cpu_time_ms = delta("proc_cpu_usage_usec") / 1000.0
        cpu_time_ms = cgroup_cpu_time_ms if cgroup_cpu_time_ms > 0.0 else proc_cpu_time_ms

        cgroup_cpu_user_time_ms = delta("cpu_user_usec") / 1000.0
        proc_cpu_user_time_ms = delta("proc_cpu_user_usec") / 1000.0
        cpu_user_time_ms = (
            cgroup_cpu_user_time_ms if cgroup_cpu_user_time_ms > 0.0 else proc_cpu_user_time_ms
        )

        cgroup_cpu_system_time_ms = delta("cpu_system_usec") / 1000.0
        proc_cpu_system_time_ms = delta("proc_cpu_system_usec") / 1000.0
        cpu_system_time_ms = (
            cgroup_cpu_system_time_ms if cgroup_cpu_system_time_ms > 0.0 else proc_cpu_system_time_ms
        )
        cpu_throttled_ms = delta("cpu_throttled_usec") / 1000.0

        peak_memory_bytes = max(
            int(final.get("memory_peak_bytes", 0) or 0),
            int(final.get("proc_memory_peak_bytes", 0) or 0),
            int(self.max_sampled_memory_bytes or 0),
            int(final.get("memory_current_bytes", 0) or 0),
            int(final.get("proc_memory_current_bytes", 0) or 0),
        )
        current_memory_bytes = int(
            final.get("memory_current_bytes", 0)
            or final.get("proc_memory_current_bytes", 0)
            or self.last_memory_bytes
            or 0
        )
        avg_memory_bytes = (
            (self.memory_sum_bytes / self.sample_count) if self.sample_count else float(current_memory_bytes)
        )

        memory_current_mb = current_memory_bytes / (1024.0 * 1024.0)
        memory_peak_mb = peak_memory_bytes / (1024.0 * 1024.0)
        memory_avg_mb = avg_memory_bytes / (1024.0 * 1024.0)

        cpu_capacity_ms = max(duration_ms * max(self.cpu_limit, 1e-9), 1.0)
        mem_capacity_mb = max(self.mem_limit_mb, 1e-9)
        sampled_cpu_util_pct = (
            sum(self.cpu_util_samples) / len(self.cpu_util_samples) if self.cpu_util_samples else 0.0
        )
        if cpu_time_ms <= 0.0 and sampled_cpu_util_pct > 0.0:
            cpu_time_ms = (sampled_cpu_util_pct / 100.0) * cpu_capacity_ms
            cpu_user_time_ms = cpu_time_ms
            cpu_system_time_ms = 0.0

        io_read_delta = int(delta("io_read_bytes"))
        io_write_delta = int(delta("io_write_bytes"))
        if io_read_delta <= 0 and len(self.io_read_samples) >= 2:
            io_read_delta = max(self.io_read_samples[-1] - self.io_read_samples[0], 0)
        if io_write_delta <= 0 and len(self.io_write_samples) >= 2:
            io_write_delta = max(self.io_write_samples[-1] - self.io_write_samples[0], 0)

        return {
            "cpu_time_ms": cpu_time_ms,
            "cpu_user_time_ms": cpu_user_time_ms,
            "cpu_system_time_ms": cpu_system_time_ms,
            "cpu_util_pct": (cpu_time_ms / cpu_capacity_ms) * 100.0,
            "cpu_nr_periods": int(delta("cpu_nr_periods")),
            "cpu_nr_throttled": int(delta("cpu_nr_throttled")),
            "cpu_throttled_ms": cpu_throttled_ms,
            "cpu_throttled_pct": (cpu_throttled_ms / cpu_capacity_ms) * 100.0,
            "peak_rss_mb": memory_peak_mb,
            "rss_mb": memory_current_mb,
            "memory_current_mb": memory_current_mb,
            "memory_peak_mb": memory_peak_mb,
            "memory_avg_mb": memory_avg_mb,
            "memory_util_pct": (memory_current_mb / mem_capacity_mb) * 100.0,
            "memory_peak_util_pct": (memory_peak_mb / mem_capacity_mb) * 100.0,
            "memory_avg_util_pct": (memory_avg_mb / mem_capacity_mb) * 100.0,
            "memory_max_events": int(delta("memory_max_events")),
            "memory_high_events": int(delta("memory_high_events")),
            "memory_oom_events": int(delta("memory_oom_events")),
            "memory_oom_kill_events": int(delta("memory_oom_kill_events")),
            "cpu_pressure_some_avg10": self.cpu_pressure_some_avg10,
            "cpu_pressure_full_avg10": self.cpu_pressure_full_avg10,
            "memory_pressure_some_avg10": self.memory_pressure_some_avg10,
            "memory_pressure_full_avg10": self.memory_pressure_full_avg10,
            "io_read_bytes": io_read_delta,
            "io_write_bytes": io_write_delta,
            "queue_length": 0.0,
            "queue_signal_available": 0,
            "monitor_sample_count": int(self.sample_count),
        }

    def window_records(
        self,
        *,
        run_id,
        invocation_id,
        workload,
        spec,
        mem_mb,
        cpu_limit,
        target_seconds,
        run_index,
        idle_gap_ms,
        summary,
        total_energy_uj,
        rapl_max_range_uj,
    ):
        if len(self.samples) < 2:
            return []

        elapsed_seconds = None
        if "elapsed_seconds" in summary:
            try:
                elapsed_seconds = float(summary["elapsed_seconds"])
            except (TypeError, ValueError):
                elapsed_seconds = None

        iterations_completed = None
        if "iterations_completed" in summary:
            try:
                iterations_completed = int(summary["iterations_completed"])
            except (TypeError, ValueError):
                iterations_completed = None

        command_runs = None
        if "command_runs" in summary:
            try:
                command_runs = int(summary["command_runs"])
            except (TypeError, ValueError):
                command_runs = None

        service_time_ms = None
        throughput_ops_per_s = None
        if elapsed_seconds and iterations_completed and iterations_completed > 0:
            service_time_ms = (elapsed_seconds * 1000.0) / iterations_completed
            throughput_ops_per_s = iterations_completed / elapsed_seconds

        raw_energy_deltas_uj = []
        window_durations_ms = []
        for idx in range(1, len(self.samples)):
            prev = self.samples[idx - 1]
            cur = self.samples[idx]
            window_duration_ms = max(
                float(cur.get("timestamp_ms", 0.0)) - float(prev.get("timestamp_ms", 0.0)),
                1.0,
            )
            window_durations_ms.append(window_duration_ms)
            raw_energy_deltas_uj.append(
                float(cur.get("host_energy_uj", 0.0) or 0.0) - float(prev.get("host_energy_uj", 0.0) or 0.0)
            )

        sanitized_energy_deltas_uj = _sanitize_window_energy_deltas_uj(
            raw_energy_deltas_uj,
            window_durations_ms,
            total_energy_uj,
            rapl_max_range_uj,
        )

        window_rows = []
        mem_capacity_mb = max(float(mem_mb), 1e-9)
        for idx in range(1, len(self.samples)):
            prev = self.samples[idx - 1]
            cur = self.samples[idx]
            window_duration_ms = window_durations_ms[idx - 1]
            cpu_capacity_ms = max(window_duration_ms * max(float(cpu_limit), 1e-9), 1.0)

            cgroup_cpu_usage_delta_usec = max(
                float(cur.get("cpu_usage_usec", 0.0) or 0.0) - float(prev.get("cpu_usage_usec", 0.0) or 0.0),
                0.0,
            )
            proc_cpu_usage_delta_usec = max(
                float(cur.get("proc_cpu_usage_usec", 0.0) or 0.0) - float(prev.get("proc_cpu_usage_usec", 0.0) or 0.0),
                0.0,
            )
            cpu_time_ms = (
                cgroup_cpu_usage_delta_usec if cgroup_cpu_usage_delta_usec > 0.0 else proc_cpu_usage_delta_usec
            ) / 1000.0

            prev_cpu_pct = prev.get("docker_cpu_util_pct")
            cur_cpu_pct = cur.get("docker_cpu_util_pct")
            if cpu_time_ms <= 0.0 and prev_cpu_pct is not None and cur_cpu_pct is not None:
                avg_cpu_pct = (float(prev_cpu_pct) + float(cur_cpu_pct)) / 2.0
                cpu_time_ms = (avg_cpu_pct / 100.0) * cpu_capacity_ms
            else:
                avg_cpu_pct = (cpu_time_ms / cpu_capacity_ms) * 100.0

            cgroup_cpu_user_time_ms = max(
                float(cur.get("cpu_user_usec", 0.0) or 0.0) - float(prev.get("cpu_user_usec", 0.0) or 0.0),
                0.0,
            ) / 1000.0
            proc_cpu_user_time_ms = max(
                float(cur.get("proc_cpu_user_usec", 0.0) or 0.0) - float(prev.get("proc_cpu_user_usec", 0.0) or 0.0),
                0.0,
            ) / 1000.0
            cpu_user_time_ms = (
                cgroup_cpu_user_time_ms if cgroup_cpu_user_time_ms > 0.0 else proc_cpu_user_time_ms
            )

            cgroup_cpu_system_time_ms = max(
                float(cur.get("cpu_system_usec", 0.0) or 0.0) - float(prev.get("cpu_system_usec", 0.0) or 0.0),
                0.0,
            ) / 1000.0
            proc_cpu_system_time_ms = max(
                float(cur.get("proc_cpu_system_usec", 0.0) or 0.0) - float(prev.get("proc_cpu_system_usec", 0.0) or 0.0),
                0.0,
            ) / 1000.0
            cpu_system_time_ms = (
                cgroup_cpu_system_time_ms if cgroup_cpu_system_time_ms > 0.0 else proc_cpu_system_time_ms
            )
            if cpu_user_time_ms <= 0.0 and cpu_system_time_ms <= 0.0 and cpu_time_ms > 0.0:
                cpu_user_time_ms = cpu_time_ms

            cpu_throttled_ms = max(
                float(cur.get("cpu_throttled_usec", 0.0) or 0.0) - float(prev.get("cpu_throttled_usec", 0.0) or 0.0),
                0.0,
            ) / 1000.0

            io_read_bytes = max(
                int(cur.get("io_read_bytes", 0) or 0) - int(prev.get("io_read_bytes", 0) or 0),
                0,
            )
            io_write_bytes = max(
                int(cur.get("io_write_bytes", 0) or 0) - int(prev.get("io_write_bytes", 0) or 0),
                0,
            )
            docker_read_prev = prev.get("docker_io_read_bytes")
            docker_read_cur = cur.get("docker_io_read_bytes")
            if io_read_bytes <= 0 and docker_read_prev is not None and docker_read_cur is not None:
                io_read_bytes = max(int(docker_read_cur) - int(docker_read_prev), 0)
            docker_write_prev = prev.get("docker_io_write_bytes")
            docker_write_cur = cur.get("docker_io_write_bytes")
            if io_write_bytes <= 0 and docker_write_prev is not None and docker_write_cur is not None:
                io_write_bytes = max(int(docker_write_cur) - int(docker_write_prev), 0)

            current_memory_bytes = int(
                cur.get("memory_current_bytes", 0)
                or cur.get("proc_memory_current_bytes", 0)
                or cur.get("docker_memory_current_bytes", 0)
                or 0
            )
            peak_memory_bytes = max(
                int(prev.get("memory_peak_bytes", 0) or 0),
                int(prev.get("proc_memory_peak_bytes", 0) or 0),
                int(cur.get("memory_peak_bytes", 0) or 0),
                int(cur.get("proc_memory_peak_bytes", 0) or 0),
                int(
                    prev.get("memory_current_bytes", 0)
                    or prev.get("proc_memory_current_bytes", 0)
                    or prev.get("docker_memory_current_bytes", 0)
                    or 0
                ),
                current_memory_bytes,
            )

            memory_current_mb = current_memory_bytes / (1024.0 * 1024.0)
            memory_peak_mb = peak_memory_bytes / (1024.0 * 1024.0)
            memory_avg_mb = (
                int(
                    prev.get("memory_current_bytes", 0)
                    or prev.get("proc_memory_current_bytes", 0)
                    or prev.get("docker_memory_current_bytes", 0)
                    or 0
                )
                + current_memory_bytes
            ) / (2.0 * 1024.0 * 1024.0)

            row = {
                "platform": "docker_ubuntu22_on_ubuntu20",
                "workload": workload,
                "run_id": run_id,
                "invocation_id": invocation_id,
                "mem_limit_mb": mem_mb,
                "cold_start": False,
                "concurrency": 1,
                "queue_delay_ms": 0.0,
                "duration_ms": window_duration_ms,
                "cpu_time_ms": cpu_time_ms,
                "cpu_user_time_ms": cpu_user_time_ms,
                "cpu_system_time_ms": cpu_system_time_ms,
                "cpu_util_pct": (cpu_time_ms / cpu_capacity_ms) * 100.0,
                "cpu_nr_periods": int(
                    max(float(cur.get("cpu_nr_periods", 0.0) or 0.0) - float(prev.get("cpu_nr_periods", 0.0) or 0.0), 0.0)
                ),
                "cpu_nr_throttled": int(
                    max(float(cur.get("cpu_nr_throttled", 0.0) or 0.0) - float(prev.get("cpu_nr_throttled", 0.0) or 0.0), 0.0)
                ),
                "cpu_throttled_ms": cpu_throttled_ms,
                "cpu_throttled_pct": (cpu_throttled_ms / cpu_capacity_ms) * 100.0,
                "peak_rss_mb": memory_peak_mb,
                "rss_mb": memory_current_mb,
                "memory_current_mb": memory_current_mb,
                "memory_peak_mb": memory_peak_mb,
                "memory_avg_mb": memory_avg_mb,
                "memory_util_pct": (memory_current_mb / mem_capacity_mb) * 100.0,
                "memory_peak_util_pct": (memory_peak_mb / mem_capacity_mb) * 100.0,
                "memory_avg_util_pct": (memory_avg_mb / mem_capacity_mb) * 100.0,
                "memory_max_events": int(
                    max(float(cur.get("memory_max_events", 0.0) or 0.0) - float(prev.get("memory_max_events", 0.0) or 0.0), 0.0)
                ),
                "memory_high_events": int(
                    max(float(cur.get("memory_high_events", 0.0) or 0.0) - float(prev.get("memory_high_events", 0.0) or 0.0), 0.0)
                ),
                "memory_oom_events": int(
                    max(float(cur.get("memory_oom_events", 0.0) or 0.0) - float(prev.get("memory_oom_events", 0.0) or 0.0), 0.0)
                ),
                "memory_oom_kill_events": int(
                    max(float(cur.get("memory_oom_kill_events", 0.0) or 0.0) - float(prev.get("memory_oom_kill_events", 0.0) or 0.0), 0.0)
                ),
                "cpu_pressure_some_avg10": float(cur.get("cpu_pressure_some_avg10", 0.0) or 0.0),
                "cpu_pressure_full_avg10": float(cur.get("cpu_pressure_full_avg10", 0.0) or 0.0),
                "memory_pressure_some_avg10": float(cur.get("memory_pressure_some_avg10", 0.0) or 0.0),
                "memory_pressure_full_avg10": float(cur.get("memory_pressure_full_avg10", 0.0) or 0.0),
                "energy_joules": sanitized_energy_deltas_uj[idx - 1] / 1_000_000.0,
                "io_read_bytes": io_read_bytes,
                "io_write_bytes": io_write_bytes,
                "display_name": spec["display_name"],
                "suite": spec["suite"],
                "workload_type": spec["workload_type"],
                "partition": spec["partition"],
                "benchmark_name": summary.get("benchmark_name", spec.get("benchmark_name")),
                "resource_profile": spec.get("resource_profile"),
                "resource_profile_index": spec.get("resource_profile_index"),
                "resource_profile_label": spec.get("resource_profile_label"),
                "plot_workload_name": spec.get("plot_workload_name"),
                "input_profile": summary.get("input_profile"),
                "package_name": summary.get("package_name"),
                "build_config": summary.get("build_config"),
                "target_seconds": float(summary.get("target_seconds", target_seconds)),
                "target_iterations": int(summary["target_iterations"]) if summary.get("target_iterations") is not None else None,
                "work_mode": summary.get("work_mode", "fixed_duration"),
                "elapsed_seconds": elapsed_seconds,
                "iterations_completed": iterations_completed,
                "command_runs": command_runs,
                "omp_threads": int(summary.get("omp_threads", summary.get("threads", 1))),
                "cpu_limit": float(cpu_limit),
                "idle_gap_ms": float(idle_gap_ms if idx == 1 else 0.0),
                "run_index": int(run_index),
                "instrumentation_mode": "docker_cgroup_sampling_windows",
                "queue_length": 0.0,
                "queue_signal_available": 0,
                "service_time_ms": service_time_ms,
                "throughput_ops_per_s": throughput_ops_per_s,
                "window_index": idx,
                "window_start_ms": float(prev.get("timestamp_ms", 0.0) or 0.0),
                "window_end_ms": float(cur.get("timestamp_ms", 0.0) or 0.0),
                "sample_kind": "telemetry_window",
            }
            window_rows.append(row)

        return window_rows


def run_instrumented_container(
    *,
    workload,
    script_name,
    spec,
    mem_mb,
    target_seconds,
    image_name,
    host_workload_dir,
    host_sebs_dir,
    cpu_limit,
    run_index=1,
    idle_gap_ms=0.0,
    sample_interval_ms=250,
    target_iterations=None,
):
    event_payload = {"target_seconds": target_seconds}
    if target_iterations is not None and int(target_iterations) > 0:
        event_payload["target_iterations"] = int(target_iterations)
    event_json = json.dumps(event_payload)
    create_cmd = [
        "docker",
        "create",
        "--entrypoint",
        "",
        f"--memory={mem_mb}m",
        f"--cpus={cpu_limit}",
        "-v",
        f"{host_workload_dir}:/app",
        "-v",
        f"{host_sebs_dir}:/sebs:ro",
        "-w",
        "/app",
        image_name,
        "python3",
        script_name,
        event_json,
    ]

    created = subprocess.run(create_cmd, capture_output=True, text=True, check=False)
    if created.returncode != 0:
        print(f"  [Error] Failed to create container for {workload}")
        if created.stderr:
            print(f"  STDERR: {created.stderr}")
        return None

    container_id = created.stdout.strip()
    monitor = ContainerMetricsMonitor(container_id, mem_mb, cpu_limit, sample_interval_s=max(sample_interval_ms, 50) / 1000.0)
    start_time = time.time()
    start_energy = read_energy()
    energy_max_range_uj = read_energy_max_range()
    run_id = str(uuid.uuid4())
    invocation_id = str(uuid.uuid4())

    started = None
    stdout_text = ""
    stderr_text = ""
    try:
        started = subprocess.Popen(
            ["docker", "start", "-a", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        monitor.start()
        stdout_text, stderr_text = started.communicate()
    finally:
        end_energy = read_energy()
        end_time = time.time()
        monitor.stop()

    exit_code_raw = _docker_inspect_value(container_id, "{{.State.ExitCode}}")
    fallback_exit_code = started.returncode if started is not None else 1
    try:
        exit_code = int(exit_code_raw) if exit_code_raw is not None else fallback_exit_code
    except ValueError:
        exit_code = fallback_exit_code

    removed = subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, text=True, check=False)
    if removed.returncode != 0 and removed.stderr:
        print(f"  [Warning] Failed to remove container {container_id}: {removed.stderr.strip()}")

    if exit_code != 0:
        print(f"  [Error] Container run failed with code {exit_code}")
        if stdout_text:
            print(f"  STDOUT: {stdout_text}")
        if stderr_text:
            print(f"  STDERR: {stderr_text}")
        return None

    duration_ms = (end_time - start_time) * 1000.0
    energy_delta_uj = end_energy - start_energy
    rapl_wrapped = False
    if energy_delta_uj < 0:
        energy_delta_uj += energy_max_range_uj
        rapl_wrapped = True

    summary = parse_summary(stdout_text)
    runtime_metrics = monitor.metrics(duration_ms)

    record = {
        "platform": "docker_ubuntu22_on_ubuntu20",
        "workload": workload,
        "run_id": run_id,
        "invocation_id": invocation_id,
        "mem_limit_mb": mem_mb,
        "cold_start": False,
        "concurrency": 1,
        "queue_delay_ms": 0.0,
        "duration_ms": duration_ms,
        "energy_joules": energy_delta_uj / 1_000_000.0,
        "display_name": spec["display_name"],
        "suite": spec["suite"],
        "workload_type": spec["workload_type"],
        "partition": spec["partition"],
        "benchmark_name": summary.get("benchmark_name", spec.get("benchmark_name")),
        "resource_profile": spec.get("resource_profile"),
        "resource_profile_index": spec.get("resource_profile_index"),
        "resource_profile_label": spec.get("resource_profile_label"),
        "plot_workload_name": spec.get("plot_workload_name"),
        "input_profile": summary.get("input_profile"),
        "package_name": summary.get("package_name"),
        "build_config": summary.get("build_config"),
        "target_seconds": float(summary.get("target_seconds", target_seconds)),
        "target_iterations": int(summary["target_iterations"]) if summary.get("target_iterations") is not None else None,
        "work_mode": summary.get("work_mode", "fixed_duration"),
        "elapsed_seconds": float(summary["elapsed_seconds"]) if "elapsed_seconds" in summary else None,
        "iterations_completed": int(summary["iterations_completed"]) if "iterations_completed" in summary else None,
        "command_runs": int(summary["command_runs"]) if "command_runs" in summary else None,
        "omp_threads": int(summary.get("omp_threads", summary.get("threads", 1))),
        "cpu_limit": float(cpu_limit),
        "idle_gap_ms": float(idle_gap_ms),
        "run_index": int(run_index),
        "instrumentation_mode": "docker_cgroup_sampling",
        "sample_kind": "aggregate_run",
    }
    record.update(runtime_metrics)
    if record["elapsed_seconds"] is not None:
        record["launch_overhead_ms"] = max(duration_ms - (record["elapsed_seconds"] * 1000.0), 0.0)
    if rapl_wrapped:
        record["rapl_wrapped"] = True
    window_rows = monitor.window_records(
        run_id=run_id,
        invocation_id=invocation_id,
        workload=workload,
        spec=spec,
        mem_mb=mem_mb,
        cpu_limit=cpu_limit,
        target_seconds=target_seconds,
        run_index=run_index,
        idle_gap_ms=idle_gap_ms,
        summary=summary,
        total_energy_uj=energy_delta_uj,
        rapl_max_range_uj=energy_max_range_uj,
    )
    return {
        "aggregate_record": record,
        "window_records": window_rows,
    }

import numpy as np
import pandas as pd


DECISION_BASE_FEATURES = [
    "mem_limit_mb",
    "cold_start",
]

DECISION_RUNTIME_NUMERIC = [
    "queue_delay_ms",
    "cpu_time_ms",
    "cpu_user_time_ms",
    "cpu_system_time_ms",
    "cpu_util_pct",
    "cpu_nr_periods",
    "cpu_nr_throttled",
    "cpu_throttled_ms",
    "cpu_throttled_pct",
    "peak_rss_mb",
    "rss_mb",
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
    "io_read_bytes",
    "io_write_bytes",
    "omp_threads",
    "cpu_limit",
    "target_seconds",
    "target_iterations",
    "elapsed_seconds",
    "iterations_completed",
    "command_runs",
    "idle_gap_ms",
    "launch_overhead_ms",
    "queue_length",
    "queue_signal_available",
    "baseline_energy_1024_joules",
    "projected_slowdown_factor",
    "service_time_ms",
    "throughput_ops_per_s",
    "window_index",
    "window_start_ms",
    "window_end_ms",
]

DECISION_RUNTIME_DEFAULTS = {
    "queue_delay_ms": 0.0,
    "cpu_time_ms": 0.0,
    "cpu_user_time_ms": 0.0,
    "cpu_system_time_ms": 0.0,
    "cpu_util_pct": 0.0,
    "cpu_nr_periods": 0.0,
    "cpu_nr_throttled": 0.0,
    "cpu_throttled_ms": 0.0,
    "cpu_throttled_pct": 0.0,
    "peak_rss_mb": 0.0,
    "rss_mb": 0.0,
    "memory_current_mb": 0.0,
    "memory_peak_mb": 0.0,
    "memory_avg_mb": 0.0,
    "memory_util_pct": 0.0,
    "memory_peak_util_pct": 0.0,
    "memory_avg_util_pct": 0.0,
    "memory_max_events": 0.0,
    "memory_high_events": 0.0,
    "memory_oom_events": 0.0,
    "memory_oom_kill_events": 0.0,
    "cpu_pressure_some_avg10": 0.0,
    "cpu_pressure_full_avg10": 0.0,
    "memory_pressure_some_avg10": 0.0,
    "memory_pressure_full_avg10": 0.0,
    "io_read_bytes": 0.0,
    "io_write_bytes": 0.0,
    "omp_threads": 1.0,
    "cpu_limit": 1.0,
    "target_seconds": 120.0,
    "target_iterations": 0.0,
    "elapsed_seconds": 0.0,
    "iterations_completed": 0.0,
    "command_runs": 0.0,
    "idle_gap_ms": 0.0,
    "launch_overhead_ms": 0.0,
    "queue_length": 0.0,
    "queue_signal_available": 0.0,
    "baseline_energy_1024_joules": 0.0,
    "projected_slowdown_factor": 1.0,
    "service_time_ms": 0.0,
    "throughput_ops_per_s": 0.0,
    "window_index": 0.0,
    "window_start_ms": 0.0,
    "window_end_ms": 0.0,
}

DECISION_CATEGORICAL = [
    "suite",
    "workload_type",
    "input_profile",
    "work_mode",
]


def map_workload_type(workload_series, workload_type_series=None, workload_type_map=None):
    if workload_type_series is not None:
        workload_type = workload_type_series.astype(str).replace({"nan": np.nan})
        if not workload_type.isna().any():
            return workload_type

    if workload_type_map is None:
        raise ValueError("workload_type_map is required when workload_type is missing.")

    workload_type = workload_series.astype(str).map(workload_type_map)
    if workload_type.isna().any():
        missing = sorted(workload_series[workload_type.isna()].astype(str).unique().tolist())
        raise ValueError(f"Missing workload_type mapping for: {missing}")
    return workload_type


def append_categorical_dummies(feature_df, df, col_name, prefix):
    if col_name not in df.columns:
        return feature_df

    valid = df[col_name].replace({"nan": np.nan}).dropna()
    if valid.empty:
        return feature_df

    dummies = pd.get_dummies(
        df[col_name].astype(str),
        prefix=prefix,
        prefix_sep="__",
    )
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    return pd.concat([feature_df, dummies], axis=1)


def _numeric_series(df, col_name, default_value):
    if col_name in df.columns:
        series = pd.to_numeric(df[col_name], errors="coerce")
        if not series.isna().all():
            return series.fillna(default_value)
    return pd.Series(default_value, index=df.index, dtype=float)


def derive_service_time_ms(df):
    if "service_time_ms" in df.columns:
        explicit = pd.to_numeric(df["service_time_ms"], errors="coerce")
        if not explicit.isna().all():
            return explicit.fillna(explicit.median() if not explicit.dropna().empty else 0.0)

    elapsed_seconds = _numeric_series(df, "elapsed_seconds", 0.0)
    iterations_completed = _numeric_series(df, "iterations_completed", 0.0)
    duration_ms = _numeric_series(df, "duration_ms", 0.0)

    safe_iterations = iterations_completed.where(iterations_completed > 0, np.nan)
    service_time_ms = (elapsed_seconds * 1000.0) / safe_iterations
    service_time_ms = service_time_ms.fillna(duration_ms)
    return service_time_ms


def derive_throughput_ops_per_s(df):
    if "throughput_ops_per_s" in df.columns:
        explicit = pd.to_numeric(df["throughput_ops_per_s"], errors="coerce")
        if not explicit.isna().all():
            return explicit.fillna(explicit.median() if not explicit.dropna().empty else 0.0)

    elapsed_seconds = _numeric_series(df, "elapsed_seconds", 0.0)
    iterations_completed = _numeric_series(df, "iterations_completed", 0.0)
    safe_elapsed = elapsed_seconds.where(elapsed_seconds > 0, np.nan)
    throughput = iterations_completed / safe_elapsed
    return throughput.fillna(0.0)


def aggregate_policy_rows(df):
    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame()

    working = df.copy()
    has_windows = False
    if "sample_kind" in working.columns:
        has_windows = working["sample_kind"].astype(str).eq("telemetry_window").any()
    elif "window_index" in working.columns:
        has_windows = pd.to_numeric(working["window_index"], errors="coerce").notna().any()

    if not has_windows:
        return working

    sort_cols = [col for col in ["run_id", "window_index", "window_end_ms"] if col in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols)

    group_cols = [
        col for col in [
            "platform",
            "workload",
            "run_id",
            "invocation_id",
            "mem_limit_mb",
            "cold_start",
            "concurrency",
            "display_name",
            "suite",
            "workload_type",
            "partition",
            "benchmark_name",
            "resource_profile",
            "resource_profile_index",
            "resource_profile_label",
            "plot_workload_name",
            "input_profile",
            "package_name",
            "build_config",
            "target_seconds",
            "target_iterations",
            "work_mode",
            "cpu_limit",
            "run_index",
            "instrumentation_mode",
        ]
        if col in working.columns
    ]

    aggregation = {}
    sum_cols = [
        "energy_joules",
        "duration_ms",
        "cpu_time_ms",
        "cpu_user_time_ms",
        "cpu_system_time_ms",
        "cpu_nr_periods",
        "cpu_nr_throttled",
        "cpu_throttled_ms",
        "io_read_bytes",
        "io_write_bytes",
        "memory_max_events",
        "memory_high_events",
        "memory_oom_events",
        "memory_oom_kill_events",
    ]
    mean_cols = [
        "memory_avg_mb",
        "memory_avg_util_pct",
        "cpu_pressure_some_avg10",
        "cpu_pressure_full_avg10",
        "memory_pressure_some_avg10",
        "memory_pressure_full_avg10",
        "queue_length",
        "queue_signal_available",
    ]
    max_cols = [
        "peak_rss_mb",
        "memory_peak_mb",
        "memory_peak_util_pct",
        "launch_overhead_ms",
        "idle_gap_ms",
        "elapsed_seconds",
        "iterations_completed",
        "command_runs",
        "omp_threads",
        "window_index",
        "window_start_ms",
        "window_end_ms",
    ]
    last_cols = [
        "rss_mb",
        "memory_current_mb",
        "memory_util_pct",
        "sample_kind",
    ]

    for col in sum_cols:
        if col in working.columns:
            aggregation[col] = "sum"
    for col in mean_cols:
        if col in working.columns:
            aggregation[col] = "mean"
    for col in max_cols:
        if col in working.columns:
            aggregation[col] = "max"
    for col in last_cols:
        if col in working.columns:
            aggregation[col] = "last"

    grouped = working.groupby(group_cols, dropna=False).agg(aggregation).reset_index()
    grouped["sample_kind"] = "aggregated_run"
    grouped["monitor_sample_count"] = working.groupby(group_cols, dropna=False).size().values

    if "duration_ms" in grouped.columns and "cpu_time_ms" in grouped.columns and "cpu_limit" in grouped.columns:
        cpu_capacity_ms = np.maximum(
            pd.to_numeric(grouped["duration_ms"], errors="coerce").fillna(0.0)
            * np.maximum(pd.to_numeric(grouped["cpu_limit"], errors="coerce").fillna(1.0), 1e-9),
            1.0,
        )
        grouped["cpu_util_pct"] = (
            pd.to_numeric(grouped["cpu_time_ms"], errors="coerce").fillna(0.0) / cpu_capacity_ms
        ) * 100.0
        grouped["cpu_throttled_pct"] = (
            _numeric_series(grouped, "cpu_throttled_ms", 0.0) / cpu_capacity_ms
        ) * 100.0

    if "mem_limit_mb" in grouped.columns:
        mem_limit = np.maximum(pd.to_numeric(grouped["mem_limit_mb"], errors="coerce").fillna(1.0), 1e-9)
        if "memory_current_mb" in grouped.columns:
            grouped["memory_util_pct"] = (
                pd.to_numeric(grouped["memory_current_mb"], errors="coerce").fillna(0.0) / mem_limit
            ) * 100.0
        if "memory_peak_mb" in grouped.columns:
            grouped["memory_peak_util_pct"] = (
                pd.to_numeric(grouped["memory_peak_mb"], errors="coerce").fillna(0.0) / mem_limit
            ) * 100.0
        if "memory_avg_mb" in grouped.columns:
            grouped["memory_avg_util_pct"] = (
                pd.to_numeric(grouped["memory_avg_mb"], errors="coerce").fillna(0.0) / mem_limit
            ) * 100.0

    grouped["service_time_ms"] = derive_service_time_ms(grouped)
    grouped["throughput_ops_per_s"] = derive_throughput_ops_per_s(grouped)
    return grouped


def aggregate_repeated_measurements(df, agg_mode="median"):
    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame()

    runs = aggregate_policy_rows(df)
    if runs is None or len(runs) == 0:
        return runs

    agg_mode = str(agg_mode).strip().lower()
    if agg_mode not in {"median", "mean"}:
        agg_mode = "median"

    group_cols = [
        col for col in [
            "platform",
            "workload",
            "mem_limit_mb",
            "cpu_limit",
            "cold_start",
            "concurrency",
            "display_name",
            "suite",
            "workload_type",
            "partition",
            "benchmark_name",
            "resource_profile",
            "resource_profile_index",
            "resource_profile_label",
            "plot_workload_name",
            "input_profile",
            "package_name",
            "build_config",
            "target_seconds",
            "target_iterations",
            "work_mode",
            "instrumentation_mode",
        ]
        if col in runs.columns
    ]

    if not group_cols:
        return runs

    numeric_candidates = [
        "energy_joules",
        "duration_ms",
        "queue_delay_ms",
        "cpu_time_ms",
        "cpu_user_time_ms",
        "cpu_system_time_ms",
        "cpu_util_pct",
        "cpu_nr_periods",
        "cpu_nr_throttled",
        "cpu_throttled_ms",
        "cpu_throttled_pct",
        "peak_rss_mb",
        "rss_mb",
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
        "io_read_bytes",
        "io_write_bytes",
        "omp_threads",
        "elapsed_seconds",
        "iterations_completed",
        "command_runs",
        "idle_gap_ms",
        "launch_overhead_ms",
        "queue_length",
        "queue_signal_available",
        "monitor_sample_count",
        "baseline_energy_1024_joules",
        "projected_slowdown_factor",
        "service_time_ms",
        "throughput_ops_per_s",
        "window_index",
        "window_start_ms",
        "window_end_ms",
    ]

    aggregation = {}
    for col in numeric_candidates:
        if col in runs.columns and col not in group_cols:
            aggregation[col] = agg_mode

    for col in runs.columns:
        if col in group_cols or col in aggregation:
            continue
        aggregation[col] = "first"

    grouped = runs.groupby(group_cols, dropna=False).agg(aggregation).reset_index()
    repeat_counts = runs.groupby(group_cols, dropna=False).size().reset_index(name="repeat_count")
    grouped = grouped.merge(repeat_counts, on=group_cols, how="left")
    grouped["sample_kind"] = f"repeat_{agg_mode}"

    if "duration_ms" in grouped.columns and "cpu_time_ms" in grouped.columns and "cpu_limit" in grouped.columns:
        cpu_capacity_ms = np.maximum(
            pd.to_numeric(grouped["duration_ms"], errors="coerce").fillna(0.0)
            * np.maximum(pd.to_numeric(grouped["cpu_limit"], errors="coerce").fillna(1.0), 1e-9),
            1.0,
        )
        grouped["cpu_util_pct"] = (
            pd.to_numeric(grouped["cpu_time_ms"], errors="coerce").fillna(0.0) / cpu_capacity_ms
        ) * 100.0
        grouped["cpu_throttled_pct"] = (
            _numeric_series(grouped, "cpu_throttled_ms", 0.0) / cpu_capacity_ms
        ) * 100.0

    grouped = _recompute_memory_utilization_columns(grouped)
    grouped["service_time_ms"] = derive_service_time_ms(grouped)
    grouped["throughput_ops_per_s"] = derive_throughput_ops_per_s(grouped)
    return grouped


def _recompute_memory_utilization_columns(df):
    if "mem_limit_mb" not in df.columns:
        return df

    mem_limit = np.maximum(pd.to_numeric(df["mem_limit_mb"], errors="coerce").fillna(1.0), 1e-9)
    if "memory_current_mb" in df.columns:
        df["memory_util_pct"] = (
            pd.to_numeric(df["memory_current_mb"], errors="coerce").fillna(0.0) / mem_limit
        ) * 100.0
    if "memory_peak_mb" in df.columns:
        df["memory_peak_util_pct"] = (
            pd.to_numeric(df["memory_peak_mb"], errors="coerce").fillna(0.0) / mem_limit
        ) * 100.0
    if "memory_avg_mb" in df.columns:
        df["memory_avg_util_pct"] = (
            pd.to_numeric(df["memory_avg_mb"], errors="coerce").fillna(0.0) / mem_limit
        ) * 100.0
    return df


def project_baseline_rows_to_memory(
    baseline_df,
    target_mem_mb,
    baseline_mem_mb=1024,
    target_cpu_limit=None,
    baseline_cpu_limit=1.0,
):
    if baseline_df is None or len(baseline_df) == 0:
        return pd.DataFrame()

    projected = baseline_df.copy()
    source_mem = float(
        pd.to_numeric(projected.get("mem_limit_mb", baseline_mem_mb), errors="coerce").fillna(baseline_mem_mb).iloc[0]
    )
    source_cpu_limit = float(
        pd.to_numeric(projected.get("cpu_limit", baseline_cpu_limit), errors="coerce").fillna(baseline_cpu_limit).iloc[0]
    )
    target_mem_mb = float(target_mem_mb)
    target_cpu_limit = float(source_cpu_limit if target_cpu_limit is None else target_cpu_limit)

    projected["source_mem_limit_mb"] = source_mem
    projected["source_cpu_limit"] = source_cpu_limit
    projected["mem_limit_mb"] = target_mem_mb
    projected["cpu_limit"] = target_cpu_limit
    projected["baseline_energy_1024_joules"] = pd.to_numeric(
        projected.get("baseline_energy_1024_joules", projected.get("energy_joules", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    projected["sample_kind"] = "projected_from_baseline"
    projected = _recompute_memory_utilization_columns(projected)

    base_service_time_ms = derive_service_time_ms(baseline_df)
    base_throughput = derive_throughput_ops_per_s(baseline_df)
    cpu_util_fraction = (
        _numeric_series(baseline_df, "cpu_util_pct", 0.0).clip(lower=0.0, upper=100.0) / 100.0
    ).clip(lower=0.0, upper=1.0)
    if target_cpu_limit < source_cpu_limit:
        cpu_scale_ratio = source_cpu_limit / max(target_cpu_limit, 1e-9)
        slowdown_factor = 1.0 + (cpu_util_fraction * (cpu_scale_ratio - 1.0))
    else:
        slowdown_factor = pd.Series(1.0, index=projected.index, dtype=float)

    base_service_time_ms = pd.to_numeric(base_service_time_ms, errors="coerce").fillna(0.0)
    base_throughput = pd.to_numeric(base_throughput, errors="coerce").fillna(0.0)
    projected["projected_slowdown_factor"] = slowdown_factor.astype(float)
    projected["service_time_ms"] = base_service_time_ms * projected["projected_slowdown_factor"]
    projected["throughput_ops_per_s"] = base_throughput / projected["projected_slowdown_factor"].replace(0.0, 1.0)

    if "iterations_completed" in projected.columns:
        projected["iterations_completed"] = (
            pd.to_numeric(projected["iterations_completed"], errors="coerce").fillna(0.0)
            / projected["projected_slowdown_factor"].replace(0.0, 1.0)
        )
    if "command_runs" in projected.columns:
        projected["command_runs"] = (
            pd.to_numeric(projected["command_runs"], errors="coerce").fillna(0.0)
            / projected["projected_slowdown_factor"].replace(0.0, 1.0)
        )

    duration_ms = _numeric_series(projected, "duration_ms", 0.0)
    cpu_time_ms = _numeric_series(projected, "cpu_time_ms", 0.0)
    cpu_throttled_ms = _numeric_series(projected, "cpu_throttled_ms", 0.0)
    cpu_capacity_ms = np.maximum(duration_ms * max(target_cpu_limit, 1e-9), 1.0)
    projected["cpu_util_pct"] = (cpu_time_ms / cpu_capacity_ms) * 100.0
    projected["cpu_throttled_pct"] = (cpu_throttled_ms / cpu_capacity_ms) * 100.0
    return projected


def build_projected_decision_dataframe(df, baseline_mem_mb=1024, baseline_cpu_limit=1.0, agg_mode="median"):
    aggregated = aggregate_repeated_measurements(df, agg_mode=agg_mode)
    if aggregated is None or len(aggregated) == 0:
        return pd.DataFrame()

    if "mem_limit_mb" not in aggregated.columns:
        raise ValueError("mem_limit_mb is required to build the projected decision dataframe.")

    baseline_pool = aggregated[
        (pd.to_numeric(aggregated["mem_limit_mb"], errors="coerce") == float(baseline_mem_mb))
        & (
            pd.to_numeric(aggregated.get("cpu_limit", baseline_cpu_limit), errors="coerce").fillna(baseline_cpu_limit)
            == float(baseline_cpu_limit)
        )
    ].copy()
    if baseline_pool.empty:
        workload_summaries = []
        for workload_name, workload_df in aggregated.groupby(aggregated["workload"].astype(str)):
            configs = sorted(
                {
                    (
                        int(mem_mb),
                        float(cpu_limit),
                    )
                    for mem_mb, cpu_limit in zip(
                        pd.to_numeric(workload_df["mem_limit_mb"], errors="coerce"),
                        pd.to_numeric(workload_df.get("cpu_limit", baseline_cpu_limit), errors="coerce").fillna(baseline_cpu_limit),
                    )
                }
            )
            workload_summaries.append(f"{workload_name}: {configs}")
        raise ValueError(
            "Missing baseline rows for projected decision dataset at "
            f"{baseline_mem_mb}MB and CPU {baseline_cpu_limit}. "
            "Available configs by workload: "
            + "; ".join(workload_summaries)
        )

    rows = []
    match_cols = [
        col for col in [
            "workload",
            "run_index",
            "cold_start",
            "concurrency",
            "input_profile",
            "target_seconds",
            "target_iterations",
            "work_mode",
        ]
        if col in aggregated.columns
    ]

    for _, target_row in aggregated.iterrows():
        target_df = target_row.to_frame().T
        target_workload = str(target_row["workload"])
        target_mem = float(pd.to_numeric(target_row["mem_limit_mb"], errors="coerce"))
        target_cpu = float(pd.to_numeric(target_row.get("cpu_limit", baseline_cpu_limit), errors="coerce"))

        candidates = baseline_pool[baseline_pool["workload"].astype(str) == target_workload].copy()
        if candidates.empty:
            workload_df = aggregated[aggregated["workload"].astype(str) == target_workload].copy()
            configs = sorted(
                {
                    (
                        int(mem_mb),
                        float(cpu_limit),
                    )
                    for mem_mb, cpu_limit in zip(
                        pd.to_numeric(workload_df["mem_limit_mb"], errors="coerce"),
                        pd.to_numeric(workload_df.get("cpu_limit", baseline_cpu_limit), errors="coerce").fillna(baseline_cpu_limit),
                    )
                }
            )
            raise ValueError(
                "Missing baseline rows for workload "
                f"{target_workload} at {baseline_mem_mb}MB / CPU {baseline_cpu_limit}. "
                f"Available configs for this workload: {configs}"
            )

        for col in match_cols:
            if col == "workload":
                continue
            if col not in candidates.columns:
                continue
            target_value = target_row[col]
            if pd.isna(target_value):
                candidates = candidates[candidates[col].isna()]
            else:
                candidates = candidates[candidates[col] == target_value]
            if candidates.empty:
                break

        if candidates.empty:
            candidates = baseline_pool[baseline_pool["workload"].astype(str) == target_workload].copy()

        baseline_row = candidates.iloc[[0]].copy()
        baseline_energy = float(pd.to_numeric(baseline_row["energy_joules"], errors="coerce").fillna(0.0).iloc[0])
        baseline_service_time_ms = float(derive_service_time_ms(baseline_row).fillna(0.0).iloc[0])
        projected = project_baseline_rows_to_memory(
            baseline_row,
            target_mem_mb=target_mem,
            baseline_mem_mb=baseline_mem_mb,
            target_cpu_limit=target_cpu,
            baseline_cpu_limit=baseline_cpu_limit,
        )
        projected["baseline_energy_1024_joules"] = baseline_energy
        projected["baseline_service_time_ms"] = baseline_service_time_ms
        projected["observed_energy_joules"] = pd.to_numeric(target_df["energy_joules"], errors="coerce").fillna(0.0).iloc[0]
        projected["energy_delta_vs_1024_joules"] = projected["observed_energy_joules"] - baseline_energy
        rows.append(projected)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_resource_decision_feature_df(df, workload_type_map, feature_names=None):
    df = df.copy()
    bool_map = {
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        True: 1,
        False: 0,
        1: 1,
        0: 0,
    }

    df["cold_start"] = df.get("cold_start", 0)
    df["cold_start"] = df["cold_start"].map(bool_map).fillna(df["cold_start"]).astype(int)
    df["mem_limit_mb"] = pd.to_numeric(df["mem_limit_mb"], errors="coerce")
    source_workload_type = df["workload_type"] if "workload_type" in df.columns else None
    df["workload_type"] = map_workload_type(df["workload"], source_workload_type, workload_type_map)

    feature_df = df[DECISION_BASE_FEATURES].copy()
    feature_df["log_mem"] = np.log1p(df["mem_limit_mb"].astype(float))

    for col_name in DECISION_RUNTIME_NUMERIC:
        include_col = feature_names is None or col_name in feature_names
        if not include_col:
            continue
        if col_name == "service_time_ms":
            feature_df[col_name] = derive_service_time_ms(df)
        elif col_name == "throughput_ops_per_s":
            feature_df[col_name] = derive_throughput_ops_per_s(df)
        else:
            feature_df[col_name] = _numeric_series(df, col_name, DECISION_RUNTIME_DEFAULTS[col_name])

    peak_rss = _numeric_series(df, "peak_rss_mb", 0.0)
    current_rss = _numeric_series(df, "rss_mb", 0.0)
    cpu_time_ms = _numeric_series(df, "cpu_time_ms", 0.0)
    cpu_util_pct = _numeric_series(df, "cpu_util_pct", 0.0)
    peak_mem_util_pct = _numeric_series(df, "memory_peak_util_pct", 0.0)
    queue_length = _numeric_series(df, "queue_length", 0.0)
    service_time_ms = derive_service_time_ms(df)
    target_seconds = _numeric_series(df, "target_seconds", DECISION_RUNTIME_DEFAULTS["target_seconds"])
    target_iterations = _numeric_series(df, "target_iterations", 0.0)
    window_end_ms = _numeric_series(df, "window_end_ms", 0.0)
    elapsed_seconds = _numeric_series(df, "elapsed_seconds", 0.0)
    mem_limit = df["mem_limit_mb"].astype(float).clip(lower=1e-9)

    progress_denominator_ms = (elapsed_seconds * 1000.0).where(elapsed_seconds > 0, target_seconds * 1000.0)
    progress_denominator_ms = progress_denominator_ms.where(progress_denominator_ms > 0, np.nan)
    progress_ratio = (window_end_ms / progress_denominator_ms).clip(lower=0.0)
    progress_ratio = progress_ratio.fillna(0.0).clip(upper=1.0)

    derived_columns = {
        "rss_ratio": peak_rss / mem_limit,
        "current_rss_ratio": current_rss / mem_limit,
        "cpu_per_mem": cpu_time_ms / mem_limit,
        "cpu_util_fraction": cpu_util_pct / 100.0,
        "memory_headroom_pct": 100.0 - peak_mem_util_pct,
        "under_18_cpu_util": (cpu_util_pct < 18.0).astype(float),
        "low_queue_signal": (queue_length <= 0.0).astype(float),
        "progress_ratio": progress_ratio,
        "is_fixed_work": (target_iterations > 0).astype(float),
        "latency_pressure_flag": (
            (service_time_ms > 0)
            & (cpu_util_pct >= 65.0)
            & (peak_mem_util_pct >= 70.0)
        ).astype(float),
    }
    for col_name, series in derived_columns.items():
        if feature_names is None or col_name in feature_names:
            feature_df[col_name] = series

    for col_name in DECISION_CATEGORICAL:
        prefix = "workload_type" if col_name == "workload_type" else col_name
        feature_df = append_categorical_dummies(feature_df, df, col_name, prefix)

    if feature_names is not None:
        feature_df = feature_df.reindex(columns=feature_names, fill_value=0.0)

    return feature_df


def _longest_true_run_fraction(mask_series):
    if mask_series is None or len(mask_series) == 0:
        return 0.0

    longest = 0
    current = 0
    for flag in mask_series.astype(bool).tolist():
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest) / float(len(mask_series))


def compute_resource_safety_penalty(
    candidate_df,
    slo_reference_ms=None,
    slo_multiplier=1.10,
    current_mem_mb=1024,
):
    if candidate_df.empty:
        return 0.0

    penalty = 0.0

    def median_col(col_name, default_value=0.0):
        if col_name not in candidate_df.columns:
            return default_value
        series = pd.to_numeric(candidate_df[col_name], errors="coerce").dropna()
        if series.empty:
            return default_value
        return float(series.median())

    ordered_df = candidate_df.copy()
    if "window_end_ms" in ordered_df.columns:
        ordered_df["window_end_ms"] = pd.to_numeric(ordered_df["window_end_ms"], errors="coerce")
        ordered_df = ordered_df.sort_values(["window_end_ms", "window_index"] if "window_index" in ordered_df.columns else ["window_end_ms"])
    elif "window_index" in ordered_df.columns:
        ordered_df["window_index"] = pd.to_numeric(ordered_df["window_index"], errors="coerce")
        ordered_df = ordered_df.sort_values("window_index")

    if median_col("memory_oom_kill_events") > 0 or median_col("memory_oom_events") > 0:
        penalty += 1_000_000.0

    peak_mem_util = median_col("memory_peak_util_pct")
    if peak_mem_util >= 98.0:
        penalty += 10_000.0
    elif peak_mem_util >= 92.0:
        penalty += 2_500.0
    elif peak_mem_util >= 85.0:
        penalty += 250.0
    elif peak_mem_util >= 75.0:
        penalty += 25.0

    cpu_throttled_pct = median_col("cpu_throttled_pct")
    cpu_nr_periods = median_col("cpu_nr_periods")
    cpu_nr_throttled = median_col("cpu_nr_throttled")
    throttled_period_ratio_pct = 0.0
    if cpu_nr_periods > 0:
        throttled_period_ratio_pct = min((cpu_nr_throttled / max(cpu_nr_periods, 1e-9)) * 100.0, 100.0)
    throttled_signal_pct = cpu_throttled_pct
    if not np.isfinite(throttled_signal_pct) or throttled_signal_pct > 100.0:
        throttled_signal_pct = throttled_period_ratio_pct
    throttled_signal_pct = float(np.clip(throttled_signal_pct, 0.0, 100.0))
    if throttled_signal_pct >= 80.0:
        penalty += 25.0
    elif throttled_signal_pct >= 40.0:
        penalty += 10.0
    elif throttled_signal_pct >= 10.0:
        penalty += 2.5

    if median_col("queue_signal_available") > 0:
        penalty += 100.0 * max(median_col("queue_length"), 0.0)

    cpu_util_pct = median_col("cpu_util_pct")

    service_time_ms = median_col("service_time_ms")
    allowed_ms = None
    if slo_reference_ms is not None and slo_reference_ms > 0:
        allowed_ms = slo_reference_ms * float(slo_multiplier)
        if service_time_ms > allowed_ms:
            violation_ms = service_time_ms - allowed_ms
            penalty += 10_000.0 + (50.0 * (violation_ms / max(allowed_ms, 1e-9)))

    if "sample_kind" in ordered_df.columns and ordered_df["sample_kind"].astype(str).eq("telemetry_window").any():
        service_series = pd.to_numeric(ordered_df.get("service_time_ms"), errors="coerce").fillna(service_time_ms)
        cpu_series = pd.to_numeric(ordered_df.get("cpu_util_pct"), errors="coerce").fillna(cpu_util_pct)
        peak_mem_series = pd.to_numeric(ordered_df.get("memory_peak_util_pct"), errors="coerce").fillna(peak_mem_util)
        throttle_series = pd.to_numeric(ordered_df.get("cpu_throttled_pct"), errors="coerce").fillna(cpu_throttled_pct)
        throttle_series = throttle_series.clip(lower=0.0, upper=100.0)

        bad_mask = (peak_mem_series >= 85.0) | (throttle_series >= 5.0)
        if allowed_ms is not None and allowed_ms > 0:
            bad_mask = bad_mask | (service_series > allowed_ms)
            near_slo_mask = (service_series > (allowed_ms * 0.97)) & (cpu_series >= 98.0) & (peak_mem_series >= 70.0)
            bad_mask = bad_mask | near_slo_mask

        bad_fraction = float(bad_mask.mean()) if len(bad_mask) else 0.0
        longest_bad_fraction = _longest_true_run_fraction(bad_mask)
        penalty += 120.0 * bad_fraction
        penalty += 80.0 * longest_bad_fraction

    return penalty

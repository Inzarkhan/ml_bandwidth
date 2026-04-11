#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${SEBS_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${SEBS_PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python3" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python3"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -n "${CONDA_PYTHON_EXE:-}" && -x "${CONDA_PYTHON_EXE}" ]]; then
  PYTHON_BIN="${CONDA_PYTHON_EXE}"
else
  PYTHON_BIN="$(command -v python3)"
fi

KNOWN_DIR="${ROOT_DIR}/known_slo14_recollect"
KNOWN_RAW="${ROOT_DIR}/raw_known_full_plusfb.jsonl"
KNOWN_WINDOWS="${ROOT_DIR}/raw_known_full_plusfb_windows.jsonl"
KNOWN_PREPARED="${ROOT_DIR}/prepared_known_full_plusfb.csv"
KNOWN_DATASET_DIR="${ROOT_DIR}/dataset_reg_known_full_plusfb"
MODEL_DIR="${ROOT_DIR}/models_known_full_plusfb_slo14"

UNSEEN_RAW="${ROOT_DIR}/raw_functionbench_download_upload_unseen.jsonl"
UNSEEN_WINDOWS="${ROOT_DIR}/raw_functionbench_download_upload_unseen_windows.jsonl"
UNSEEN_PREPARED="${ROOT_DIR}/prepared_functionbench_download_upload_unseen_after_plusfb.csv"

KNOWN_HOST_BUDGET_SECONDS="${KNOWN_HOST_BUDGET_SECONDS:-300}"
KNOWN_TARGET_SECONDS="${KNOWN_TARGET_SECONDS:-30}"
KNOWN_RUNS="${KNOWN_RUNS:-2}"
KNOWN_MEMORY_SEED="${KNOWN_MEMORY_SEED:-50}"

UNSEEN_HOST_BUDGET_SECONDS="${UNSEEN_HOST_BUDGET_SECONDS:-300}"
UNSEEN_TARGET_SECONDS="${UNSEEN_TARGET_SECONDS:-30}"
UNSEEN_RUNS="${UNSEEN_RUNS:-1}"
UNSEEN_MEMORY_SEED="${UNSEEN_MEMORY_SEED:-61}"
UNSEEN_WORKLOADS="${UNSEEN_WORKLOADS:-functionbench_download_upload_unseen}"

MEMORY_SIZES="${MEMORY_SIZES:-512,1024}"
CPU_LIMITS="${CPU_LIMITS:-0.75,1.0}"
SLO_MULTIPLIER="${SLO_MULTIPLIER:-1.4}"
SAMPLE_INTERVAL_MS="${SAMPLE_INTERVAL_MS:-250}"

maybe_fix_ownership() {
  local owner_user owner_group
  owner_user="${SUDO_USER:-}"
  if [[ -z "${owner_user}" ]]; then
    return 0
  fi
  owner_group="$(id -gn "${owner_user}")"
  chown "${owner_user}:${owner_group}" "$@" 2>/dev/null || true
}

run_as_owner() {
  if [[ "${EUID}" -eq 0 && -n "${SUDO_USER:-}" ]]; then
    sudo -u "${SUDO_USER}" "$@"
  else
    "$@"
  fi
}

require_root_for_collection() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Collection stages need sudo/root because the collectors use Docker + host energy access." >&2
    echo "Run: sudo -E bash ./run_slo14_experiment.sh <stage>" >&2
    exit 1
  fi
}

collect_known() {
  require_root_for_collection
  cd "${ROOT_DIR}"
  mkdir -p "${KNOWN_DIR}"
  rm -f "${KNOWN_DIR}"/*.jsonl "${KNOWN_DIR}"/*_windows.jsonl 2>/dev/null || true

  local known_workloads=(
    sebs_compression_known
    sebs_video_processing_known
    sebs_graph_bfs_known
    sebs_graph_mst_known
    sebs_dynamic_html_known
    sebs_crud_api_known
    sebs_uploader_known
    sebs_dna_visualisation_known
    functionbench_download_upload_known
  )

  for wl in "${known_workloads[@]}"; do
    env \
      SEBS_HOST_BUDGET_SECONDS="${KNOWN_HOST_BUDGET_SECONDS}" \
      SEBS_TARGET_SECONDS="${KNOWN_TARGET_SECONDS}" \
      SEBS_RUNS="${KNOWN_RUNS}" \
      SEBS_SAMPLE_INTERVAL_MS="${SAMPLE_INTERVAL_MS}" \
      SEBS_RANDOMIZE_MEMORY_ORDER=1 \
      SEBS_MEMORY_ORDER_SEED="${KNOWN_MEMORY_SEED}" \
      SEBS_MEMORY_SIZES="${MEMORY_SIZES}" \
      SEBS_CPU_LIMITS="${CPU_LIMITS}" \
      SEBS_KNOWN_WORKLOADS="${wl}" \
      SEBS_KNOWN_OUTPUT="${KNOWN_DIR}/${wl}.jsonl" \
      SEBS_KNOWN_WINDOWS_OUTPUT="${KNOWN_DIR}/${wl}_windows.jsonl" \
      "${PYTHON_BIN}" -u "${ROOT_DIR}/run_known_containers.py"
  done

  cat "${KNOWN_DIR}"/*.jsonl > "${KNOWN_RAW}"
  cat "${KNOWN_DIR}"/*_windows.jsonl > "${KNOWN_WINDOWS}"
  maybe_fix_ownership "${KNOWN_RAW}" "${KNOWN_WINDOWS}" "${KNOWN_DIR}"/*
}

train_model() {
  cd "${ROOT_DIR}"
  run_as_owner env \
    PREPARE_INPUT_FILES="${KNOWN_WINDOWS}" \
    PREPARE_OUTPUT_CSV="${KNOWN_PREPARED}" \
    "${PYTHON_BIN}" -u "${ROOT_DIR}/1. prepare_serverless_csv.py"

  run_as_owner env \
    DATASET_INPUT_CSV="${KNOWN_PREPARED}" \
    DATASET_OUT_DIR="${KNOWN_DATASET_DIR}" \
    "${PYTHON_BIN}" -u "${ROOT_DIR}/2. build_dataset_regression.py"

  run_as_owner env \
    SEBS_DECISION_SLO_MULTIPLIER="${SLO_MULTIPLIER}" \
    PREPARED_CSV="${KNOWN_PREPARED}" \
    DATASET_DIR="${KNOWN_DATASET_DIR}" \
    MODEL_DIR="${MODEL_DIR}" \
    "${PYTHON_BIN}" -u "${ROOT_DIR}/3. train_hgbdt_regressors.py"
}

collect_unseen() {
  require_root_for_collection
  cd "${ROOT_DIR}"
  env \
    SEBS_HOST_BUDGET_SECONDS="${UNSEEN_HOST_BUDGET_SECONDS}" \
    SEBS_TARGET_SECONDS="${UNSEEN_TARGET_SECONDS}" \
    SEBS_RUNS="${UNSEEN_RUNS}" \
    SEBS_SAMPLE_INTERVAL_MS="${SAMPLE_INTERVAL_MS}" \
    SEBS_RANDOMIZE_MEMORY_ORDER=1 \
    SEBS_MEMORY_ORDER_SEED="${UNSEEN_MEMORY_SEED}" \
    SEBS_MEMORY_SIZES="${MEMORY_SIZES}" \
    SEBS_CPU_LIMITS="${CPU_LIMITS}" \
    SEBS_UNSEEN_WORKLOADS="${UNSEEN_WORKLOADS}" \
    SEBS_UNSEEN_OUTPUT="${UNSEEN_RAW}" \
    SEBS_UNSEEN_WINDOWS_OUTPUT="${UNSEEN_WINDOWS}" \
    "${PYTHON_BIN}" -u "${ROOT_DIR}/7. run_unseen_containers.py"
  maybe_fix_ownership "${UNSEEN_RAW}" "${UNSEEN_WINDOWS}"

  run_as_owner env \
    PREPARE_INPUT_FILES="${UNSEEN_WINDOWS}" \
    PREPARE_OUTPUT_CSV="${UNSEEN_PREPARED}" \
    "${PYTHON_BIN}" -u "${ROOT_DIR}/1. prepare_serverless_csv.py"
  maybe_fix_ownership "${UNSEEN_PREPARED}"
}

eval_known() {
  cd "${ROOT_DIR}"
  run_as_owner bash -lc "cd '${ROOT_DIR}' && printf 'n\\n' | env \
    SEBS_SLO_MULTIPLIER='${SLO_MULTIPLIER}' \
    SEBS_MEMORY_SIZES='${MEMORY_SIZES}' \
    SEBS_CPU_LIMITS='${CPU_LIMITS}' \
    EVAL_REPEAT_AGG_MODE=median \
    EVAL_KNOWN_POLICY_FILE='${KNOWN_PREPARED}' \
    EVAL_KNOWN_ACTUAL_FILE='${KNOWN_RAW}' \
    EVAL_MODEL_FILE='${MODEL_DIR}/energy_hgbdt_decision.joblib' \
    EVAL_MODEL_META_FILE='${MODEL_DIR}/energy_hgbdt_decision_meta.json' \
    EVAL_CLASSIFIER_FILE='${MODEL_DIR}/energy_hgbdt_decision_classifier.joblib' \
    '${PYTHON_BIN}' -u '${ROOT_DIR}/6. real_energy_savings.py'"
}

eval_unseen() {
  cd "${ROOT_DIR}"
  run_as_owner bash -lc "cd '${ROOT_DIR}' && printf 'n\\n' | env \
    SEBS_SLO_MULTIPLIER='${SLO_MULTIPLIER}' \
    SEBS_MEMORY_SIZES='${MEMORY_SIZES}' \
    SEBS_CPU_LIMITS='${CPU_LIMITS}' \
    EVAL_REPEAT_AGG_MODE=median \
    EVAL_UNSEEN_POLICY_INPUT='${UNSEEN_PREPARED}' \
    EVAL_UNSEEN_ACTUAL_INPUT='${UNSEEN_RAW}' \
    EVAL_MODEL_FILE='${MODEL_DIR}/energy_hgbdt_decision.joblib' \
    EVAL_MODEL_META_FILE='${MODEL_DIR}/energy_hgbdt_decision_meta.json' \
    EVAL_CLASSIFIER_FILE='${MODEL_DIR}/energy_hgbdt_decision_classifier.joblib' \
    '${PYTHON_BIN}' -u '${ROOT_DIR}/8. validate_unseen.py'"
}

usage() {
  cat <<'EOF'
Usage:
  bash ./run_slo14_experiment.sh <stage>

Stages:
  collect_known   Run the final broader known/train recollection in per-workload batches (needs sudo)
  train           Prepare known CSV, build dataset, and train the slo14 model
  collect_unseen  Run the default final unseen validation workload(s) and prepare the unseen CSV (needs sudo)
  eval_known      Evaluate the trained model on the known set
  eval_unseen     Evaluate the trained model on the unseen validation set
  all             Run collect_known -> train -> collect_unseen -> eval_known -> eval_unseen

Important:
  - Collection stages need sudo/root.
  - Training and evaluation can run without sudo.
  - Current defaults in this script:
      known budget = 300s per workload batch
      unseen budget = 300s
      target = 30s
      memory = 512,1024
      cpu = 0.75,1.0
      SLO = 1.4
      unseen workloads = functionbench_download_upload_unseen
      python = auto-detected Conda/user interpreter when available
  - The known collector already splits the full known set into per-workload 300s batches and then combines them.
  - Override UNSEEN_WORKLOADS if you want to test a different kept unseen workload or set.

Examples:
  sudo -E bash ./run_slo14_experiment.sh collect_known
  bash ./run_slo14_experiment.sh train
  sudo -E bash ./run_slo14_experiment.sh collect_unseen
  bash ./run_slo14_experiment.sh eval_known
  bash ./run_slo14_experiment.sh eval_unseen

Quick pilot overrides:
  KNOWN_HOST_BUDGET_SECONDS=300 KNOWN_RUNS=1 sudo -E bash ./run_slo14_experiment.sh collect_known
EOF
}

main() {
  local stage="${1:-help}"
  case "${stage}" in
    collect_known) collect_known ;;
    train) train_model ;;
    collect_unseen) collect_unseen ;;
    eval_known) eval_known ;;
    eval_unseen) eval_unseen ;;
    all)
      collect_known
      train_model
      collect_unseen
      eval_known
      eval_unseen
      ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown stage: ${stage}" >&2
      usage
      exit 1
      ;;
  esac
}

main "${1:-help}"

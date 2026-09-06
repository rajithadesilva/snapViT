#!/usr/bin/env bash
# Unattended, resumable entrypoint for the DeepSets fusion ablation campaign.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/fusion_ablations}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "${RUN_DIR}"

campaign_args=(
  -m src.training.fusion_ablation
  campaign
  --run-dir "${RUN_DIR}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  campaign_args+=(--dry-run)
fi
if [[ "${SMOKE_TEST}" == "1" ]]; then
  campaign_args+=(--smoke-test)
fi

echo "Fusion ablation campaign"
echo "  run directory: ${RUN_DIR}"
echo "  device:        ${DEVICE}"
echo "  workers:       ${NUM_WORKERS}"

"${PYTHON_BIN}" "${campaign_args[@]}"
campaign_status=$?

# Refresh the human-readable report even when a campaign subprocess failed.
"${PYTHON_BIN}" -m src.evaluation.fusion_ablation_report --run-dir "${RUN_DIR}"
report_status=$?

if [[ ${campaign_status} -ne 0 ]]; then
  echo "Campaign exited with status ${campaign_status}; partial results are retained in ${RUN_DIR}." >&2
  exit "${campaign_status}"
fi
if [[ ${report_status} -ne 0 ]]; then
  echo "Campaign completed, but report generation exited with status ${report_status}." >&2
  exit "${report_status}"
fi

echo "Campaign complete. Read ${RUN_DIR}/summary.md"

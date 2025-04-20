#!/bin/bash
###############################################################################
#  SLURM  JOB‑ARRAY  —  RPP network & homophily feature extraction
#
#  One task  =  one CSV chunk (≈ 1 ~ n DOIs) created by split_targets.py
#  Each task:
#     • creates a disposable venv in $TMPDIR  (fast local storage)
#     • installs Python deps             (cached wheels → quick)
#     • calls  src/run_worker.py  with   --chunk_id = $SLURM_ARRAY_TASK_ID
#
#  Output:
#     data/features/results_chunk_<ID>.csv
#     data/networks_raw/<doi>.json
#     logs/<JOBID>_<ARRAYID>.out   (stdout / stderr)
###############################################################################

#SBATCH --job-name=rpp_net
#SBATCH --array=0-5                     # 100 chunks → adjust if n_chunks differs
#SBATCH --cpus-per-task=4               # aiohttp concurrency lives on threads
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out          # %A = jobID, %a = array index
# SBATCH --gres=       # comment in if GPUs are ever needed

set -euo pipefail
echo "🟢  SLURM task $SLURM_ARRAY_TASK_ID on $(hostname) @ $(date)"

VENV_HOME=$HOME/venvs/rpp_net_env          # same path used above
source "$VENV_HOME/bin/activate"

export OPENALEX_API_KEY="averylou@stanford.edu"
export OPENALEX_USER_AGENT="mailto:averylou@stanford.edu"

CHUNK_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
python src/run_worker.py \
        --chunk_id "$CHUNK_ID" \
        --max_depth 2 \
        --max_nodes 1000 \
        --n_concurrent 1

echo "✅  SLURM task $SLURM_ARRAY_TASK_ID finished @ $(date)"
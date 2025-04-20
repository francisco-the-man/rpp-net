#!/bin/bash
###############################################################################
#  SLURM  JOB‑ARRAY  —  RPP network & homophily feature extraction
#
#  One task  =  one CSV chunk (≈ 1 ~ n DOIs) created by split_targets.py
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
#SBATCH --array=0-10                     # 100 chunks → adjust if n_chunks differs
#SBATCH --cpus-per-task=4               # aiohttp concurrency lives on threads
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out          # %A = jobID, %a = array index
# SBATCH --gres=       # comment in if GPUs are ever needed

set -euo pipefail
echo "🟢  SLURM task $SLURM_ARRAY_TASK_ID starting on $(hostname) at $(date)"

# Create virtual environment
VENV_DIR="$TMPDIR/rpp_net_env"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install packages one by one to avoid dependency issues
pip install --no-cache-dir multidict
pip install --no-cache-dir aiohttp
pip install --no-cache-dir nest_asyncio
pip install --no-cache-dir orjson
pip install --no-cache-dir numpy
pip install --no-cache-dir scipy
pip install --no-cache-dir pandas
pip install --no-cache-dir networkx
pip install --no-cache-dir requests
pip install --no-cache-dir pyalex
pip install --no-cache-dir python-louvain

# Set API key for OpenAlex
export OPENALEX_API_KEY="averylou@stanford.edu"
export OPENALEX_USER_AGENT="mailto:averylou@stanford.edu"

# Run the worker script
CHUNK_ID=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
python src/run_worker.py \
    --chunk_id "$CHUNK_ID" \
    --max_depth 2 \
    --max_nodes 1000 \
    --n_concurrent 5

echo "✅  SLURM task $SLURM_ARRAY_TASK_ID finished at $(date)"
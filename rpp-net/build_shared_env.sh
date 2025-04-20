#!/bin/bash
set -euo pipefail

VENV_HOME=$HOME/venvs/rpp_net_env          # or /scratch/$USER/rpp_net_env
python3 -m venv "$VENV_HOME"
source "$VENV_HOME/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
pip freeze > "$VENV_HOME/requirements.lock"

deactivate
echo "✅  Shared venv created at $VENV_HOME"

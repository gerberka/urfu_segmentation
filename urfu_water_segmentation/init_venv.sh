#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Cannot find Python interpreter '${PYTHON_BIN}'. Set PYTHON_BIN to a valid binary." >&2
  exit 1
fi

poetry install -vv

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

poetry install --sync

OS_NAME="$(uname -s)"

if [[ "${OS_NAME}" == "Darwin" ]]; then
  python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
  python -m pip install mmcv==2.0.0
else
  python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116 --default-timeout=1000
  python -m pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
  export LD_LIBRARY_PATH="/opt/cuda-11.6/lib64/:${LD_LIBRARY_PATH:-}"
fi

export LD_LIBRARY_PATH="/opt/cuda-11.6/lib64/:$LD_LIBRARY_PATH"

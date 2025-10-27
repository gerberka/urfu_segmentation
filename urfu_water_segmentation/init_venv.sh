#!/bin/bash
python3.9 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install poetry==1.8.3

poetry install

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 --default-timeout=1000

pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

pip install mmdet

pip install future tensorboard
pip install 'numpy<2.0.0'

export LD_LIBRARY_PATH="/opt/cuda-11.6/lib64/:$LD_LIBRARY_PATH"
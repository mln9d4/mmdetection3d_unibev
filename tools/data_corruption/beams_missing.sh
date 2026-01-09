#!/usr/bin/env bash

PORT=${PORT:-29502}

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=5
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export OMP_NUM_THREADS=6
export NCCL_P2P_LEVEL=NVL

python ./tools/gen_beam_missing.py \
            --n_cpus 4\
            --root_folder ./data/nuscenes \
            --dst_folder  ./save_root/beam_missing/24beam \
            --num_beam_to_drop 24
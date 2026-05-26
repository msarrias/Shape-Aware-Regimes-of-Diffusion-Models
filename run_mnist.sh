#!/bin/bash

THREADS=20
SAVE_PATH="Saves/History/MNIST_5000_newUnet"
KERNEL="gaussian"
LAPLACIAN="unnormalized"
NORM="log_global_scale_and_shift"
EXP_NAME="hierarchical_log_global_clipping"
SEED=123
python gmms_sagd_distance_matrices.py \
      --data_model MNIST \
      --save_path "$SAVE_PATH" \
      --threads $THREADS \
      --seed $SEED \
      --kernel "$KERNEL" \
      --laplacian "$LAPLACIAN" \
      --norm_type "$NORM" \
      --inject_edges \
      --generate_sasne_embedding \
      --clipping 
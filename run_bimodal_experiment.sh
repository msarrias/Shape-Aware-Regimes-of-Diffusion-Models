#!/bin/bash

# Configuration
THREADS=20
MU=4.0
T=10.0
SAMPLES=1000
STEPS=1000
SEED=123
DS_LIST="2 3 4 5 6 10 15 20 25 30 35 40 45 50 256 1024 4096 16384"

KERNEL="gaussian"
LAPLACIAN="unnormalized"

NORM_TYPES=("log_scale_and_shift" "log_global_scale_and_shift" "scale_and_shift" "norm_wrt_avg_ctd" "norm_wrt_volume")

for idx in "${!NORM_TYPES[@]}"; do
    NORM="${NORM_TYPES[$idx]}"
    EXP_NAME="exp_${idx}"

    echo "Running experiment: $EXP_NAME with norm: $NORM"

    python gmms_sagd_distance_matrices.py \
        --exp_name "$EXP_NAME" \
        --ds $DS_LIST \
        --threads $THREADS \
        --mu $MU \
        --T $T \
        --n_samples $SAMPLES \
        --n_steps $STEPS \
        --seed $SEED \
        --kernel "$KERNEL" \
        --laplacian "$LAPLACIAN" \
        --norm_type "$NORM" \
        --inject_edges
done
#!/bin/bash

# Configuration
THREADS=20
MU=4.0
T=10.0
SAMPLES=1000
STEPS=1000
SEED=123
DS_LIST="2 50 256 1024 16384"

KERNEL="gaussian"
LAPLACIAN="unnormalized"
MODEL="bimodal_gaussian"
DISTANCE="SAGD"
SAVE_PATH="/extra/shared/groups/marinaivan/data_marina/recurrence_matrices/"
EXP_NAME="scale_and_shift_log_scale_bimodal"
NORM="log_scale_and_shift"

python ../analysis/sagd_pipeline.py \
    --exp_name "$EXP_NAME" \
    --save_path "$SAVE_PATH" \
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
    --data_model "$MODEL" \
    --inject_edges \
    --generate_sasne_embedding \
    --distance "$DISTANCE" \
    --clipping
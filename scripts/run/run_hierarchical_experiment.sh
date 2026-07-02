#!/bin/bash

EXP_NAME="exp_02"
THREADS=20
MU=4.0
SIGMA="1.0 1.0 1.0 1.0 1.0 1.0"
T=10.0
SAMPLES=1500
STEPS=1000
SEED=123
DS_LIST="3"

KERNEL="gaussian"
LAPLACIAN="unnormalized"
NORM="log_scale_and_shift"
MODEL="bimodal"
python ../analysis/sagd_pipeline.py \
    --exp_name "$EXP_NAME" \
    --ds "$DS_LIST" \
    --threads $THREADS \
    --mu $MU \
    --T $T \
    --n_samples $SAMPLES \
    --n_steps $STEPS \
    --seed $SEED \
    --kernel "$KERNEL" \
    --data-model "$MODEL" \
    --laplacian "$LAPLACIAN" \
    --norm_type "$NORM" \
    --inject_edges
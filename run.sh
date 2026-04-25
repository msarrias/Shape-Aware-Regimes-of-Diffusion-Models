#!/bin/bash

EXP_NAME="exp_04_fixed_grid"
THREADS=20
MU=4.0
T=10.0
SAMPLES=1000
STEPS=1000
SEED=123456

KERNEL="gaussian"
LAPLACIAN="unnormalized"
NORM="norm_wrt_avg_ctd"

python run_exp.py \
    --exp_name "$EXP_NAME" \
    --ds $(seq 2 15) \
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
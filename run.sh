#!/bin/bash

EXP_NAME="exp_01"
THREADS=20
MU=4.0
T=10.0
SAMPLES=1000
STEPS=1000
SEED=123

KERNEL="gaussian"
LAPLACIAN="unnormalized"
NORM="norm_wrt_avg_ctd"

python gaussians_sagd_embeddings.py \
    --exp_name "$EXP_NAME" \
    --ds $(seq 2 6) \
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
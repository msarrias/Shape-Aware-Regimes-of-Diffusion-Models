#!/bin/bash

# Sweep over n_samples for the bimodal model at a single dimension (D=1024)
# using log_global_scale_and_shift normalization.
# Based on run_bimodal_experiment.sh.

# Configuration
THREADS=20
MU=4.0
T=10.0
STEPS=1000
SEED=123
DS="256 1024 4096"

KERNEL="gaussian"
LAPLACIAN="unnormalized"
MODEL="bimodal_gaussian"
DISTANCE="SAGD"
NORM="log_scale_and_shift"

SAMPLES_LIST="1000 2000 4000"

for SAMPLES in $SAMPLES_LIST; do

    EXP_NAME="${NORM}_D${DS}_n${SAMPLES}"

    echo "Running experiment: $EXP_NAME | norm: $NORM | n_samples: $SAMPLES"

    python ../analysis/sagd_pipeline.py \
        --exp_name "$EXP_NAME" \
        --ds $DS \
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

done

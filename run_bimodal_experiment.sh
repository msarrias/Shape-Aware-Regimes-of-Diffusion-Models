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
MODEL="bimodal"
DISTANCE="SAGD"
NORM_TYPES=("log_global_scale_and_shift" "scale_and_shift" "norm_wrt_avg_ctd" "norm_wrt_volume")

for NORM in "${NORM_TYPES[@]}"; do
    for CLIP in true false; do

        if [ "$CLIP" = "true" ]; then
            EXP_NAME="${NORM}_clipped"
            CLIPPING_FLAG="--clipping"
        else
            EXP_NAME="${NORM}"
            CLIPPING_FLAG=""
        fi

        echo "Running experiment: $EXP_NAME | norm: $NORM | clipping: $CLIP"

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
            --data_model "$MODEL" \
            --inject_edges \
            --generate_sasne_embedding \
            --distance "$DISTANCE" \
            $CLIPPING_FLAG

    done
done
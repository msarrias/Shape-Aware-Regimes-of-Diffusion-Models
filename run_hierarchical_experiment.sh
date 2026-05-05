#!/bin/bash


    D        = 3
    N_MACRO  = 2
    N_MICRO  = 3
    T        = 10.0
    DT       = 0.1

    N_PER_CLUSTER     = [400, 200, 100, 300, 150, 350]
    
    

    # history_forward, times, centers = simulate_hierarchical_ou(
    #     n_per_cluster     = N_PER_CLUSTER,
    #     sigma_per_cluster = SIGMA_PER_CLUSTER,
    #     d=D, T=T, dt=DT,
    #     mu_macro=12, mu_micro=3.5, seed=42
    # )

    history_backward = generate_samples(
        n_samples         = 1500,
        d                 = D,
        T                 = T,
        dt                = DT,
        centers           = centers,
        sigma_per_cluster = SIGMA_PER_CLUSTER,  # ← true sigmas
        mixture_weights   = MIXTURE_WEIGHTS,    # ← true proportions
        n_snapshots       = 100
    )
    
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
python gmms_sagd_distance_matrices.py \
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
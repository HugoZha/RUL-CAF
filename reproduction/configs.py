"""Exact configurations used for every reported experiment in the manuscript.

The per-dataset adaptive settings (window length w and Transformer encoder
depth L) are the CAS-Fuse rules R1-R2 of Sec. 5.1.  Everything else is shared
across the four subsets (FIXED_CONFIG).

Do not change these values when verifying the reported numbers: they are the
values that produced the archived results under ``results/``.
"""

# CAS-Fuse condition-adaptive configurations (manuscript Table 4 / Sec. 5.1)
#   FD001: single condition  -> w=40, L=1
#   FD002: six conditions    -> w=20, L=3  (short-horizon fleet)
#   FD003: single condition  -> w=30, L=1
#   FD004: six conditions    -> w=60, L=3  (long-horizon fleet)
BEST_CONFIG = {
    1: {"window_size": 40, "num_heads": 4, "n_encoder_layers": 1},
    2: {"window_size": 20, "num_heads": 4, "n_encoder_layers": 3},
    3: {"window_size": 30, "num_heads": 4, "n_encoder_layers": 1},
    4: {"window_size": 60, "num_heads": 4, "n_encoder_layers": 3},
}

# Shared hyperparameters (manuscript Sec. 3.2 / Sec. 6.1)
FIXED_CONFIG = {
    "hidden_dim": 128,
    "gru_num_layers": 2,
    "dropout": 0.3,
    "sensor_dropout_p": 0.1,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 150,
    "patience": 20,
    "grad_clip": 1.0,
    "rul_cap": 125,
    "norm_method": "minmax",
    "val_ratio": 0.15,
    "split_seed": 42,          # fixed train/val window split (comparability)
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "loss": "MSE",
    "adversarial": False,      # GRL deliberately removed (Sec. 5.2)
}

# Five-seed audit (Sec. 6.1).  Seed 42 is the archived "seed 42" run from the
# original final experiment; the other four are the E1 multi-seed audit runs.
AUDIT_SEEDS = [42, 7, 123, 2026, 3407]
E1_SEEDS = [7, 123, 2026, 3407]   # seeds trained in the E1 script

# R5 tripwire thresholds (Sec. 5.3)
TRIPWIRE = {
    "pred_std_floor": 0.01,        # Tripwire A: std(y_hat) < 0.01 cycles
    "rmse_tol_pct": 0.02,          # ... or RMSE within 2% of constant predictor
    "attn_entropy_ceiling": 0.99,  # Tripwire B: mean normalised attention entropy
    "plateau_pred_std": 0.1,       # Tripwire C: pred std < 0.1 cycles
    "plateau_epsilon": 0.5,        # ... while val RMSE within eps of std(y_val)
    "plateau_epochs": 5,           # ... for N consecutive epochs
}

DATASETS = [1, 2, 3, 4]

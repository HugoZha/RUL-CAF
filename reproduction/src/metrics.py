"""Evaluation metrics and the R5 collapse tripwires (manuscript Sec. 5.3/6.1)."""

import math

import numpy as np


def phm08_score(pred, true) -> float:
    """NASA PHM08 asymmetric score: exp(-d/13)-1 for early, exp(d/10)-1 for late."""
    s = 0.0
    for di in np.asarray(pred) - np.asarray(true):
        if di < 0:
            s += math.exp(-di / 13.0) - 1
        else:
            s += math.exp(di / 10.0) - 1
    return float(s)


def regression_metrics(pred, true) -> dict:
    """RMSE, MAE, R^2, prediction std and segment-wise errors."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    r2 = float(1 - np.sum((true - pred) ** 2) / (np.sum((true - true.mean()) ** 2) + 1e-8))
    segs = {}
    for lo, hi in [(0, 30), (30, 60), (60, 100), (100, 9999)]:
        m = (true >= lo) & (true < hi)
        if m.sum() > 0:
            segs[f"{lo}-{hi}"] = {
                "n": int(m.sum()),
                "rmse": float(np.sqrt(np.mean((pred[m] - true[m]) ** 2))),
                "mae": float(np.mean(np.abs(pred[m] - true[m]))),
            }
    return {"rmse": rmse, "mae": mae, "r2": r2,
            "pred_std": float(np.std(pred)), "segment_errors": segs}


def constant_predictor_rmse(true) -> float:
    """Closed-form RMSE of the always-mean predictor (Eq. (1) in the paper).

    A collapsed model predicts a single constant; its RMSE converges to the
    standard deviation of the labels.  This is the reference value for
    Tripwire A and the collapsed runs (~43.7 cycles on FD004).
    """
    true = np.asarray(true, dtype=np.float64)
    return float(np.std(true))


def attention_entropy(attn) -> float:
    """Mean normalised entropy over valid causal rows of a (B,H,L,L) tensor.

    Healthy models: 0.70-0.93; collapsed models: ~1.0000 (Sec. 4.2).
    """
    attn = attn.detach().float().cpu().numpy()
    B, H, L, _ = attn.shape
    ents = []
    for t in range(L):
        row = attn[:, :, t, :t + 1]
        row = row / (row.sum(-1, keepdims=True) + 1e-12)
        ent = -(row * np.log2(row + 1e-12)).sum(-1)
        norm = np.log2(t + 1)
        if norm > 0:
            ents.append(float((ent / norm).mean()))
    return float(np.mean(ents))


def check_tripwires(test_rmse, pred_std, y_test, attn_entropy_value, tripwire: dict) -> dict:
    """Tripwires A and B (post-hoc; Tripwire C is measured during training)."""
    const_rmse = constant_predictor_rmse(y_test)
    a_constancy = bool(pred_std < tripwire["pred_std_floor"])
    a_rmse_match = bool(abs(test_rmse - const_rmse) <=
                        tripwire["rmse_tol_pct"] * const_rmse)
    b_entropy = bool(attn_entropy_value > tripwire["attn_entropy_ceiling"])
    return {
        "constant_predictor_rmse": const_rmse,
        "tripwire_A_constancy": a_constancy,
        "tripwire_A_rmse_match": a_rmse_match,
        "tripwire_B_attention_entropy": b_entropy,
        "collapsed": bool(a_constancy or a_rmse_match or b_entropy),
        "rms_above_40": bool(test_rmse > 40),
    }

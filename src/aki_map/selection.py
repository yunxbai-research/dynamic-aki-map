"""Historical S1 candidate grid and patient-balanced selection rule."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASELINE = "stage_depth4_l2_15_trees350"
PROFILES = {"depth4_l2_15": {"depth": 4, "l2_leaf_reg": 15.0},
            "depth5_l2_20": {"depth": 5, "l2_leaf_reg": 20.0}}
ROUNDS = (150, 350, 600)

def configurations():
    return [dict(name=BASELINE, sampling="stage", profile="depth4_l2_15",
                 iterations=350, **PROFILES["depth4_l2_15"])] + [
        dict(name=f"uniform_{profile}_trees{count}", sampling="uniform",
             profile=profile, iterations=count, **settings)
        for profile, settings in PROFILES.items() for count in ROUNDS
    ]


def s1_scores(pairs, labels, probability):
    probability = np.asarray(probability)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid S1 probabilities")
    _, inverse, counts = np.unique(pairs[:, 0], return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return dict(auroc=float(roc_auc_score(labels, probability, sample_weight=weights)),
                auprc=float(average_precision_score(labels, probability, sample_weight=weights)),
                brier=float(brier_score_loss(labels, probability, sample_weight=weights)))


def select_candidate(scores):
    reference = next(row for row in scores if row["name"] == BASELINE)
    eligible = [row for row in scores
                if row["auroc"] >= reference["auroc"] - .005
                and row["auprc"] >= .95 * reference["auprc"]]
    best = max(row["auroc"] for row in eligible)
    near = [row for row in eligible if row["auroc"] >= best - .0005]
    # Do not replace the reference for a near-tied inner-CV result.
    return min(near, key=lambda row: (row["name"] != BASELINE,
                                     row["iterations"], row["depth"],
                                     -row["auprc"], row["name"]))["name"]

"""Patient-balanced predictive metrics with patient-cluster bootstrap."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from .risk import patient_weights


def metrics(y, prediction, weights):
    return np.array([roc_auc_score(y, prediction, sample_weight=weights),
                     average_precision_score(y, prediction, sample_weight=weights),
                     brier_score_loss(y, prediction, sample_weight=weights)])


def evaluate(data, prediction, cohort, reps, seed):
    ids = data['pairs'][:, 0]
    patients, inverse = np.unique(ids, return_inverse=True)
    weights = patient_weights(ids)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        multiplicity = np.bincount(rng.integers(len(patients), size=len(patients)), minlength=len(patients))
        w = weights*multiplicity[inverse]
        if any(len(np.unique(data['y'][w > 0, j])) < 2 for j in range(2)):
            continue
        boot.append([metrics(data['y'][:, j], prediction[:, j], w) for j in range(2)])
    if not boot:
        raise ValueError('No bootstrap replicate contains both outcome classes')
    intervals = np.quantile(boot, [.025, .975], axis=0)
    result = []
    for j, name in enumerate(['Any postoperative AKI', 'Moderate-to-severe postoperative AKI']):
        point = metrics(data['y'][:, j], prediction[:, j], weights)
        for k, metric in enumerate(['AUROC', 'AUPRC', 'Brier']):
            result.append(dict(cohort=cohort, outcome=name, metric=metric,
                estimate=float(point[k]), ci_lower=float(intervals[0, j, k]),
                ci_upper=float(intervals[1, j, k]), bootstrap_valid=len(boot),
                patients=len(patients), landmarks=len(ids), synthetic=True))
    return result


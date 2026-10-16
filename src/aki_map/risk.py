"""Final selected risk architecture, trained afresh on supplied data."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORY, NUMERIC, STATIC
from .methods import stage_sample_pairs, uniform_pairs, fit_calibration, calibrate


def patient_weights(patient_indices):
    _, inverse, counts = np.unique(patient_indices, return_inverse=True, return_counts=True)
    return (1. / counts[inverse]).astype(np.float32)


def rows_for(data, patients):
    return np.flatnonzero(np.isin(data['pairs'][:, 0], patients))


def sampled_rows(data, patients, count, seed, mode):
    sample = uniform_pairs if mode == 'uniform' else stage_sample_pairs
    query = sample(data, patients, count, seed)
    stride = int(data['pairs'][:, 1].max()) + 1
    keys = data['pairs'][:, 0].astype(np.int64)*stride + data['pairs'][:, 1]
    rows = np.searchsorted(keys, query[:, 0].astype(np.int64)*stride + query[:, 1])
    if not np.array_equal(data['pairs'][rows], query):
        raise ValueError('Sampled landmark absent from design')
    return rows


def fit_encoder(data, rows, patients):
    raw = data['raw'][rows]
    medians = pd.DataFrame(raw).median().fillna(0).to_numpy(np.float32)
    category = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    category.fit(data['frame'].iloc[patients][[CATEGORY]])
    return dict(medians=medians, category=category)


def encode_risk(data, rows, encoder):
    raw = data['raw'][rows]
    missing = ~np.isfinite(raw)
    # elapsed_hours has a missingness flag in the archived reference encoder.
    flags = [name not in STATIC for name in NUMERIC]
    cats = data['frame'].iloc[data['pairs'][rows, 0]][[CATEGORY]]
    return np.column_stack((np.where(missing, encoder['medians'], raw),
                            missing[:, flags], encoder['category'].transform(cats))).astype(np.float32)


def fit_ensemble(data, patients, config):
    result = []
    for seed in config['risk_seeds']:
        fitted = []
        for j, (key, sampling) in enumerate([('any_aki', 'uniform'), ('moderate_severe_aki', 'stage')]):
            rows = sampled_rows(data, patients, config['risk_landmarks_per_patient'], seed, sampling)
            encoder = fit_encoder(data, rows, patients)
            model = CatBoostClassifier(**config[key], learning_rate=config['learning_rate'],
                loss_function='Logloss', random_seed=seed+j, thread_count=config['threads'],
                verbose=False, allow_writing_files=False)
            model.fit(encode_risk(data, rows, encoder), data['y'][rows, j])
            fitted.append(dict(model=model, encoder=encoder))
        result.append(fitted)
    return result


def predict_raw(ensemble, data, rows):
    predictions = np.zeros((len(rows), 2), float)
    for fitted in ensemble:
        for j, item in enumerate(fitted):
            predictions[:, j] += item['model'].predict_proba(
                encode_risk(data, rows, item['encoder']))[:, 1] / len(ensemble)
    return predictions


def assist_score(probability, weight):
    z = logit(np.clip(probability, 1e-7, 1-1e-7))
    return (1-weight)*z[:, 1] + weight*z[:, 0]


def apply_calibration(probability, calibration, weight):
    result = probability.copy()
    result[:, 1] = calibrate(assist_score(probability, weight), calibration)
    return result


def train_crossfit(data, patients, config, tag_number):
    """Inner OOF calibration and response anchors, then full-training-set fits."""
    strata = data['frame'].iloc[patients].any_aki.to_numpy()*2 + data['frame'].iloc[patients].moderate_severe_aki.to_numpy()
    splitter = StratifiedKFold(config['inner_folds'], shuffle=True,
                              random_state=config['inner_split_seed']+tag_number)
    oof = np.full((len(data['pairs']), 2), np.nan)
    audit = []
    for inner, (it, iv) in enumerate(splitter.split(patients, strata), 1):
        train, valid = patients[it], patients[iv]
        rows = rows_for(data, valid)
        models = fit_ensemble(data, train, config)
        oof[rows] = predict_raw(models, data, rows)
        audit.append(dict(inner_fold=inner, train_patients=train.tolist(), valid_patients=valid.tolist()))
    used = rows_for(data, patients)
    weights = patient_weights(data['pairs'][used, 0])
    calibration = fit_calibration(data['y'][used, 1],
                                  assist_score(oof[used], config['s2_assist_weight']), weights)
    anchored = np.full_like(oof, np.nan)
    anchored[used] = apply_calibration(oof[used], calibration, config['s2_assist_weight'])
    return dict(models=fit_ensemble(data, patients, config), calibration=calibration,
                training_oof=anchored, inner_assignments=audit)


def predict(deployment, data, rows, config):
    return apply_calibration(predict_raw(deployment['models'], data, rows),
                              deployment['calibration'], config['s2_assist_weight'])

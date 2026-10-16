"""Build the final 112 pre-encoding features from harmonised observations."""
from importlib.resources import files
import json

import numpy as np
import pandas as pd

from .temporal import patient_grid, sequence_and_summary, multiscale_features


FEATURES = json.loads(files('aki_map').joinpath('feature_order.json').read_text())
CATEGORY = 'surgery_department_detail'
NUMERIC = [name for name in FEATURES if name != CATEGORY]
STATIC = ['age', 'sex', 'Hypertension', 'asa', 'emor', 'baseline_recent',
          'albumin', 'drug_diur', 'eGFR', 'eGFR_lt60']
LIMITS = {'sys': (40, 300), 'dia': (20, 200), 'map': (20, 250), 'hr': (20, 250)}


def egfr(age, sex, creatinine):
    """CKD-EPI 2021, creatinine in micromol/L; female=0, male=1."""
    scr = np.asarray(creatinine, float) / 88.4
    female = np.asarray(sex) == 0
    k = np.where(female, .7, .9)
    alpha = np.where(female, -.241, -.302)
    ratio = scr / k
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (142 * np.minimum(ratio, 1)**alpha * np.maximum(ratio, 1)**-1.2 *
                  .9938**np.asarray(age) * np.where(female, 1.012, 1.))
    return np.where((scr <= 0) | pd.isna(np.asarray(sex)), np.nan, result)


def build_landmarks(baseline, observations):
    """One preselected episode per patient; minutes measured from anaesthesia start.

    Baseline values must already be available before anaesthesia. Future MAP is
    returned separately as a training/evaluation target, never in raw features.
    """
    baseline = baseline.sort_values('patient_id').reset_index(drop=True).copy()
    if baseline.patient_id.duplicated().any():
        raise ValueError('Provide exactly one indexed episode per patient')
    if not np.all(baseline.duration_min >= 120):
        raise ValueError('Indexed anaesthesia duration must be at least 120 min')
    if not baseline.sex.dropna().isin([0, 1]).all():
        raise ValueError('sex must be female=0, male=1')
    required = set(STATIC) - {'eGFR', 'eGFR_lt60'}
    if not required.issubset(baseline):
        raise ValueError(f'Missing baseline fields: {sorted(required-set(baseline))}')
    if not ((baseline.any_aki.isin([0, 1])) &
            (baseline.moderate_severe_aki.isin([0, 1])) &
            (baseline.moderate_severe_aki <= baseline.any_aki)).all():
        raise ValueError('Outcomes must be binary and nested')
    if not set(observations.patient_id).issubset(set(baseline.patient_id)):
        raise ValueError('Unknown patient in observations')
    baseline['eGFR'] = egfr(baseline.age, baseline.sex, baseline.baseline_recent)
    baseline['eGFR_lt60'] = np.where(baseline.eGFR.notna(), (baseline.eGFR < 60).astype(float), np.nan)
    baseline[CATEGORY] = baseline[CATEGORY].fillna('Unknown').astype(str)
    groups = {pid: group for pid, group in observations.groupby('patient_id')}
    patients, matrices, pairs, targets = [], [], [], []
    for i, row in baseline.iterrows():
        if row.patient_id not in groups:
            raise ValueError(f'No monitoring data for patient index {i}')
        r = groups[row.patient_id].copy()
        r['value'] = pd.to_numeric(r.value, errors='coerce')
        r['duration'] = float(row.duration_min)
        lower = r.component.map({c: b[0] for c, b in LIMITS.items()})
        upper = r.component.map({c: b[1] for c, b in LIMITS.items()})
        valid = (r.minute.ge(0) & r.minute.le(row.duration_min) &
                 r.value.between(lower, upper))
        r = r.loc[valid].copy()
        r['bin'] = np.ceil(r.minute / 5).astype(int)
        r = r[r.bin * 5 <= row.duration_min]
        grid = patient_grid(r)
        _, summary, names, landmarks, _ = sequence_and_summary(grid)
        multi, multi_names = multiscale_features(grid['measured'])
        if not len(landmarks):
            raise ValueError(f'No eligible prediction landmark for patient index {i}')
        dynamic = pd.DataFrame(np.column_stack((summary, multi)), columns=names+multi_names)
        raw = np.empty((len(landmarks), len(NUMERIC)), np.float32)
        for j, name in enumerate(NUMERIC):
            raw[:, j] = row[name] if name in STATIC else dynamic[name].to_numpy()[landmarks]
        future = np.full(len(landmarks), np.nan, np.float32)
        for k, t in enumerate(landmarks):
            part = grid['measured'][t+1:t+4, 2]
            if len(part) == 3 and np.isfinite(part).all():
                future[k] = part.mean()
        patients.append(dict(landmarks=landmarks, measured=grid['measured']))
        matrices.append(raw)
        pairs.extend((i, int(t)) for t in landmarks)
        targets.append(future)
    pair_array = np.asarray(pairs, np.int32)
    return dict(frame=baseline, patients=patients, pairs=pair_array,
                raw=np.vstack(matrices), future_map=np.concatenate(targets),
                y=baseline[['any_aki','moderate_severe_aki']].to_numpy(np.float32)[pair_array[:, 0]])


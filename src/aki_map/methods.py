"""Sampling and transformations; no fitted study parameters are stored here."""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit

TOL = .20
MAP_GRID = np.arange(60., 91., dtype=np.float32)
FLAT_CONTRAST = .05

def stage_sample_pairs(bundle, patients, count, seed):
    """Patient-equal sampling with early/middle/late coverage where available."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in patients:
        landmarks = np.asarray(bundle["patients"][int(patient)]["landmarks"], int)
        bands = [landmarks[landmarks * 5 <= 120],
                 landmarks[(landmarks * 5 > 120) & (landmarks * 5 <= 180)],
                 landmarks[landmarks * 5 > 180]]
        chosen = []
        while len(chosen) < count:
            available = [v for v in bands if len(v)]
            pool = available[len(chosen) % len(available)] if available else landmarks
            chosen.append(int(rng.choice(pool)))
        rows.extend((int(patient), t) for t in chosen)
    return np.asarray(rows, np.int32).reshape(-1, 2)


def uniform_pairs(bundle, patients, count, seed):
    """Exactly count rows per patient; every eligible landmark is equiprobable."""
    rng = np.random.default_rng(seed)
    result = []
    for patient in patients:
        landmarks = np.asarray(bundle["patients"][int(patient)]["landmarks"], int)
        if not len(landmarks):
            raise ValueError(f"Patient {patient} has no eligible landmark")
        chosen = rng.choice(landmarks, count, replace=len(landmarks) < count)
        result.extend((int(patient), int(t)) for t in chosen)
    return np.asarray(result, np.int32).reshape(-1, 2)


def fit_calibration(y, score, weights):
    """Fit only on UNION inner OOF, with positive slope to preserve within-fit ranks."""
    y, score, weights = (np.asarray(x, np.float64) for x in (y, score, weights))
    if len(np.unique(y)) != 2 or not np.isfinite(score).all() or np.any(weights <= 0):
        raise ValueError("S2 calibration requires finite scores, positive weights and both labels")
    w = weights / weights.sum()
    center = float(np.sum(w*score))
    scale = max(float(np.sqrt(np.sum(w*(score-center)**2))), 1e-6)
    z = (score-center)/scale

    def loss(beta):
        eta = beta[0] + beta[1]*z
        error = (expit(eta)-y)*w
        return float(np.sum(w*(np.logaddexp(0, eta)-y*eta))), np.array([
            error.sum(), np.sum(error*z)])

    fitted = minimize(loss, [logit(np.sum(w*y)), 1.], jac=True, method="L-BFGS-B",
                      bounds=[(None, None), (1e-4, 20.)],
                      options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"S2 calibration failed: {fitted.message}")
    intercept, slope = map(float, fitted.x)
    return dict(center=center, scale=scale, intercept=intercept, slope=slope,
                raw_intercept=intercept-slope*center/scale, raw_slope=slope/scale,
                slope_at_boundary=bool(slope <= 1.01e-4 or slope >= 19.999),
                fitted_prevalence=float(np.sum(w*y)), fit_status=str(fitted.message))


def calibrate(score, params):
    if params["slope"] <= 0:
        raise ValueError("Calibration must preserve score ordering")
    return expit(params["intercept"] + params["slope"] *
                 (np.asarray(score)-params["center"])/params["scale"])


def encode_fit(raw, base, categories, weights):
    values = np.column_stack((logit(np.clip(base, 1e-6, 1-1e-6)), raw))
    median = pd.DataFrame(values).median().fillna(0).to_numpy(np.float32)
    filled = np.where(np.isfinite(values), values, median)
    mean = np.average(filled, axis=0, weights=weights)
    sd = np.maximum(np.sqrt(np.average((filled-mean)**2, axis=0, weights=weights)), 1e-3)
    return dict(median=median, mean=mean, scale=sd,
                categories=sorted(set(map(str, categories))))


def encode(raw, base, categories, encoder):
    values = np.column_stack((logit(np.clip(base, 1e-6, 1-1e-6)), raw))
    missing = ~np.isfinite(values)
    numeric = (np.where(missing, encoder['median'], values)-encoder['mean'])/encoder['scale']
    onehot = np.column_stack([np.asarray(categories).astype(str) == item
                              for item in encoder['categories']])
    return np.column_stack((numeric, missing, onehot)).astype(np.float32)


@torch.no_grad()
def candidate_curves(model, x, base, device):
    model.eval()
    state = torch.as_tensor(x, dtype=torch.float32, device=device)
    anchor = torch.as_tensor(logit(np.clip(base, 1e-6, 1-1e-6)),
                             dtype=torch.float32, device=device)
    mu, offset, slope, curvature = model.components(state)
    grid = torch.as_tensor(MAP_GRID, device=device)
    dose = ((grid[None, :]-mu[:, None])/15).clamp(-2, 2)
    response = (offset[:, None, :] + slope[:, None, :]*dose[:, :, None]
                + curvature[:, None, :]*dose[:, :, None].square())
    return torch.sigmoid(anchor[:, None, :]+response).cpu().numpy()


def choose_target(curves, scales):
    if curves.shape[1:] != (31, 2) or not np.isfinite(curves).all():
        raise ValueError('Invalid candidate-MAP response curves')
    norm = [(curves[:, :, i]-curves[:, :, i].min(axis=1, keepdims=True)) /
            max(scales[f'S{i+1}'], 1e-6) for i in range(2)]
    composite = .75*np.clip(norm[0], 0, 1) + .25*np.clip(norm[1], 0, 1)
    contrast = np.ptp(composite, axis=1)
    accepted = composite <= (composite.min(axis=1)+TOL*contrast)[:, None]
    target = MAP_GRID[np.argmax(accepted, axis=1)].astype(float)
    target[contrast < FLAT_CONTRAST] = np.nan
    return target, contrast

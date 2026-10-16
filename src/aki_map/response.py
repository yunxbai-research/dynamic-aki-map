"""Response training and threshold inference using newly fitted state only."""
import random
import numpy as np
import torch
from scipy.special import logit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import CATEGORY
from .methods import encode_fit, encode, candidate_curves, choose_target
from .network import NativeMAPResponseHead
from .risk import patient_weights, rows_for


def response_rows(data, patients, count, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for patient in patients:
        eligible = rows_for(data, [patient])
        eligible = eligible[np.isfinite(data['future_map'][eligible])]
        if not len(eligible):
            continue
        if len(eligible) > count:
            minute = data['pairs'][eligible, 1]*5
            bands = [eligible[minute <= 120], eligible[(minute > 120) & (minute <= 180)],
                     eligible[minute > 180]]
            allocation = [count//3 + int(i < count % 3) for i in range(3)]
            chosen = []
            for band, n in zip(bands, allocation):
                if len(band):
                    chosen.extend(rng.choice(band, min(n, len(band)), replace=False).tolist())
            if len(chosen) < count:
                remaining = np.setdiff1d(eligible, chosen)
                chosen.extend(rng.choice(remaining, min(count-len(chosen), len(remaining)),
                                          replace=False).tolist())
            eligible = np.sort(chosen)
        rows.extend(eligible)
    return np.asarray(rows, int)


def fit_response(data, patients, anchors, config):
    rows = response_rows(data, patients, config['response_landmarks_per_patient'],
                         config['response_sample_seed'])
    if not len(rows) or not np.isfinite(anchors[rows]).all():
        raise ValueError('Response fitting requires finite OOF anchors and complete future windows')
    weights = patient_weights(data['pairs'][rows, 0])
    categories = data['frame'][CATEGORY].to_numpy()[data['pairs'][rows, 0]]
    encoder = encode_fit(data['raw'][rows], anchors[rows], categories, weights)
    x = encode(data['raw'][rows], anchors[rows], categories, encoder)
    base = logit(np.clip(anchors[rows], 1e-6, 1-1e-6)).astype(np.float32)
    labels = data['y'][rows]
    prevalence = np.average(labels, axis=0, weights=weights)
    entropy = -(prevalence*np.log(np.clip(prevalence, 1e-6, 1)) +
                (1-prevalence)*np.log(np.clip(1-prevalence, 1e-6, 1)))
    task_scale = torch.as_tensor(1/np.maximum(entropy, 1e-3), dtype=torch.float32)
    ds = TensorDataset(*(torch.from_numpy(np.asarray(v, np.float32))
                         for v in (x, base, data['future_map'][rows], labels, weights)))
    models, losses = [], []
    for seed in config['response_seeds']:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = NativeMAPResponseHead(x.shape[1], 64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['response_learning_rate'],
                                      weight_decay=config['response_weight_decay'])
        loader = DataLoader(ds, batch_size=config['response_batch_size'], shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
        history = []
        for _ in range(config['response_epochs']):
            model.train()
            total, mass = 0., 0.
            for xb, bb, ab, yb, wb in loader:
                optimizer.zero_grad(set_to_none=True)
                logits, mu, slope, curvature = model(xb, bb, ab)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
                per_row = (bce*task_scale[None]).mean(1) + .1*((mu-ab)/15).square()
                per_row += .01*(slope.square()+curvature.square()).mean(1)
                loss = (per_row*wb).sum()/wb.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach())*float(wb.sum())
                mass += float(wb.sum())
            history.append(total/mass)
        models.append(model)
        losses.append(history)
    deployment = dict(models=models, encoder=encoder, training_rows=rows, losses=losses)
    # The scale sample draws one of each patient's sampled response-training rows.
    rng = np.random.default_rng(config['curve_scale_sample_seed'])
    _, first, counts = np.unique(data['pairs'][rows, 0], return_index=True, return_counts=True)
    scale_rows = rows[first+rng.integers(counts)]
    if len(scale_rows) > 8000:
        scale_rows = np.sort(rng.choice(scale_rows, 8000, replace=False))
    curves = predict_curves(deployment, data, scale_rows, anchors[scale_rows])
    deployment['scales'] = {f'S{i+1}': max(float(np.quantile(np.ptp(curves[:, :, i], axis=1), .75)), 1e-6)
                            for i in range(2)}
    return deployment


def predict_curves(deployment, data, rows, base):
    categories = data['frame'][CATEGORY].to_numpy()[data['pairs'][rows, 0]]
    x = encode(data['raw'][rows], base, categories, deployment['encoder'])
    result = np.zeros((len(rows), 31, 2), np.float32)
    for model in deployment['models']:
        for start in range(0, len(rows), 4096):
            stop = min(start+4096, len(rows))
            result[start:stop] += candidate_curves(model, x[start:stop], base[start:stop], 'cpu') / len(deployment['models'])
    return result


def predict_threshold(deployment, data, rows, base):
    curves = predict_curves(deployment, data, rows, base)
    targets, contrast = choose_target(curves, deployment['scales'])
    return targets, contrast


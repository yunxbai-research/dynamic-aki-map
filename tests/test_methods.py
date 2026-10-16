import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aki_map.features import FEATURES, STATIC, build_landmarks, egfr
from aki_map.methods import choose_target, encode, encode_fit, fit_calibration, calibrate
from aki_map.network import NativeMAPResponseHead
from aki_map.risk import encode_risk, fit_encoder, patient_weights, sampled_rows
from aki_map.synthetic import generate
from aki_map.temporal import patient_grid


def fixture():
    b, v = generate(90, 20261016, 'TEST')
    return b, v, build_landmarks(b, v)


def test_feature_order_and_causal_boundary():
    b, v, data = fixture()
    assert len(FEATURES) == 112 and data['raw'].shape[1] == 111
    changed = v.copy()
    changed.loc[changed.minute > 120, 'value'] += 12
    later = build_landmarks(b, changed)
    early = data['pairs'][:, 1]*5 <= 120
    np.testing.assert_array_equal(data['pairs'], later['pairs'])
    np.testing.assert_allclose(data['raw'][early], later['raw'][early], equal_nan=True)
    # A measurement at 120.01 cannot enter the prediction at 120.
    extra = v.iloc[[0]].copy()
    extra['minute'], extra['component'], extra['value'] = 120.01, 'map', 210.
    after = build_landmarks(b, pd.concat([v, extra], ignore_index=True))
    np.testing.assert_allclose(data['raw'][early], after['raw'][early], equal_nan=True)


def test_patient_equal_sampling_and_encoding():
    _, _, data = fixture()
    patients = np.arange(len(data['frame']))
    for mode in ['uniform','stage']:
        rows = sampled_rows(data, patients, 12, 42, mode)
        assert np.all(np.unique(data['pairs'][rows,0], return_counts=True)[1] == 12)
    rows = sampled_rows(data, patients, 12, 42, 'uniform')
    encoder = fit_encoder(data, rows, patients)
    x = encode_risk(data, rows, encoder)
    expected = 111 + (111-len(STATIC)) + len(encoder['category'].categories_[0])
    assert x.shape == (len(rows),expected) and np.isfinite(x).all()
    w = patient_weights(data['pairs'][:,0])
    for patient in patients:
        assert np.isclose(w[data['pairs'][:,0] == patient].sum(),1.)


def test_fitted_transform_uses_training_parameters():
    raw = np.array([[1.,np.nan],[3.,4.],[5.,6.]],np.float32)
    base = np.array([[.1,.01],[.2,.02],[.3,.03]])
    encoder = encode_fit(raw,base,['A','A','B'],np.ones(3))
    external = encode(np.array([[np.nan,8.]]),np.array([[.2,.02]]),['UNSEEN'],encoder)
    assert external.shape[1] == 10
    np.testing.assert_array_equal(external[0,-2:],[0,0])
    assert external[0,6] == 1  # raw first column's missing indicator


def test_positive_slope_calibration():
    rng = np.random.default_rng(42)
    score = rng.normal(size=300)
    y = rng.binomial(1,1/(1+np.exp(-score)))
    fit = fit_calibration(y,score,np.ones(len(y)))
    assert fit['slope'] > 0
    assert np.all(np.diff(calibrate(np.linspace(-4,4,31),fit)) > 0)


def test_network_initial_anchor_and_threshold_rule():
    torch.manual_seed(42)
    model = NativeMAPResponseHead(20)
    x, base, actual = torch.randn(4,20),torch.randn(4,2),torch.full((4,),80.)
    prediction,*_ = model(x,base,actual)
    torch.testing.assert_close(prediction,base)
    # Falling risk: lowest accepted candidate is 84, not 90.
    curves = np.stack([np.linspace(1,0,31),np.linspace(1,0,31)],axis=-1)[None]
    targets, contrast = choose_target(curves,{'S1':1.,'S2':1.})
    assert targets[0] == 84 and contrast[0] == 1
    flat,_ = choose_target(np.ones((1,31,2))*.2,{'S1':1.,'S2':1.})
    assert np.isnan(flat[0])


def test_ablation_sets():
    root = Path(__file__).resolve().parents[1]
    nodes = json.loads((root/'configs/ablation_nodes.json').read_text())
    assert [node['clinical_field_count'] for node in nodes] == [28,25,23,19,15,13,9]
    assert [len(node['features']) for node in nodes] == [177,153,150,146,142,112,106]
    assert nodes[5]['features'] == FEATURES


def test_source_priority_and_derived_map_timestamp():
    records = pd.DataFrame([
        (1,1.,120.,'art','sys',120.),(1,2.,120.,'art','dia',60.),
        (1,3.,120.,'nibp','map',95.)],
        columns=['bin','minute','duration','source','component','value'])
    grid = patient_grid(records)
    # Same-source ART-derived MAP takes priority over lower-priority NIBP.
    assert grid['measured'][1,2] == 80
    assert grid['stamp'][1,2] == 1  # older component time, never the later one
    assert grid['source'][1] == 1 and grid['method'][1] == 2


def test_egfr_unknown_sex_and_invalid_creatinine():
    values = egfr(np.array([60,60,60]),np.array([np.nan,0,1]),np.array([70,0,70]))
    assert np.isnan(values[:2]).all() and np.isfinite(values[2])


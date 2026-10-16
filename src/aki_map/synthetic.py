"""Artificial records generated from fixed distributions, never from patient data."""
import numpy as np
import pandas as pd


def generate(n, seed, prefix):
    if n < 90:
        raise ValueError('Use at least 90 synthetic patients for stratified cross-fitting')
    rng = np.random.default_rng(seed)
    stage = np.zeros(n, int)
    stage[:max(18, n//8)] = 2
    stage[max(18, n//8):max(18, n//8)+n//5] = 1
    rng.shuffle(stage)
    age = rng.integers(25, 86, n)
    ht = rng.binomial(1, .4, n)
    durations = rng.choice([120, 150, 180, 240, 300, 540], n)
    baseline = pd.DataFrame(dict(patient_id=[f'{prefix}_SYNTH_{i:05d}' for i in range(n)],
        age=age, sex=rng.binomial(1, .5, n), Hypertension=ht,
        asa=rng.choice([1, 2, 3, 4], n, p=[.1,.4,.4,.1]), emor=rng.binomial(1,.08,n),
        baseline_recent=np.clip(rng.normal(70+stage*8,15,n),35,150),
        albumin=rng.normal(40-stage,4,n), drug_diur=rng.binomial(1,.1,n),
        surgery_department_detail=rng.choice(['General surgery','Orthopaedics','Neurosurgery','Other'],n),
        duration_min=durations, any_aki=(stage>=1).astype(int), moderate_severe_aki=(stage>=2).astype(int)))
    baseline.loc[rng.random(n)<.1, 'albumin'] = np.nan
    records = []
    for i, row in baseline.iterrows():
        latent = 82 + ht[i]*5 - stage[i]*6 + rng.normal(0,5)
        for minute in np.arange(0, durations[i]+1, 5):
            mp = latent + 7*np.sin(minute/70) + rng.normal(0,4)
            dbp = mp-rng.normal(15,2)
            sbp = 3*mp-2*dbp
            # Some missing DBP bins; direct MAP remains present in the example.
            for component, value in [('sys',sbp),('dia',dbp),('map',mp)]:
                if component == 'dia' and rng.random()<.03:
                    continue
                records.append((row.patient_id,float(minute),component,'art',float(value)))
    observations = pd.DataFrame(records, columns=['patient_id','minute','component','source','value'])
    return baseline, observations


"""Run the synthetic example without reading study files or serialising models."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import FEATURES, build_landmarks
from .synthetic import generate
from .pipeline import run
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/smoke.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/synthetic-demo'))
    parser.add_argument('--patients', type=int, default=120)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if list(output.iterdir()):
        raise ValueError('Use an empty output directory to avoid mixing runs')
    torch.set_num_threads(config['threads'])
    db, dv = generate(args.patients, 20261016, 'DEV')
    eb, ev = generate(max(90, args.patients//2), 20261017, 'EXT')
    development, external = build_landmarks(db, dv), build_landmarks(eb, ev)
    results = run(development, external, config)
    metrics = []
    for cohort, data, baseline, observations in [('development',development,db,dv),('external',external,eb,ev)]:
        p = results[f'{cohort}_probability']
        ids = data['pairs'][:, 0]
        frame = pd.DataFrame(dict(patient_id=data['frame'].patient_id.to_numpy()[ids],
            elapsed_min=data['pairs'][:,1]*5, any_aki=data['y'][:,0], moderate_severe_aki=data['y'][:,1],
            probability_any_aki=p[:,0], probability_moderate_severe_aki=p[:,1],
            target_map=results[f'{cohort}_target'], curve_contrast=results[f'{cohort}_contrast'],
            synthetic=True))
        frame.to_csv(output/f'{cohort}_synthetic_predictions.csv', index=False)
        baseline.to_csv(output/f'{cohort}_synthetic_baseline.csv',index=False)
        observations.to_csv(output/f'{cohort}_synthetic_observations.csv',index=False)
        metrics.extend(evaluate(data,p,cohort,config['bootstrap_reps'],config['bootstrap_seed']))
    pd.DataFrame(metrics).to_csv(output/'synthetic_metrics.csv',index=False)
    # Only artificial patient indices appear in this audit; weights remain in memory.
    (output/'synthetic_fold_audit.json').write_text(json.dumps(results['isolation_audit'],indent=2))
    environment = {p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost','torch']}
    manifest = dict(synthetic=True, config=config, python=platform.python_version(), packages=environment,
        numeric_features=len(FEATURES)-1, preencoding_features=len(FEATURES),
        trained_study_weights_included=False, model_files_saved=False,
        note='Synthetic performance is not an estimate of clinical performance.')
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Completed synthetic example: {output.resolve()}',flush=True)


if __name__ == '__main__':
    main()

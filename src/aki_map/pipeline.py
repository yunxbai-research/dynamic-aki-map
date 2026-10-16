"""Patient-level outer evaluation, then full-development fits for external data."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .risk import rows_for, train_crossfit, predict
from .response import fit_response, predict_threshold


def run(development, external, config, outer_assignment=None):
    patients = np.arange(len(development['frame']))
    if outer_assignment is None:
        strata = development['frame'].any_aki*2 + development['frame'].moderate_severe_aki
        split = StratifiedKFold(config['outer_folds'], shuffle=True,
                               random_state=config['synthetic_split_seed'])
        folds = np.empty(len(patients), int)
        for fold, (_, valid) in enumerate(split.split(patients, strata), 1):
            folds[valid] = fold
    else:
        folds = np.asarray(outer_assignment, int)
        if folds.shape != (len(patients),) or set(folds) != set(range(1, config['outer_folds']+1)):
            raise ValueError('Invalid patient-level outer fold assignment')
    n = len(development['pairs'])
    oof = np.full((n, 2), np.nan)
    targets = np.full(n, np.nan)
    contrast = np.full(n, np.nan)
    audit = []
    for fold in range(1, config['outer_folds']+1):
        train, valid = patients[folds != fold], patients[folds == fold]
        rows = rows_for(development, valid)
        print(f'Outer fold {fold}/{config["outer_folds"]}: risk cross-fitting and response training', flush=True)
        risk = train_crossfit(development, train, config, fold)
        response = fit_response(development, train, risk['training_oof'], config)
        oof[rows] = predict(risk, development, rows, config)
        targets[rows], contrast[rows] = predict_threshold(response, development, rows, oof[rows])
        audit.append(dict(outer_fold=fold, train_patients=train.tolist(), valid_patients=valid.tolist(),
                          inner=risk['inner_assignments']))
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete outer OOF predictions')
    print('Full-development fits and synthetic external inference', flush=True)
    risk = train_crossfit(development, patients, config, 0)
    # Archived full response training used outer OOF risks as its anchors.
    response = fit_response(development, patients, oof, config)
    erows = np.arange(len(external['pairs']))
    epred = predict(risk, external, erows, config)
    etarget, econtrast = predict_threshold(response, external, erows, epred)
    return dict(development_probability=oof, development_target=targets,
                development_contrast=contrast, external_probability=epred,
                external_target=etarget, external_contrast=econtrast,
                outer_folds=folds, isolation_audit=audit)


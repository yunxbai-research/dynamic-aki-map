# Dynamic AKI-MAP

Dynamic postoperative acute kidney injury (AKI) prediction using CatBoost risk
models and a neural mean arterial pressure (MAP) response model. Includes
temporal feature processing, patient-level cross-fitting, MAP-threshold
estimation and a runnable synthetic-data example.

## Data format

- **Baseline table:** one indexed anaesthesia episode per patient, with preoperative variables, episode duration and two AKI outcome labels.
- **Monitoring table:** `patient_id`, `minute`, `component`, `source` and `value`.
- **Blood pressure:** `sys` (SBP), `dia` (DBP) and `map` (MAP), in mm Hg.
- **Sources:** `art`, `nibp` or `generic`.
- **Time:** minutes from anaesthesia start; episodes must last at least 120 minutes. Predictions begin at 60 minutes and repeat at eligible 5-minute boundaries.
- **Features:** 13 clinical inputs produce 112 pre-encoding features: 111 numeric and one categorical.

Use the most recent available pre-anaesthesia measurements for creatinine and
albumin.

Outcome labels are used for training and evaluation only. The number of encoded
columns depends on the surgical-department categories fitted in the training
data. Full column definitions, units and time boundaries are in
[docs/INPUTS.md](docs/INPUTS.md) and
[docs/feature_dictionary.csv](docs/feature_dictionary.csv).

## Files

| File | Purpose |
|------|---------|
| `src/aki_map/temporal.py` | Source-aware 5-minute binning and causal temporal statistics |
| `src/aki_map/features.py` | Input validation, derived variables and final feature assembly |
| `src/aki_map/risk.py` | CatBoost risk ensembles, sampling and probability calibration |
| `src/aki_map/network.py` | Neural MAP-response model definitions |
| `src/aki_map/response.py` | Response-model training, curve scaling and MAP-threshold inference |
| `src/aki_map/pipeline.py` | Outer-fold evaluation and full-development external inference |
| `src/aki_map/selection.py` | Historical risk-model candidate configurations and selection rule |
| `src/aki_map/synthetic.py` | Artificial baseline and monitoring data generation |
| `src/aki_map/cli.py` | Command-line demonstration and result export |
| `configs/` | Demonstration and research settings, feature order and seven ablation sets |
| `docs/` | Input specifications, methods, feature definitions, provenance and validation |
| `tests/` | Method and release checks |
| `tools/package_release.py` | Build the source-only release ZIP |

## Installation

Use Python 3.11-3.13. The tested dependency versions are listed in
`requirements.txt`. Run from the repository root; the activation command below
uses a macOS/Linux shell.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

## Synthetic demonstration

The example generates its own input data and runs on CPU. No separate dataset
download or GPU is required.

**Quick demonstration**

```bash
aki-map-demo --config configs/smoke.json --output outputs/demo
```

Creates 120 synthetic development patients and 90 synthetic external patients,
then runs feature processing, risk and response training, and threshold inference.

Key defaults: 5 outer folds, 3 inner folds, 20 trees per risk model, 2 response
epochs, one seed per model type and 30 bootstrap replicates. The model
architecture is retained while training settings are reduced for speed.

**Research configuration**

Uses the archived training hyperparameters on freshly generated synthetic data.

```bash
aki-map-demo --config configs/research.json --patients 240 --output outputs/research-demo
```

Key defaults: 350 trees per risk model, 8 response epochs, 3 seeds per model type
and 1000 bootstrap replicates. This run takes longer than the quick demonstration.

## Outputs

Results are written to the directory specified by `--output`. It must be empty;
use a different output directory for another run.

| File | Content |
|------|---------|
| `*_synthetic_baseline.csv` | Generated patient-level baseline tables |
| `*_synthetic_observations.csv` | Generated monitoring records |
| `*_synthetic_predictions.csv` | Patient ID, elapsed time, outcome labels, AKI probabilities, MAP threshold and curve contrast |
| `synthetic_metrics.csv` | Patient-balanced AUROC, AUPRC and Brier scores with bootstrap intervals |
| `synthetic_fold_audit.json` | Outer- and inner-fold patient assignments |
| `run_manifest.json` | Configuration, Python and package versions, feature counts and release-scope flags |

The `development` and `external` prefixes distinguish the two synthetic cohorts.
A missing `target_map` can be a valid result of the flat-response-curve rule,
especially after the short demonstration training. No model files are saved.

## Reproducibility scope

The public implementation uses the final selected `dynamic_aki_map` architecture.
Core methods were extracted from the archived research code. Descriptive source
identifiers and the original recorded SHA-256 hashes are provided in
[docs/source_provenance.json](docs/source_provenance.json); identifiers are aliases
for archived source files. Detailed formulas and implementation choices are in
[docs/METHODS.md](docs/METHODS.md).

**Patient-level study data and study-trained models are not distributed.**
This includes fitted trees, neural-network weights, imputation values, category
vocabularies, calibration coefficients and response-curve scales. These
parameters are fitted afresh during each demonstration.

The portable example uses harmonised input tables and synthetic data. It does
not reconstruct the historical feature search or original patient-fold
assignment. The archived response run used Apple MPS; this example uses CPU.

Synthetic metrics demonstrate the computational workflow and must not be
reported as study performance. MAP thresholds are model-derived research
outputs; the implementation does not establish a causal treatment effect.

## Licence

This software is made available under the
[PolyForm Strict License 1.0.0](LICENSE). The full, unmodified licence text governs
permitted uses, including its specific organisational permissions. It does not
grant permission to modify the software, create new works based on it or
redistribute it. Uses outside the granted scope require separate permission
from the rights holder. This is a source-available release, not an open-source
licence.

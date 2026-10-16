# Methods implementation

## Features and encoding

The final 13 clinical inputs are age, sex, hypertension history, ASA class,
emergency status, preoperative creatinine, albumin, preoperative diuretic use,
surgical department, elapsed anaesthesia time, SBP, DBP and MAP. eGFR is derived
from age, sex and creatinine. The final ordered list contains 111 numeric
features and one categorical feature.

Preoperative albumin is the most recent available measurement before the
indexed anaesthesia start, expressed in g/L. This selection is performed
upstream when preparing the patient-level baseline table.

Temporal statistics use observed 5-min bins. Cumulative windows include `[0,t]`;
recent windows are `(t-15,t]`, `(t-30,t]` and `(t-60,t]`. SD uses `ddof=0`.
Slopes are ordinary least-squares slopes versus time in minutes with at least
two observed bins. Low-pressure fractions and deficits use observed bins as
their denominator, not imputed minutes. Forward-carried recent values expire
after 10 min; carried values do not enter observed-only summaries. Complete
definitions and numeric cutoffs are in `feature_dictionary.csv`.

Each risk model fits numeric imputation medians on its sampled training rows,
falling back to zero for a completely missing column. The risk matrix includes
missingness indicators for elapsed time and dynamic features, and one-hot
encodes surgical department from training patients. Unseen categories map to
all zeros. The final private model had 228 encoded columns; synthetic data can
have a different count because the fitted category vocabulary is different.

## Risk training and patient contribution

Any-AKI CatBoost uses Logloss, depth 5, 350 trees, learning rate 0.035 and L2
regularisation 20. Moderate-to-severe-AKI CatBoost uses depth 4, 350 trees,
learning rate 0.035 and L2 regularisation 15. There is no early stopping, class
weighting or clinical-category handling inside CatBoost; categories have
already been one-hot encoded. The CPU defaults of CatBoost 1.2.10 apply to
other settings; the tested package version is pinned.

For any AKI, exactly 12 eligible landmarks per patient are sampled uniformly,
with replacement when fewer than 12 exist. For the original severe-outcome
branch, exactly 12 landmarks are sampled with replacement while cycling among
nonempty early (<=120 min), middle (>120 to <=180 min) and late (>180 min)
bands. All patients therefore contribute 12 training rows. CatBoost seeds are
42/142/242 for any AKI and 43/143/243 for the severe-outcome branch.
Three seed predictions are averaged on the probability scale before further
processing.

The historical S1 candidate grid and selection rule are in `selection.py`.
Selection evaluated stage sampling with depth 4/350 trees and uniform sampling
with depths 4 or 5 and 150/350/600 trees. The rule requires AUROC loss <=0.005
and AUPRC >=95% of the reference, maximises AUROC, and treats candidates within
0.0005 as near ties with the documented complexity/reference tie breakers.
The demonstration uses the final selected architecture; it does not claim to
reproduce that historical search on synthetic data.

## Patient-level cross-fitting and probability transformation

The demonstration uses five stratified outer folds and three stratified inner
folds, stratifying patients by `2*any_aki+moderate_severe_aki`. All landmarks of
a patient remain together. The default outer split uses `StratifiedKFold` with
`shuffle=True` and `random_state=20261016`, set by `synthetic_split_seed` in both
supplied configurations. This seed specifies the public demonstration split
for the same patient set, labels and ordering. For historical reproduction,
supply the original study's patient-level fold assignments through
`outer_assignment`; this bypasses the seed-based default. Inner seeds are 8161
for full development and 8162-8166 for the outer training sets.

Within each outer training set, three-fold inner OOF risk predictions are
generated, using preprocessing fitted only on the corresponding inner training
patients. Let p1 be the three-seed any-AKI mean and p2 the three-seed original
severe-outcome mean. With clipping to `[1e-7,1-1e-7]`, the assisted score is

```
z = 0.5*logit(p2) + 0.5*logit(p1)
p2_final = sigmoid(a + b*(z-center)/scale)
```

`center`, `scale`, `a`, and positive `b` are fitted using patient-weighted S2
binary log loss on the training set's inner OOF predictions. The standardised
slope is bounded to `[1e-4,20]`. They are not included as study-fitted constants
in this repository. They differ from calibration intercept/slope estimated
later to evaluate model performance.

The same training-set calibrator transforms the inner OOF scores for response
training. Thus the base risk learner is OOF for those patients; the calibration
layer itself is fitted using the training set's pooled inner OOF outcomes and
is not additionally cross-fitted within those inner rows. Outer validation
patients remain excluded from both fits. This mirrors the archived procedure.
Outer validation uses risk models refitted on the complete outer training set.

After outer evaluation, risk models are refitted on the entire development
cohort; their deployment calibrator is fitted from full-development three-fold
inner OOF scores. Full-development response training uses the five-fold outer
OOF final risk predictions as anchors. External evaluation uses these full-
development models, not an average of outer-fold models. Each development
patient's reported prediction and threshold come only from its own held-out
outer fold. Fold audits are exported by the synthetic example.

## Native MAP-response network

The response input is the 111 numeric features plus two base-risk logits,
clipped to `[1e-6,1-1e-6]` before logit transformation. Training medians fill
missing numeric values. Patient-weighted means and SDs standardise them, with
SD floor 0.001. Missingness flags for all numeric inputs and a training-fitted
department one-hot code are appended. No fitted study preprocessing is shipped.

The network is `Linear -> 64 -> LayerNorm -> GELU -> Dropout(0.10) -> Linear
-> 32 -> GELU`, followed by a MAP-mean head and a six-parameter response head.
For each outcome j, it parameterises:

```
mu = 75 + 20*tanh(map_head)
offset_j = 0.75*tanh(raw_j0)
slope_j = 1.50*tanh(raw_j1)
curvature_j = 0.75*tanh(raw_j2)
d(M) = clip((M-mu)/15, -2, 2)
R_j(M) = sigmoid(logit(base_j) + offset_j + slope_j*d(M) + curvature_j*d(M)^2)
```

The response parameter head starts at zero. Up to six future-MAP-complete
landmarks per patient are sampled without replacement across the three time
bands; unused allocation is filled from remaining eligible landmarks. The
future target is the mean of the next three fully observed 5-min MAP bins.
These future observations are used as response-training targets and for
retrospective attainment, never in the current state or candidate-MAP input.

For patient i's sampled landmark count n_i, weight is `1/n_i`. Outcome-task
scales are the inverse binary entropy of the weighted training prevalence,
with entropy floor 0.001. The per-row objective is:

```
mean_j[BCEWithLogits_j / max(H(prevalence_j),0.001)]
  + 0.1*((mu-observed_future15_MAP)/15)^2
  + 0.01*mean_j(slope_j^2 + curvature_j^2)
```

Each batch loss is its weighted sum divided by batch weight sum. Optimisation
uses AdamW, learning rate 0.0005, weight decay 0.002, batch size 4096, gradient
norm clipping at 5, eight epochs and no early stopping. Three network seeds are
20260926/20260927/20260928; response-row sampling uses 20260926. The original
run used MPS; this example uses CPU. Accelerator differences can change
floating-point results.

## Curve scaling and thresholds

Candidate MAP values are integers 60-90 mm Hg inclusive. Probability curves
are averaged across the three response seeds before thresholding. One sampled
response-training landmark per patient, at most 8000 patients, is drawn with
seed 20260927. For outcome j, `s_j` is the 75th percentile of training
within-curve max-minus-min probability ranges, floored at `1e-6`.

```
N_j(M) = clip((R_j(M)-min_M R_j(M))/max(s_j,1e-6), 0, 1)
C(M) = 0.75*N_1(M) + 0.25*N_2(M)
contrast = max_M C(M)-min_M C(M)
```

If contrast is below 0.05, no threshold is returned. Otherwise the threshold
is the LOWEST candidate M satisfying `C(M)<=min(C)+0.20*contrast`. The archived
ascending-grid/first-accepted implementation is retained. The training-derived
scales are held fixed for external inference; external outcomes do not refit
them. Fold-specific predictions use that fold's own response fit and scales;
thresholds are not averaged across folds.

## Evaluation

All eligible rolling predictions are evaluated with weights `1/n_i` per
patient. AUROC, average precision (AUPRC), and Brier score use those weights.
Patient-cluster bootstrap resamples patients with replacement and carries all
their landmarks, multiplying their original per-landmark weights by the
patient's bootstrap multiplicity. Percentile 95% intervals use seed 20261007.
The research configuration uses 1000 replicates; the smoke configuration uses
30. The demonstration uses elevated artificial event frequencies so the small
cross-fitting example has enough events; they are not study prevalence.


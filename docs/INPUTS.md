# Harmonised input specification

The public training functions accept one indexed anaesthesia episode per
patient, using `build_landmarks(baseline, observations)`. The example generates
these two tables. Adapting institutional EHR exports to this schema is outside
the synthetic example; no private extraction tables or patient identifiers are
distributed.

## Baseline table

| Column | Meaning / unit / encoding |
| --- | --- |
| `patient_id` | Unique patient identifier, one episode per patient |
| `age` | Age in years |
| `sex` | Recorded sex, female=0 and male=1 |
| `Hypertension` | Preoperative history of hypertension, 0/1 |
| `asa` | Numeric ASA physical status class |
| `emor` | Harmonised emergency status, 0/1 |
| `baseline_recent` | Most recent preoperative creatinine, micromol/L |
| `albumin` | Most recent preoperative albumin, g/L |
| `drug_diur` | Preoperative diuretic use during the indexed admission, 0/1 |
| `surgery_department_detail` | Harmonised surgical department category |
| `duration_min` | Anaesthesia duration in minutes, at least 120 |
| `any_aki` | Creatinine-based postoperative AKI stages 1-3, 0/1 |
| `moderate_severe_aki` | Creatinine-based postoperative AKI stages 2-3, 0/1 |

Outcomes are used for training/evaluation and are never predictors.
`duration_min` defines the valid episode window; total duration is not a risk
feature. Creatinine-based AKI must be constructed upstream within the study's
48-h window, using the cohort-specific time anchor described in the manuscript.
The public example does not reconstruct cohort inclusion or outcome SQL.

All baseline input values must already be available before anaesthesia.
For albumin, provide the most recent available measurement before the indexed
anaesthesia start. Select this value from the source laboratory records before
creating the baseline table; if none is available, leave the field missing.
External albumin
in g/dL must be converted to g/L by multiplication by 10. Creatinine in mg/dL
must be converted to micromol/L by multiplication by 88.4. eGFR is computed
internally using CKD-EPI 2021 from age, sex and creatinine.

## Monitoring table

| Column | Meaning |
| --- | --- |
| `patient_id` | Matches the baseline table |
| `minute` | Real observation time in minutes from anaesthesia start |
| `component` | `sys` (SBP), `dia` (DBP), `map` (MAP); all in mm Hg |
| `source` | `art`, `nibp`, or `generic` |
| `value` | Numeric measured value |

Records outside the selected episode are discarded. The 5-min bin is
`ceil(minute/5)`; its right boundary must have arrived and lie inside the
episode. Within a bin and source, repeated values are aggregated by the median
and the latest real observation time is retained. Source priority is arterial,
then non-invasive, then generic. Source-specific direct MAP takes priority
over derived MAP. SBP/DBP pairs must have pulse pressure 5-180 mm Hg; derived
MAP is `(SBP+2*DBP)/3` and must be 30-200 mm Hg. Raw valid ranges are SBP
40-300, DBP 20-200 and MAP 20-250 mm Hg.

Predictions begin at 60 min and repeat at eligible 5-min boundaries. The most
recent MAP must be no more than 10 min old, using its retained real timestamp.
There is no extra future-MAP completeness filter for risk prediction.

The public final-model schema does not require HR, SpO2, urine output or
vasoactive medications. Historical candidate-set definitions are supplied for
documentation only; the executable runner fits the final 112-feature model.


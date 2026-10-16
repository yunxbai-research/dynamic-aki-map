# Public-package validation

Validated on 16 October 2026 with Python 3.13.9 and the package versions in
`requirements.txt`. This is the public-example environment, not a claim about
unrecorded historical software versions.

- Nine automated method/release tests passed. They check causal time
  boundaries, source priority and derived-MAP timestamps, exact feature order,
  patient-equal sampling, encoding, positive calibration slopes, network
  initialisation, the first-accepted threshold rule, eGFR missingness and
  release exclusions.
- The complete smoke pipeline ran with 120 artificial development patients
  (4668 rolling landmarks) and 90 artificial external patients (3954
  landmarks), producing finite probabilities and curves.
- A separate audit verified disjoint training/validation patients in every
  outer and inner split, and complete held-out coverage of development patients.
- The research-sized components were also exercised on artificial data: six
  CatBoost fits with 350 trees each and three response networks with eight
  epochs each. This component check does not represent a second clinical
  validation or the original study run.
- The final ZIP is built from an allowlist. Runtime outputs, caches, fitted
  models, installed-package metadata and the private extraction helper are
  excluded. The ZIP contains no real patient tables or study-fitted constants.

The GitHub Actions workflow is supplied for future runs on GitHub; it was not
executed on GitHub as part of this local preparation.


# Run-aware predictive-core alert confirmation

Status: **confirmed**

## Objective

Determine whether the previously confirmed rank-32 predictive core becomes a
useful warning system under a frozen 5% control-run false-alarm budget.

## Capsule

- [Frozen specification](spec.md)
- [Executable contract](contract.json)
- [Reusable policy and decision rule](implementation.py)
- [Exact collection/scoring runner](run.py)
- [Independent assessor](assess.py)
- [Behavioral tests](test.py)
- [Conclusion-bearing findings](findings.md)
- [Primary references and adaptation notes](references.md)
- Intended immutable artifact:
  `artifacts/action-dynamics/run-aware-alert-confirmation-v2-attempt-001`
- The sealed v2 execution passed all raw qualification and alert decision
  gates. The conclusion-bearing metrics and evidence hashes are retained in
  [findings.md](findings.md).

## Primary references and adaptation

Zhang et al., “Conformal anomaly detection in event sequences,” ICML 2025
([PMLR](https://proceedings.mlr.press/v267/zhang25dn.html)), motivates
calibration at the alert unit. Page, “Continuous inspection schemes,”
*Biometrika* 1954
([DOI](https://doi.org/10.1093/biomet/41.1-2.100)), supplies the resettable
cumulative-sum pattern.

Quantis does not claim the full assumptions or guarantees of either method.
The adaptation converts one-step residuals to empirical tail probabilities,
uses a negative-drift resettable statistic, and split-conformally calibrates
the maximum statistic of complete control runs. Its claim is bounded to the
fixed lab stack and declared intervention library.

## Evidence boundary

The fresh 120-pair campaign is split by a frozen action/topology-stratified
rule into 30 score-reference, 30 threshold-calibration, and 60 sealed
evaluation pairs. No fitting or policy choice observes the campaign. Passing
does not establish production paging, cross-stack transfer, unknown-event
detection, root-cause attribution, or drift-safe adaptation.

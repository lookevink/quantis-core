# SC-JEPA codebook × multi-resolution interaction v1 results

## Conclusion

Reject this exact edge SC-JEPA interaction recipe.

The frozen one-seed tracer was eligible for a scientific decision, all
protocol and restoration checks passed, and the four cells had exactly matched
capacity. The complete `codebook_multi` cell nevertheless passed neither value
lane and failed two representation-safety gates.

This rejects the frozen combination on the Quantis held-topology alert task.
It does not reject soft codebooks, multi-resolution prediction, SC-JEPA on the
authors' anomaly benchmarks, or a materially different downstream alert
design.

## Frozen identity

- Implementation commit:
  `5b6db454619aa1d6555a4cc457535c0bad86a446`
- Artifact:
  `artifacts/action-dynamics/prototype-sc-jepa-interaction-v1`
- Artifact-manifest SHA-256:
  `ce88511e935edbc2704de12ac995224162d0165a2f6284f9a2c276aa6989fea8`
- Seed: `13013`
- Schedule: 300 representation steps and 200 downstream-head steps
- Decision: `reject_sc_jepa_interaction_recipe`

The retained assessor independently rederived the event definition, role
isolation, calibrations, thresholds, representations, code usage, state
retention, factorial interactions, edge diagnostics, and final decision from
stored evidence.

## Held-topology value

| Cell | Calibrated Brier | Control FPR | Treatment detection | Median delay |
|---|---:|---:|---:|---:|
| `continuous_single` | 0.029362 | 10% | 70% | 5 |
| `continuous_multi` | 0.029369 | 10% | 70% | 5 |
| `codebook_single` | 0.033750 | 0% | 0% | none |
| `codebook_multi` | 0.033740 | 0% | 0% | none |
| `raw_low_rank` | 0.026035 | 20% | 40% | 2 |

The Brier difference-in-differences was only `0.0000174`, far below the
required `0.05 × B_cs = 0.001468`. The complete cell's Brier was also 29.6%
worse than the best reference, the raw low-rank control, instead of at least
5% better.

The detection interaction was exactly zero. Both codebook cells detected no
treatment trajectories, while both continuous cells detected 70%. The
multi-resolution target made no material difference inside either bottleneck
family. The result therefore supplies no evidence for the claimed joint
interaction; the soft codebook is the dominant failure mode in this tracer.

## Safety and edge diagnostics

- Public risks, calibrated risks, decisions, tokens, patch values, and both
  codebook probability tensors restored within `1e-6`.
- Every cell had 74,975 training parameters and 30,945 deployed inference
  parameters.
- Ten of 32 codes exceeded 0.5% marginal use and marginal perplexity was 9.43,
  but not every observed entity used multiple dominant codes.
- Candidate state-probe aggregate NRMSE was 0.9105 versus 0.3331 for matched
  entity PCA, or 2.73 times worse.
- Candidate serialized model plus sidecars was 2,266,127 bytes.
- Batch-one CPU latency was 0.716 ms mean and 1.051 ms p95 over 100
  repetitions.
- Process peak RSS was 3,248,111,616 bytes.

The capacity, restoration, edge-budget, and derived-protocol gates passed.
Code-use diversity and state retention failed.

## Interpretation

The continuous cells learned materially better alert features than the
codebook cells, but neither continuous cell met the deployment contract:
both exceeded the 5% control false-alarm ceiling. Quantization suppressed all
alerts rather than improving calibration. Because the coarse target changed
almost nothing within either bottleneck family, additional training of this
same factorial is unlikely to reveal the missing interaction.

The next preregistered target is CF-JEPA: mask-free random crops with explicit
short-, middle-, and long-horizon forward prediction and a controlled
online-versus-EMA downstream comparison.

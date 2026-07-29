# Causal-JEPA official attempt 1 restore-boundary failure

The first 1,200-step run completed scientifically but is not the accepted
ticket 018 result. Its fresh stored-array assessor found that the
prediction-only masked-completion replay differed from the original by a
maximum of `1.0728836059570312e-6`, just beyond the frozen `1e-6` restoration
bound.

The cause was an assessor-fixture mismatch in the retained runner: original
completion inference processed the full transfer array in batches of 128,
while restoration processed only the first eight rows. CPU Transformer
floating-point accumulation changed at the different batch size. Forecast
and attribution replays were exact, and candidate/coordinate completion
replays remained below the bound.

The complete bundle is retained at:

`artifacts/action-dynamics/prototype-causal-jepa-v1-attempt-1-restore-failure`

- Artifact-manifest SHA-256:
  `498f33d216f1722343efcfaa90951b3a2920d05692af6ab706b07d30c12f614a`
- Assessment SHA-256:
  `6e2bb61997d7751c7ff8e82760ac1ec96b0b507d57cdcb242c97b975b875f94a`
- Recorded decision: `reject_causal_jepa_edge_recipe`

The repair changes no model, training, selection, metric, or gate. It restores
the model over the same full transfer array and then retains the first eight
rows, making the replay batch partition identical. A new committed official
run is required; attempt 1 must not be cited as the accepted result.

# JEPA-SCORE primary-source notes

## Source

The primary source is Balestriero, Ballas, Rabbat, and LeCun,
[*Gaussian Embeddings: How JEPAs Secretly Learn Your Data
Density*](https://arxiv.org/abs/2510.05949), arXiv `2510.05949v1`,
7 October 2025.

The paper interprets a differentiable JEPA encoder as a local change of
variables. For an encoder `f` and input `x`, equation (5) defines:

```text
JEPA-SCORE(x) =
    sum_(k=1)^rank(J_f(x)) log(sigma_k(J_f(x)))
```

where `sigma_k` are singular values of the encoder Jacobian. Higher scores
represent higher learned density. An anomaly score therefore negates
JEPA-SCORE.

The paper's equation (4) is an expectation over the training transform
distribution `p_T`. Equation (5) is explicitly a one-transform Monte Carlo
estimate with `x=(mu,T)`. An identity input that lies outside the training
transform support is therefore only a projector-Jacobian heuristic, not the
paper's estimator. The Quantis screen freezes one fresh `global_a` draw from
the original telemetry view distribution. It does not average transforms.
If a future experiment uses multiple transforms, the log density must be
aggregated as `-logmeanexp(-score_T)` and the entire aggregate must be
charged to edge latency.

Appendix B gives the authors' PyTorch computation. For a sample-separable
evaluation-mode model it differentiates the batch-summed embedding, flattens
each sample's input coordinates, computes every Jacobian singular value,
clips them at `1e-6`, takes the logarithm, and sums over embedding
coordinates. The frozen Quantis implementation follows that listing,
including the clipped full singular-value vector. It does not use the
incorrect `log(sum(singular_values))` paraphrase found in some secondary
summaries.

## Applicability boundary

The density result is exact at the paper's JEPA-loss optimum under its
Gaussian-embedding argument. A finite telemetry model is not known to meet
that premise. Quantis therefore:

- scores the projector output on which complete LeJEPA actually applied
  exact SIGReg, rather than its public entity tokens;
- uses one fixed transform from the complete-LeJEPA training-view support,
  making it the paper's declared single-transform Monte Carlo estimator;
- retains the complete-LeJEPA, SIGReg-only, and invariance-only frozen cells
  to expose dependence on the training objective;
- reports Gaussianity/rank diagnostics as mechanism evidence;
- treats alert utility as empirical, not as proof of a calibrated density;
  and
- does not call an approximation "JEPA-SCORE." Any Hutchinson, randomized
  SVD, finite-difference, or learned score surrogate would be a separate
  method and contract.

The source demonstrates image encoders, not telemetry alerts, topology
transfer, or an edge runtime. Those are new Quantis evaluation questions.

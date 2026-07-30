# Primary references and adaptation notes

- Zhang et al., “Conformal anomaly detection in event sequences,” ICML 2025
  ([PMLR](https://proceedings.mlr.press/v267/zhang25dn.html)).
- Page, “Continuous inspection schemes,” *Biometrika* 1954
  ([DOI](https://doi.org/10.1093/biomet/41.1-2.100)).

The Quantis adaptation calibrates a complete control-run maximum rather than
claiming independent window tests. One-step residuals become empirical
upper-tail probabilities, and a Page-style resettable cumulative statistic
subtracts `log(4)` so routine probabilities supply negative drift. This
experiment makes neither a continuous-time rescaling claim nor a general
e-value guarantee.

# Mean-preserving residual mixture JEPA freeze v1

## Conclusion

**The MPRM-JEPA tracer is frozen and ready to begin fit-role model fitting. No
selection evidence has been collected or scored.**

The executable contract pins the candidate, controls, numeric
canonicalization, fresh selection identity, statistical test, edge runtime,
zero-retry policy, and the two explicit supersessions of the original
multi-hypothesis scoring contract.

The retained fit campaign and its protocol-aware validity audit are present
and match the frozen hashes. The public campaign qualification seam rejects
incomplete coverage, binding drift, failed action realization or recovery,
and any attestation that does not repeat all campaign bindings and
content-address every prepared manifest.

## Execution boundary

Execution is ordered:

1. fit and freeze all models using only the retained v1 fit role;
2. restore every model and require exact prediction parity;
3. bind the resulting model-freeze manifest into a fresh selection campaign;
4. collect all 90 pairs / 180 captures with no retries;
5. qualify the complete campaign before loading any model; and
6. score once and reproduce the decision with the independent assessor.

An operational failure leaves the attempt immutable and produces no model
claim. Passing selection authorizes a separate evaluation proposal, not
production paging.

## Entry points

```bash
PYTHONPATH=src:lab/action_dynamics .venv/bin/python \
  lab/action_dynamics/run_mprm_jepa.py preflight

PYTHONPATH=src:lab/action_dynamics .venv/bin/python \
  lab/action_dynamics/run_mprm_jepa.py fit-and-freeze
```

Selection preparation is intentionally impossible until the model-freeze
manifest exists. Subsequent phases use `run_mprm_selection.py` with the
`prepare`, `collect`, `qualify`, and `score` commands in that order.

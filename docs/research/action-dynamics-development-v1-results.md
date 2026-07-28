# Action-conditioned graph dynamics development v1 result

## Result

The fresh 120-pair development corpus qualified, but the preregistered graph
model did not.

The bounded decision is:

`publish_bounded_negative_result`

This is not evidence for a graph world model. It is also not evidence that the
telemetry corpus lacks predictable action-conditioned dynamics: the frozen
dense action-conditioned baseline was stable and substantially better than
persistence. The failure is concentrated in the strict local graph
factorization and its free-running rollout.

## Immutable inputs

- Preregistration commit: `f4adee6`
- Development protocol SHA-256:
  `d11e043521a7ca355b48faf58ca46b1a82d1b6f12a08c4e2175b9f14cf6ad1dc`
- Execution plan SHA-256:
  `9be323e8a07630e715ebe0927982e1ba93a087f0dfa778653650ff93d9a0209e`
- Qualified corpus SHA-256:
  `df03af282f48591216251f22f934b97e1555147df9ed6f6c6d1f7684f9644a26`
- Training query declaration SHA-256:
  `a0a22fbedd6e9ed907ac399e1cd42b8c66c0f43c8af5e504ddac9f2c7eb946b0`
- Training artifact-manifest SHA-256:
  `cf719444815bd7bca6b9f66de0f4d0fc58a242f6b2c5d3e0eb523bc92131f948`

Raw evidence is preserved under the ignored local artifact directories
`artifacts/action-dynamics/development-v1` and
`artifacts/action-dynamics/development-training-v1`.

## Corpus qualification

All 24 frozen collection gates passed:

- 120/120 matched pairs and 240/240 captures;
- 90 training pairs and 30 whole-pair development-validation pairs;
- 24 pairs per action family;
- 40 pairs per worker topology;
- six training and two validation pairs in every action-by-topology cell;
- zero missing captures and zero automatic retries;
- zero failed effect or recovery pairs;
- 100% eligible event-to-trace linkage;
- 99.966% complete checkout-path coverage;
- zero cross-case trace references;
- zero placebo false positives;
- maximum primary recovery ratio `0.173` against the frozen `0.30` limit; and
- maximum enqueue mechanistic recovery ratio `0.0824` against `0.30`.

The corpus therefore passed the condition required to start model fitting.
The instrumentation result is strong and separate from the model result.

## Frozen model matrix

All transforms were fit on the 90 training pairs only. The 30 validation pairs
were used only for the frozen 10-step rollout and 60 treatment/control
attribution queries.

| Model | Action-overlap normalized MSE | All-state normalized MSE |
|---|---:|---:|
| Action-conditioned dense VARX | 0.202 | 0.0801 |
| Persistence | 1.726 | 0.601 |
| Action-conditioned graph VARX | 15.722 | 8.363 |
| Action-agnostic graph VARX | 6.68e12 | 9.45e11 |

The dense action-conditioned baseline improved `88.3%` relative to
persistence on active-action rollout. It also improved `69.7%` relative to
persistence on paired downstream intervention-effect error.

Those are meaningful development diagnostics: the training split contains
learnable action-conditioned transition signal, and a linear model can recover
it on held-out pairs. They do not pass the graph model's preregistered gates
and were not used for the frozen attribution query.

The graph model behaved differently by action:

| Action | Graph action MSE | Persistence MSE | Relative graph improvement |
|---|---:|---:|---:|
| API rejection | 1.680 | 2.187 | +23.2% |
| Redis enqueue delay | 0.690 | 0.866 | +20.4% |
| Redis dequeue delay | 1.580 | 1.101 | -43.6% |
| Worker pause | 37.118 | 1.046 | -3447.8% |
| PostgreSQL lock | 54.021 | 3.553 | -1420.4% |

The aggregate graph failure is therefore not uniform representation collapse.
The local model captures API rejection and enqueue delay better than
persistence, but becomes badly wrong for worker and PostgreSQL interventions.

## Frozen gates

| Gate | Threshold | Observed | Result |
|---|---:|---:|---|
| Graph action vs graph action-agnostic | >=10% | ~100% | PASS |
| Graph action vs persistence | >=10% | -810.9% | FAIL |
| Graph vs dense paired downstream effect | >=5% | -18477.2% | FAIL |
| Action-and-target hit@1 | >=70% | 56.7% | FAIL |
| No-action specificity | >=90% | 96.7% | PASS |

The first pass is not positive evidence. The action-agnostic graph VARX
exploded, so beating it by nearly 100% only says that explicit actions reduce
an already unstable rollout. The action-conditioned graph model still loses
badly to persistence and the stable dense baseline.

## Attribution

The frozen graph-model ranker correctly recognized 29/30 matched no-action
controls. Treatment attribution was:

| True action | Correct / 6 | Hit@1 |
|---|---:|---:|
| Redis enqueue delay | 6 | 100% |
| Redis dequeue delay | 6 | 100% |
| API rejection | 5 | 83.3% |
| Worker pause | 0 | 0% |
| PostgreSQL lock | 0 | 0% |

Every worker-pause and PostgreSQL-lock miss was ranked as no action. Exact
severity-duration variant hit@1 was `20%`; family hit@3 remained `56.7%`
because the missing families did not appear in the first three candidates.

This is consistent with the rollout result: the graph model is useful for
three intervention mechanisms but does not carry dependable state transitions
for worker and database actions.

## Post-hoc stability diagnosis

This diagnosis was computed after the frozen gates and did not change them.

The spectral radius of each fitted autonomous state transition was:

| Model | Spectral radius | Eigenvalues with magnitude > 1 |
|---|---:|---:|
| Dense action-conditioned VARX | 0.889 | 0 |
| Graph action-conditioned VARX | 1.616 | 1 |
| Graph action-agnostic VARX | 7.947 | 3 |

The graph transitions are dynamically unstable under autoregressive rollout;
the dense transition is contractive. That explains both the huge
action-agnostic error and the action-conditioned graph model's rapid
10-step error amplification.

The most likely interpretation is not “the representation cannot generalize”
or “similar log events swamped the corpus.” The stronger evidence is:

1. the balanced validation corpus exposes all actions and topologies;
2. the dense action-conditioned linear model generalizes well;
3. the strict incoming-neighbor graph factorization removes cross-entity
   information that the dense model uses; and
4. unconstrained local autoregressive fits create an unstable global
   transition.

The structured log/trace counts were present, but this run did not include a
metrics-only ablation. It therefore cannot claim that event counts helped or
hurt.

## Scientific meaning for JEPA and the world-model claim

This iteration intentionally required the supervised transition baselines to
earn the right to add a latent JEPA objective. They did not earn that right for
the strict graph model.

That follows the evaluation logic of action-conditioned predictive models:
future state under an action must be accurate in free-running rollout and
useful for downstream action inference, not merely low in an embedding loss.
V-JEPA 2 similarly separates representation learning from an
action-conditioned predictor and evaluates downstream interaction
([Assran et al., 2025](https://arxiv.org/abs/2506.09985)).

There is no learned latent compression in this result. The operational state
has `7 entities × 31 features = 217` scalar slots. The fitted models contain:

- 16,182 graph action-conditioned parameters;
- 13,361 graph action-agnostic parameters; and
- 67,704 dense action-conditioned parameters.

The result is a linear state-space baseline study, not a JEPA model and not a
compressed world state.

## Next development step

The highest-probability next experiment is a stable hybrid transition, using
this already-open development corpus:

1. preserve the dense model's global low-rank channel;
2. add typed local graph messages as a residual, rather than making them the
   only permitted inputs;
3. constrain or parameterize the transition so its free-running spectral
   radius is at most one;
4. train an explicit paired treatment-minus-control effect head for target,
   downstream, and recovery trajectories;
5. run both dense and stabilized-graph attribution;
6. add a frozen metrics-only versus metrics-plus-structured-events ablation;
   and
7. require one-step and 10-step stability before considering a masked JEPA
   auxiliary.

If that development model beats persistence and the dense baseline while
meeting attribution gates, freeze it before collecting the planned fresh
60-pair confirmation corpus. Only a successful sealed confirmation should
support the phrase **constrained action-conditioned world model for the
Quantis lab stack**.


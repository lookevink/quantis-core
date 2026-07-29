# Retrieval-JEPA for telemetry evidence: primary-source notes

Research date: 2026-07-28.

This note uses only primary sources: the authors' papers and official
implementations. “Retrieval-JEPA” is not the name of a standard published
objective in those sources. It is a proposed operational role in which a JEPA
learns the query and evidence representations, while nearest-neighbor retrieval
and abstention are separately specified downstream.

## Bottom line

The most materially distinct recipe to test next is an **episode-predictive
retrieval JEPA**, not another LeJEPA view-invariance run:

1. Encode only telemetry available at the alert/query time.
2. Predict the latent representation of one or more withheld, contiguous
   episode-evidence blocks with a V-JEPA-style context encoder, EMA target
   encoder, positional predictor, stop-gradient target, and mean `L1` loss.
3. At assessment and inference, normalize the predicted query vector and
   precomputed evidence-bank vectors, then rank the bank by cosine similarity.
4. Fit one fixed abstention threshold on an independent calibration role. Use
   the exact high-probability selective-risk construction from Geifman and
   El-Yaniv only if there are enough independent calibration episodes; otherwise
   report the risk-coverage curve without claiming a finite-sample guarantee.
5. Compare against raw telemetry, deterministic PCA, a capacity-matched
   label-supervised retriever, and a capacity-matched CPC/InfoNCE retriever.

This is materially different from the completed LeJEPA test. It deploys the
JEPA predictor as the retrieval query map and asks whether a current episode
can retrieve useful prior evidence. The value claim is retrieval and
investigation utility, not direct trajectory error.

## What the primary JEPA sources actually specify

### I-JEPA: masked context-to-target latent regression

[I-JEPA](https://ar5iv.labs.arxiv.org/html/2301.08243) samples target blocks
from the output of a target encoder, removes target overlap from a large
context block, and predicts each target token from the context tokens plus
target-position mask tokens. For target blocks `B_i`, its published loss is

```text
L_I-JEPA =
  (1 / M) Σ_i Σ_(j in B_i) || g_phi(f_theta(context), mask_j)
                              - f_bar_theta(full_input)_j ||_2^2.
```

The context encoder and predictor receive gradients. The target encoder is an
EMA of the context encoder. The paper reports that sufficiently large target
blocks and a spatially distributed informative context are central to the
result, and that target blocks must be selected from target-encoder outputs
rather than by first masking the target input. See
[I-JEPA §3](https://ar5iv.labs.arxiv.org/html/2301.08243#S3) and the
[official implementation](https://github.com/facebookresearch/ijepa).

**Telemetry translation:** a query prefix should predict representations of
large contiguous time/topology evidence blocks, not a few adjacent points.
Overlapping windows make an easy local-copy task; removing query/target overlap
and withholding complete blocks are necessary shortcut controls.

### V-JEPA: the closest source recipe for episode retrieval

[V-JEPA](https://ar5iv.labs.arxiv.org/html/2404.08471) predicts masked
spatio-temporal target tokens from visible tokens. Let `M` be the masked token
indices, `z_N = E_theta(x_N)` the visible context tokens,
`s_L = E_bar_theta(x_L)` the full target-encoder tokens, and
`s_hat_M = P_phi(z_N, m_M)` the positional predictions. Its loss is

```text
L_V-JEPA = (1 / |M|) Σ_(k in M) || s_hat_k - stopgrad(s_k) ||_1.
```

The target encoder is updated by EMA. The paper uses stop-gradient, reports
`L1` as more stable than the alternative it tested, and samples both short-
and long-range multi-block masks. Its target blocks extend over the full
temporal dimension specifically to reduce leakage from video redundancy. See
[V-JEPA §3](https://ar5iv.labs.arxiv.org/html/2404.08471#S3),
[extended method §9](https://ar5iv.labs.arxiv.org/html/2404.08471#S9), and the
[official repository](https://github.com/facebookresearch/jepa).

The source-scale optimization recipe uses AdamW, a target momentum starting at
`0.998` and increasing toward `1.0`, and two masks whose union hides about
`90%` of tokens. Those constants were established at video scale and should
not be copied silently into a small telemetry tracer. A telemetry contract
must explicitly freeze its smaller architecture, mask geometry, EMA schedule,
and optimizer rather than calling them exact V-JEPA defaults.

**Recommended source family:** V-JEPA is the best match because the telemetry
objects are temporal episodes and the retrieval query is naturally a
prediction of withheld evidence. I-JEPA's squared `L2` objective should be a
single loss ablation only if the primary `L1` recipe fails; it should not be a
selection-time sweep.

### LeJEPA: relevant as a control, not the next primary treatment

[LeJEPA](https://arxiv.org/abs/2511.08544) removes the EMA teacher, predictor,
and stop-gradient machinery. It combines multi-view alignment with Sketched
Isotropic Gaussian Regularization (SIGReg):

```text
L_LeJEPA = lambda * mean_v SIGReg(Z_v)
         + (1 - lambda) * L_invariance.
```

Its official code and exact finite-quadrature details are already recorded in
[the repository's SIGReg source note](lejepa-sigreg-primary-source-notes.md)
and the
[official LeJEPA repository](https://github.com/galilai-group/lejepa).

LeJEPA does not define a retrieval-specific neighborhood loss, evidence bank,
or calibrated reject rule. An isotropic marginal distribution does not by
itself establish that nearest neighbors are operationally relevant. Repeating
the same complete LeJEPA objective with a new metric would test a new
downstream role, but not a materially new representation objective. The
already-fitted complete-LeJEPA representation can therefore be retained as an
additional frozen control; the primary treatment should restore directed
context-to-target prediction.

## A materially distinct frozen candidate

### Episode and role definitions

Define one independent episode as a complete run or another genuinely
independent operational trial, not an overlapping window. Keep matched arms
and every derived window atomic across fit, calibration, evidence-bank, and
transfer roles.

For query episode `i`:

- `C_i` is the telemetry available at the declared alert/query timestamp.
- `T_ir`, `r = 1..R`, are predeclared contiguous evidence blocks withheld from
  `C_i`. They may include later observations for self-supervised pretraining,
  but those later observations can never enter the deployed query.
- `R_i` is the model-independent set of bank episodes judged relevant by the
  frozen evidence truth. These relevance labels are allowed in assessment and
  in the explicitly supervised control; they are forbidden from candidate
  representation fitting.

The evidence truth should describe investigation usefulness, such as shared
failure mechanism and useful root-cause evidence. A schedule, action, pair,
topology, trajectory, or file identifier is not by itself evidence relevance.

### Encoder and latent-prediction objective

Use one context encoder `E_theta`, an identically structured EMA target
encoder `E_bar`, and a narrow positional predictor `P_phi`.

```text
c_i = E_theta(C_i)
t_irj = E_bar(full_episode_i)_(r,j)
t_hat_irj = P_phi(c_i, target_position_(r,j))

L_episode-JEPA =
  mean_(i,r,j) | t_hat_irj - stopgrad(t_irj) |.
```

The fit batch must contain one anchor per independent episode or matched pair.
All masks from one episode remain multiple tasks for that sample, not
additional independent samples. Neither action labels nor relevance labels are
needed; like I-JEPA and V-JEPA, the candidate uses no negative samples.

The target blocks should include:

- one short evidence interval and one long evidence interval;
- at least one connected subsystem/topology block rather than isolated scalar
  tokens; and
- no token also visible in the query context.

Numeric jitter, channel/entity permutation, action truth, incident labels,
pair IDs, future controls, and cross-trajectory splicing should be forbidden.
Absolute time, entity, relation, ownership, and target-position identities
should be preserved.

### Concrete edge-tracer translation

The following is a recommended small telemetry translation, not a claim that
the source papers used these dimensions:

- form one `30 × 7` episode from the existing 20-point observed context and
  10-point evidence suffix;
- expose only the first `20 × 7` owned telemetry tokens to the context encoder;
- use the last `10 × 7` tokens as the long target and the last `5 × 7` tokens
  as the nested short target;
- use the repository's declared ownership mask and graph-relation biases;
- context and EMA target encoders: width 64, two pre-normalized transformer
  blocks, four heads, feed-forward width 128, GELU, zero dropout;
- positional predictor: width 64, two pre-normalized transformer blocks, four
  heads, feed-forward width 128, with learned mask tokens plus absolute time,
  entity, kind, relation, and view-presence embeddings;
- public query and bank dimension: 64 after a frozen mean over the long target
  tokens and `L2` normalization; and
- candidate loss: equal mean over short- and long-target tokenwise `L1`
  losses, with no reconstruction, negative, action, relevance, or SIGReg term.

The target encoder should process the complete 30-point episode during
training and offline bank construction; only its selected suffix tokens become
targets or evidence vectors. At inference the context encoder and predictor
receive the 20-point prefix and fixed suffix-position mask tokens. This keeps
future observations entirely out of the online query while making the query
and bank occupy the same learned target space.

### Query and bank vectors

After fitting, freeze all weights. For an alert-time context:

```text
q_i_raw = Pool(P_phi(E_theta(C_i), frozen_evidence_positions))
q_i = q_i_raw / ||q_i_raw||_2.
```

For a bank episode:

```text
e_j_raw = Pool(E_bar(evidence_slice_j))
e_j = e_j_raw / ||e_j_raw||_2.
```

Rank bank items by

```text
s_ij = q_i^T e_j
TopK(i) = K indices with largest s_ij.
```

The pooling operation, evidence positions, encoder used for the bank, vector
dimension, normalization epsilon, `K`, and deterministic tie-breaking must be
frozen. The predictor and context encoder are both part of the deployed query
path. The EMA target encoder can remain an offline index builder, but the
artifact must preserve it because the query and bank otherwise occupy
different learned maps.

This use of normalized dot products follows the official
[DINO weighted k-NN evaluator](https://github.com/facebookresearch/dino/blob/main/eval_knn.py#L130-L180),
which computes top neighbors from feature dot products. DINO's optional
class-vote score is

```text
vote_c(q) = Σ_(j in TopK(q)) exp(s_qj / temperature) * 1[label_j = c],
```

with default `temperature = 0.07`. That class vote is not automatically an
evidence-retrieval score. If it is used, the temperature and evidence label
semantics must be preregistered, and the same procedure must be applied to
every representation.

## The required non-JEPA metric-learning control

[Contrastive Predictive Coding](https://ar5iv.labs.arxiv.org/html/1807.03748)
is a natural capacity-matched non-JEPA control. CPC encodes observations
`z_t = g_enc(x_t)`, summarizes context `c_t = g_ar(z_<=t)`, and scores a future
target with a bilinear density-ratio model

```text
f_k(x_(t+k), c_t) = exp(z_(t+k)^T W_k c_t).
```

With one positive and `N-1` negatives sampled from the target marginal, its
InfoNCE loss is

```text
L_InfoNCE =
  -E log [
    f_k(x_positive, c_t)
    / Σ_(x_j in candidate_set) f_k(x_j, c_t)
  ].
```

Unlike JEPA latent regression, CPC requires negative samples. For telemetry:

- negatives must come from different independent pairs or episodes;
- overlapping windows, the other arm of a matched pair, and near-duplicate
  anchors cannot be counted as independent negatives;
- different episodes with the same evidence mechanism may be false negatives;
  and
- sampling only across different topologies can teach topology identity
  rather than incident evidence.

The control should share the candidate's context encoder, predictor capacity,
embedding width, anchor schedule, and retrieval evaluator. Its positives are
same-episode context/withheld-target pairs; all eligible other episode targets
in the pair-blocked batch are negatives. It must not use incident or action
labels, or it becomes the supervised control.

## Required label-supervised retrieval control

A strong label-bearing control should use the same normalized query and bank
vectors and the same allowed evidence truth. For query `i`, relevant bank set
`R_i`, eligible bank `A_i`, and fixed temperature `tau`, use the multi-positive
retrieval loss

```text
L_supervised =
  -mean_i log [
    Σ_(j in R_i) exp(s_ij / tau)
    / Σ_(j in A_i) exp(s_ij / tau)
  ].
```

The control must exclude the query's own trajectory, overlapping windows,
matched arm, and any other near-duplicate forbidden by the bank contract. It
should match the candidate's deployed encoder/predictor width and optimization
budget. This control answers whether the architecture and corpus support the
retrieval problem when supplied the relevance signal; it is not a lower bound
the self-supervised JEPA is entitled to beat merely by lowering pretraining
loss.

## Retrieval metrics and evidence utility

For a fixed relevant set `R_i`, report at least:

```text
hit@K_i       = 1[R_i intersects TopK(i)]
precision@K_i = |R_i intersects TopK(i)| / K
RR_i          = 1 / min_(j in R_i) rank_i(j)
```

Also report pair-balanced means, per-pair wins against each control, and the
complete rank of the first relevant episode. If relevance is graded, freeze
the gain values before fitting and report NDCG; do not infer graded gains from
the candidate's own distances.

Investigation utility should be a separate stored truth function
`u(query, retrieved_item)` or `u(query, TopK)`, recomputed by a pure assessor.
It may credit evidence such as the correct affected entity, useful trace/log
support, or a relevant counterfactual episode. It may not reuse downstream
trajectory MSE as the value claim. Model loss, latent rank, Gaussianity, and
raw similarity are diagnostics, not promotion gates.

Use exact retrieval for the tracer so approximate-index recall cannot confound
the scientific result. Approximate or compressed search is a later deployment
optimization.

## Calibrated abstention

### Selective-risk definition

Let the fixed retrieval function return `TopK(i)` and define binary retrieval
error

```text
ell_i = 1[R_i does not intersect TopK(i)].
```

Freeze a confidence-rate function that uses no truth label at inference. A
simple candidate is the cosine margin

```text
kappa_i = s_i,(1) - s_i,(2),
```

where `(1)` and `(2)` are the first and second ranked bank items with
deterministic ties. Accept when

```text
g_theta(i) = 1[kappa_i >= theta].
```

Coverage and selective risk are

```text
coverage(theta) = E[g_theta(i)]
risk(theta) =
  E[ell_i * g_theta(i)] / E[g_theta(i)].
```

These are the definitions used by
[Selective Classification for Deep Neural Networks](https://ar5iv.labs.arxiv.org/html/1705.08500#S2)
and
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf).
The confidence score need only rank examples; it need not be interpreted as a
probability.

### Exact high-probability thresholding

Geifman and El-Yaniv's Selection with Guaranteed Risk (SGR) procedure sorts an
independent calibration set by confidence and binary-searches a threshold. At
each of `k = ceil(log2(m))` tested thresholds it computes the empirical error
among accepted examples and a one-sided binomial upper bound `b` satisfying

```text
Σ_(j=0)^(m_accept * empirical_error)
  choose(m_accept, j) b^j (1-b)^(m_accept-j)
  = delta / k.
```

It accepts a threshold only when `b < target_risk`. Under the paper's i.i.d.
assumption, the final selective risk exceeds the declared target with
probability less than `delta`. See
[SGR Algorithm 1 and Theorem 3.2](https://ar5iv.labs.arxiv.org/html/1705.08500#S3).

The unit supplied to SGR must be an independent episode or pair-level
aggregate. Feeding thousands of overlapping windows to the binomial bound
would create fictitious sample size and invalidate the guarantee. The retriever,
bank, representation, `K`, confidence definition, and all other hyperparameters
must be fixed before calibration.

There is a hard sample-size consequence. With zero accepted errors, the bound
reduces to

```text
b = 1 - (delta / k)^(1 / m_accept).
```

For example, with only 40 independent calibration units,
`k = ceil(log2(40)) = 6`, `delta = 0.05`, and target risk `0.10`, even zero
errors cannot certify the target: more than 45 accepted independent units
would be required. A 20% risk target needs more than 21.5 zero-error accepted
units. The contract must perform this feasibility calculation before fitting
and must not relabel an empirical threshold as guaranteed calibration.

### Why generic conformal risk control is not the default reject rule

[Conformal Risk Control](https://ar5iv.labs.arxiv.org/html/2208.02814) chooses

```text
lambda_hat = inf {
  lambda :
  [n / (n + 1)] * R_hat_n(lambda) + B / (n + 1) <= alpha
}
```

for an exchangeable collection of bounded loss functions that are
non-increasing in the conservativeness parameter `lambda`. It then guarantees
`E[L_(n+1)(lambda_hat)] <= alpha`.

That construction is directly useful for a **retrieval set** that grows
monotonically with `lambda`, with miss loss
`1[R_i intersects C_lambda(i) is empty]`. It is not automatically a guarantee
on the conditional error among non-abstained queries: selective risk contains
a changing coverage denominator and need not be monotone in the confidence
threshold. For the reject option, SGR is the more direct primary-source
construction. Conformal risk control can be an additional set-retrieval
analysis only if its exchangeability, bounded-loss, and monotonicity
conditions are satisfied.

## Leakage and invalid-result checklist

The following conditions invalidate a retrieval-JEPA result:

1. **Query future leakage:** the alert-time query contains any withheld
   evidence, future observation, future control, action outcome, or summary
   computed from them.
2. **Identifier leakage:** pair, trajectory, topology, schedule, experiment,
   file, timestamp-origin, incident, action, or relevance IDs reach the
   encoder or normalization path.
3. **Bank self-retrieval:** the bank contains the query trajectory, its
   overlapping windows, its matched arm, or a deterministic duplicate.
4. **Split leakage:** windows from one independent episode appear in multiple
   fit/calibration/bank/transfer roles.
5. **Normalization leakage:** centers, scales, PCA components, quantizers, or
   index transformations are fit using calibration or transfer queries.
6. **Target shortcut:** target masks leave near-identical adjacent tokens or
   copied aggregate fields visible in the context.
7. **Label leakage through mining:** “hard negatives,” mask schedules, or
   retrieval-bank pruning use action/relevance labels for the self-supervised
   candidate.
8. **False-negative bias:** CPC treats another episode with the same mechanism
   as a negative without recording the collision.
9. **Calibration reuse:** the same role chooses the representation,
   checkpoint, `K`, similarity, confidence score, threshold, and reports final
   risk.
10. **Dependent confidence claims:** overlapping query windows are counted as
    independent calibration or transfer evidence.
11. **Topology shortcut:** success is driven by retrieving the same stack
    identity rather than the same evidence mechanism; topology-transfer
    evaluation and topology-only probes must expose this.
12. **Mutable bank:** candidate and controls search different item sets or an
    evidence bank is updated after seeing transfer queries.

## Matched controls

Every representation must search the exact same immutable evidence bank under
the same query time, exclusion rules, `K`, similarity, ties, relevance truth,
confidence score, calibration procedure, and pair-balanced assessor.

1. **Raw telemetry:** fit-only standardization, frozen temporal/topology
   pooling, normalized query/evidence vectors, and exact cosine retrieval.
2. **Deterministic PCA:** fit on the raw control vectors only, deterministic
   signs, same deployed dimension as the candidate, then normalized cosine
   retrieval.
3. **Supervised retriever:** same encoder/predictor capacity and optimization
   budget, trained with the frozen multi-positive relevance objective.
4. **CPC/InfoNCE:** same capacity and context/target anchors, with only
   pair-independent in-batch negatives and no labels.
5. **Masked reconstruction control:** same encoder and masks, a training-only
   decoder reconstructing owned telemetry; discard the decoder and use the
   same retrieval path.
6. **Previously fitted complete LeJEPA:** optional frozen diagnostic control;
   do not tune or retrain it using retrieval results.
7. **Untrained encoder:** same initialized architecture, useful for exposing
   topology or preprocessing shortcuts.

Raw and PCA controls need a declared common-space map because query prefixes
and evidence slices are not automatically commensurate. A learned map makes
the raw control supervised and should be named separately.

## Edge feasibility

The candidate's deployed online path is:

```text
context preprocessing
  -> context encoder
  -> positional predictor
  -> pooling and L2 normalization
  -> exact/approximate top-K search
  -> fixed confidence threshold
```

Bank embeddings, evidence metadata, and any approximate index are built
offline. For a small tracer bank, exact search is preferable and cheap: a bank
of 10,000 width-64 float32 vectors is about 2.56 MB and requires about 640,000
dot-product multiply-accumulates per query. At this scale the encoder and
predictor are likely to dominate latency.

At larger scale, [FAISS](https://github.com/facebookresearch/faiss) implements
exact and approximate dense-vector search, including compressed product-
quantized indexes. Its source paper defines exact search as top-`K` by `L2`
distance and shows the dot-product decomposition used for efficient search;
it also supports selecting the largest cosine similarities. See
[Billion-scale similarity search with GPUs §2](https://ar5iv.labs.arxiv.org/html/1702.08734#S2).

Quantization, ANN pruning, and index updates must be evaluated after scientific
value is established. A promotion artifact should separately report:

- encoder-plus-predictor parameters and serialized bytes;
- bank item count, dimension, dtype, and bytes;
- batch-one latency for preprocessing, encoding, prediction, and search;
- peak memory;
- exact-versus-ANN recall@K;
- float-versus-quantized retrieval and abstention drift; and
- restoration identity for query vectors, rankings, and threshold decisions.

## Recommended contract decision

Freeze one primary treatment:

> A V-JEPA-style episode-predictive encoder learns, without negatives or
> labels, to predict withheld contiguous telemetry evidence from the
> alert-time context. Its predicted target-space vector retrieves independent
> prior evidence episodes by exact normalized cosine search. A separately
> calibrated confidence threshold abstains when the first-versus-second
> neighbor margin is too small.

The treatment should advance only if, on held-out topology-transfer queries,
it:

- passes every no-leakage, identity, finiteness, restoration, and bank-exclusion
  check;
- improves fixed retrieval and investigation-utility metrics over raw,
  deterministic PCA, masked reconstruction, and CPC/InfoNCE;
- is competitive with the matched label-supervised retriever under a
  preregistered tolerance;
- has a better risk-coverage profile than every non-supervised control;
- meets the frozen coverage and selective-risk requirement at the
  selection-chosen threshold; and
- wins across independent pairs rather than only in the aggregate.

If there are too few independent calibration episodes for the requested SGR
bound, the tracer may still reach a bounded scientific conclusion from
held-out empirical risk-coverage evidence, but it cannot claim calibrated
production abstention. More independent episodes—not more overlapping
windows—are then the required prerequisite.

## Primary sources

- Assran et al.,
  [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://ar5iv.labs.arxiv.org/html/2301.08243);
  [official I-JEPA code](https://github.com/facebookresearch/ijepa).
- Bardes et al.,
  [Revisiting Feature Prediction for Learning Visual Representations from Video](https://ar5iv.labs.arxiv.org/html/2404.08471);
  [official V-JEPA code](https://github.com/facebookresearch/jepa).
- Balestriero and LeCun,
  [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544);
  [official LeJEPA code](https://github.com/galilai-group/lejepa).
- van den Oord, Li, and Vinyals,
  [Representation Learning with Contrastive Predictive Coding](https://ar5iv.labs.arxiv.org/html/1807.03748).
- Caron et al.,
  [official DINO weighted k-NN evaluator](https://github.com/facebookresearch/dino/blob/main/eval_knn.py#L130-L180);
  [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).
- Geifman and El-Yaniv,
  [Selective Classification for Deep Neural Networks](https://ar5iv.labs.arxiv.org/html/1705.08500).
- Geifman and El-Yaniv,
  [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf).
- Angelopoulos et al.,
  [Conformal Risk Control](https://ar5iv.labs.arxiv.org/html/2208.02814).
- Johnson, Douze, and Jégou,
  [Billion-scale similarity search with GPUs](https://ar5iv.labs.arxiv.org/html/1702.08734);
  [official FAISS code](https://github.com/facebookresearch/faiss).

# Action-dynamics lab smoke v1 results

## Decision

Do not run the v1 instrumentation pilot. Repair and preregister a v2 smoke.

The first real six-pair smoke stopped at its preregistered gate. Ten of twelve
data-quality gates passed:

- 12/12 captures were complete and hash-bound;
- treatment start/stop and control command coverage passed;
- treatment/control schedules and topologies matched;
- observation truth exclusion passed;
- eligible event trace linkage was 100%;
- complete six-span trace paths were 100%;
- recovery, cross-case trace isolation, and six-lane isolation passed.

Two gates failed:

1. A `0.34` two-worker pause did not reduce aggregate `worker_rate`. The
   remaining worker had enough capacity under the frozen light-load schedule,
   so partial worker targeting is not identifiable with the current aggregate
   observation.
2. The placebo calculation compared a control active interval with a shorter
   prior control interval. Workload variation produced one false positive.
   The preregistered prose required an equally long interval of the paired
   treatment-minus-control delta.

The evidence is retained at
`artifacts/action-dynamics/lab-smoke-v1/data-quality.json`. No capture was
rerun and the v1 pilot was never opened.

## Versioned repair

The v2 protocol:

- uses full worker pause until per-worker observations can identify partial
  targeting;
- implements the specified paired, equal-duration placebo;
- verifies the exact parent-linked six-span path and uses admitted requests
  as the completion denominator;
- requires all 27 metric features at every unique logical window;
- records cleanup state and realized paused-worker identities; and
- binds each wrapper to protocol, generated-plan, graph, observation-schema,
  image, and build identities.

This remains instrumentation qualification only. It is not model or
world-model evidence.

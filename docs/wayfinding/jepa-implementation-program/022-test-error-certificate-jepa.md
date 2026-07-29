---
status: complete
label: wayfinder:ticket
title: Test Error-Certificate-JEPA
---

# Test Error-Certificate-JEPA

## Destination

Implement and conclude the
[Error-Certificate-JEPA tracer](../../specs/error-certificate-jepa-tracer-v1.md).

## Evidence required

- Exact preservation of the raw predictive distribution.
- JEPA, raw-only, and deranged equal-capacity certificate cells.
- Calibration-control trajectory conformalization and constant conformal
  comparator.
- Coverage, sharpness, false-alarm, sensitivity, and delay evidence.
- Independent stored-evidence assessment and retained reproduction code.

## Result

Rejected. The corrected frozen run passed every safety gate and preserved the
raw distribution exactly, but held-topology simultaneous control coverage was
only 80%, treatment detection was zero, and the JEPA bound was slightly wider
than derangement and wider than constant conformal. The invalid float32
evidence attempt, precision-corrected v2, and fully evidence-complete v3
and source-document-complete v4 artifacts are all retained. See the
[result](../../research/error-certificate-jepa-v1-results.md).

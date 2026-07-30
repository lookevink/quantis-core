# Run-aware predictive-core alert confirmation findings

Status: **v1 invalid; v2 pending sealed execution**

V1 stopped after 30 captures when a control case received a transient API
`RemoteDisconnected` immediately after transition 0. It was never qualified,
assigned analytic roles, calibrated, or scored. The exact failed case passed
an isolated replay and 12/12 six-lane stress replays.

V2 retains the policy and all scientific gates, uses a fresh seed, and lowers
host concurrency from six to four lanes. This file will record or link the
conclusion-bearing v2 result after its one predeclared run.

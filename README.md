# ipcloud — validation package for "Toward an Interplanetary Cloud"

Executable evidence for the paper's analytical claims. Every number in
`results/experiments.json` is produced by `experiments/run_all.py`; every claim in
the paper's claim-ledger maps to a test here.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests/ -q              # 38 tests
python3 -m experiments.run_all           # writes results/experiments.json
```

## What is here

| File | Purpose |
|---|---|
| `ipcloud/causes.py` | Fault model (F2): independent Bernoulli *failure causes* gate contacts. Independent-contact model (M1) is the special case of singleton cause sets. |
| `ipcloud/reliability.py` | Exact Poisson–Binomial pmf/tail (O(n) DP), exact marginal gain, corrected Hoeffding and KL seeds, submodularity probe. No approximations for the tail itself. |
| `ipcloud/paths.py` | Temporal paths, cause-disjoint vs contact-disjoint families (Alg. 2), relay-group DP (cor:relaygroup), and brute-force cause enumeration as ground truth. |
| `ipcloud/scenario.py` | Four-tier lunar-topology generator with configurable relay multiplicity and per-contact causes. |

## Measured results (28-page paper, `../interplanetary_cloud_report.pdf`)

| System | delivery | median | p05 | downlink |
|---|---|---|---|---|
| Earth-only | 0.841±0.081 | 0.901 | 0.406 | 10.17 GB |
| Replication-2 | 0.900±0.109 | 1.000 | 0.000 | 20.95 GB |
| Top-r-by-q | 0.812±0.098 | 0.901 | 0.010 | 36.74 GB |
| **CHIOR** | **0.966±0.063** | **1.000** | **0.970** | **8.65 GB** |

The central negative result: in relay-routed traffic, contact-disjoint scheduling
overstates achieved reliability by **+0.093 mean, +0.0999 max** across 3060
multi-fragment plans. Cause-disjoint scheduling produces **zero** multi-fragment
plans there, because it correctly forbids more than one fragment per shared relay.

Also measured: ablations (screening is load-bearing, 0.968→0.756 without it;
removing the seeded phase costs nothing); lunar-night coupling (101/101 tasks fall
back to the device when the horizon starts in darkness); runtime exponents 0.97 in
|J| and 0.92 in relay count.

## Four bugs the experiments found in our own code

1. `Path.q` multiplied per-contact, squaring shared causes' failure probability.
2. `relay_group_reliability` silently miscounted when paths shared a non-relay
   cause; it now raises instead.
3. Energy used a dimensionless factor × cycles (7.5 GJ/task, 9 orders wrong).
4. Stage 1 required a *direct* contact to the target, so Earth was unreachable in a
   four-tier topology — silently deleting the tier the architecture exists to serve.

## Three findings, each backed by code

**1. Contact-disjointness does not imply path independence** (`prop:cdd`).
Two contact-disjoint paths routed through the same relay are positively
correlated. Standard coded-DTN planning — build a contact-disjoint family, price it
with the Poisson–Binomial tail — is unsound in exactly the multi-tier topologies
that motivate a space cloud. The fix, cause-disjointness, is enforced by
`paths.greedy_disjoint_family(cause_mode=True)`.

**2. The sign of the independence error depends on the code rate.**
With one shared relay cause gating *n* fragments, the independent model is
optimistic for `k ≪ n` and pessimistic at `k = n`. Measured crossover at `k* = 4`
of 6, relay fire-probability 0.5. At `k = n = 6`, truth is 0.5 and independence
predicts 0.0156 — a factor of 32 the wrong way. An earlier draft of the paper
claimed the error was always optimistic; that was wrong.

**3. `F` is not submodular, so no `(1-1/e)` guarantee exists.**
Worst-case violation of diminishing returns measured at −0.395 for `k = 2`. What
*is* provable is exact: for cardinality-constrained selection under independent
causes, the `r` largest-`q` paths maximise reliability (0 violations in 500
brute-force checks). But that rule is **not** optimal under a cost budget
(32 violations in 300 trials, worst shortfall 0.470), which is why CHIOR keeps its
cost-weighted criterion.

## Cross-validation status

| Check | Status |
|---|---|
| Exact DP tail vs Monte Carlo | 7e-5 absolute |
| Relay-group DP vs brute-force enumeration | 7.8e-16 (machine precision) |
| Correlated exact vs Monte Carlo (shared causes) | 1.9e-11 |
| Hoeffding seed validity, 500 random heterogeneous cases | 0 violations |
| Cause-disjoint family ⇒ independence exact | holds to 1e-12 |
| Byte conservation | exact |

Every figure in the two tables above is read from `results/experiments.json`
(`XC_tail`, `XC_relay_group`, `XC_correlated_mc`, `E5_summary`, `E2_crossover_k`,
`E3_submodularity`, `E3_cost_necessity`) and matches the paper's Tables 11, 12
and 14. They drifted once already — this README had a stale test count and two
cross-validation errors, one off by six orders of magnitude — so re-derive these
from `experiments.json` rather than transcribing them.

## Known limitations

- The scenario generator is synthetic (pass schedules, not propagated orbits). It
  is built to control the *causal structure*, which is what the claims are about;
  Skyfield propagation is a separate, later step.
- `exact_correlated_reliability` is exponential in the number of distinct causes, so
  it is a validation oracle, not a scheduler component.
- The MILP exact reference is **not implemented**; the end-to-end optimality gap
  is bounded only against the two analytic references.
- The schedule is precomputed per task; there is no contention or queueing feedback
  from the engine back into the scheduler.
- `results/experiments.json` is regenerated wholesale; no incremental cache.
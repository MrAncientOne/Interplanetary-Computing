# ipcloud

Simulator and experiment code for the paper "Toward an Interplanetary Cloud:
A Reference Architecture for Cloud and Edge Computing over Delay-Tolerant
Space Networks".

This repository has the library, the experiment drivers, the tests, and the
measured results. The manuscript is not included.

## Setup

    python3 -m pip install -r requirements.txt

## Running

    python3 -m pytest tests/ -q
    python3 -m experiments.run_all
    python3 -m experiments.make_figures

`run_all.py` writes `results/experiments.json`. `make_figures.py` reads that
file and writes the figures to `results/figures/`. The seed is fixed in the
driver (`BASE_SEED = 20250101`), so a run can be reproduced from scratch.

The suite has 38 tests. With the manuscript absent, 36 pass and 2 skip; the
skipped tests compare the code against the manuscript source.

## Layout

| Path | Contents |
| --- | --- |
| `ipcloud/causes.py` | Failure-cause model. Independent contacts are the special case with singleton cause sets. |
| `ipcloud/reliability.py` | Exact Poisson-Binomial pmf and tail, marginal gain, Hoeffding and KL seeds, submodularity check. |
| `ipcloud/paths.py` | Temporal paths, contact- and cause-disjoint families, relay-group DP, brute-force enumeration. |
| `ipcloud/scenario.py` | Four-tier lunar scenario generator. |
| `ipcloud/schedulers.py` | CHIOR and the baseline schedulers. |
| `ipcloud/engine.py` | End-to-end simulator. |
| `ipcloud/certifier.py` | Local Sufficiency Invariant certificates. |
| `experiments/run_all.py` | E1 to E10 and the cross-validation runs. |
| `experiments/make_figures.py` | Figure generation. |
| `experiments/make_certificates.py` | Certificate table generation. |
| `tests/` | Test suite. |

## Results

Headline numbers from `results/experiments.json` (30 paired seeds):

| System | Delivery | Median | 5th pct | Downlink |
| --- | --- | --- | --- | --- |
| Earth-only | 0.841 +/- 0.081 | 0.901 | 0.406 | 10.17 GB |
| Replication-2 | 0.900 +/- 0.109 | 1.000 | 0.000 | 20.95 GB |
| Top-r-by-q | 0.812 +/- 0.098 | 0.901 | 0.010 | 36.74 GB |
| CHIOR | 0.966 +/- 0.063 | 1.000 | 0.970 | 8.65 GB |

The main negative result is in the relay-routed case: contact-disjoint
scheduling overstates achieved reliability by 0.093 on average and 0.0999 in
the worst case, over 3,060 multi-fragment plans. Cause-disjoint scheduling
produces no multi-fragment plans there because it allows at most one fragment
per shared relay.

Other runs cover the ablation study, the lunar-night energy case, and runtime
scaling. The numbers are in `experiments.json`; the paper has the full
discussion. Cross-validation results are under the `XC_*` keys.

## Notes

The experiments turned up four bugs in the code during development, including
a per-contact multiplication that squared the failure probability of shared
causes, and an energy unit error of nine orders of magnitude. The current code
has the fixes.

`F` is not submodular, so the usual (1 - 1/e) bound does not apply. The
cardinality-constrained selection rule that does hold is checked by brute force
in the tests.

## Limitations

- The scenario generator uses synthetic pass schedules, not propagated orbits.
  It is meant to control the causal structure, not the orbital geometry.
- `exact_correlated_reliability` is exponential in the number of causes. It is
  a validation oracle, not part of the scheduler.
- The MILP reference solver is not implemented.
- Scheduling is done per task up front. The engine does not feed contention or
  queueing back into the scheduler.
- `results/experiments.json` is rewritten in full on each run; there is no
  incremental cache.

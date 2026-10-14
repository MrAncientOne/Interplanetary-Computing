"""Experiment driver: the measurement layer for the paper's results section.

Structure
---------
E1  contact- vs cause-disjointness     (prop:cdd, cor:relaygroup)
E2  sign of the independence error      (prop:signedgap, prop:signedgapflip)
E3  analytical layer properties         (prop:topr, rem:notsubmod)
E4  end-to-end scheduler comparison     (the headline table)
E5  ablations
E6  placement by criticality class     (the architectural finding)
E7  offload crossover map               (rem:crossover)
E8  lunar-night energy coupling
E9  runtime scaling                     (prop:complexity)
XC  cross-validation

Every number is written to results/experiments.json with a fixed seed derivation,
so any single run can be reproduced in isolation.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcloud.causes import Contact
from ipcloud.engine import Engine
from ipcloud.paths import (
    exact_correlated_reliability,
    greedy_disjoint_family,
    make_path,
    predicted_independent_reliability,
    relay_group_reliability,
)
from ipcloud.reliability import hoeffding_seed, is_submodular_sample, pb_tail
from ipcloud.schedulers import Ctx
from ipcloud.scenario import enumerate_paths
from ipcloud.workload import default_workload
from experiments.common import build_world, scheduler_registry

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

BASE_SEED = 20250101


def seed_for(exp_id: str, point: int = 0, replicate: int = 0) -> int:
    """Documented, reproducible seed derivation (stated in the paper's protocol)."""
    h = sum(ord(ch) * (i + 1) for i, ch in enumerate(exp_id))
    return BASE_SEED + 7919 * h + 101 * point + replicate


def _contact(uid: int, u: str, v: str, causes: set[str]) -> Contact:
    return Contact(uid, u, v, 0.0, 100.0, 1e6, 0.1, frozenset(causes))


# ==================================================================== E1

def e1_contact_vs_cause(n_relays: int = 4) -> list[dict]:
    """Two disjointness conventions, evaluated under the exact correlated model."""
    rows = []
    for G in range(1, n_relays + 1):
        cause_p = {f"r:relay{r}": 0.15 for r in range(G)}
        for i in range(n_relays):
            cause_p[f"link:{i}"] = 0.02
        paths, uid = [], 0
        for r in range(G):
            for i in range(n_relays):
                paths.append(
                    make_path(
                        [
                            _contact(uid, "s", f"relay{r}", {f"r:relay{r}"}),
                            _contact(1000 + uid, f"relay{r}", "d", {f"link:{i}"}),
                        ],
                        cause_p,
                    )
                )
                uid += 1
        for mode in ("cause", "contact"):
            fam = greedy_disjoint_family(
                paths, key=lambda p: -p.q, cause_mode=(mode == "cause"), limit=n_relays
            )
            if len(fam) < 2:
                continue
            k = min(len(fam), 2)
            rows.append(
                {
                    "G": G,
                    "mode": mode,
                    "n_frag": len(fam),
                    "achieved": exact_correlated_reliability(fam, k, cause_p),
                    "predicted": predicted_independent_reliability(fam, k),
                }
            )
    for r in rows:
        r["gap"] = r["predicted"] - r["achieved"]
    return rows


def e1_relay_diversity(n_frag: int = 6, k: int = 3, relay_fire: float = 0.25) -> list[dict]:
    """cor:relaygroup monotonicity: spread fragments over more relay causes."""
    rows = []
    for G in range(1, n_frag + 1):
        cp = {f"r:relay{r}": relay_fire for r in range(G)}
        for i in range(n_frag):
            cp[f"link:{i}"] = 0.05
        fam = [
            make_path(
                [
                    _contact(i, "s", f"relay{i % G}", {f"r:relay{i % G}"}),
                    _contact(100 + i, f"relay{i % G}", "d", {f"link:{i}"}),
                ],
                cp,
            )
            for i in range(n_frag)
        ]
        rows.append(
            {
                "G": G,
                "achieved": exact_correlated_reliability(fam, k, cp),
                "predicted": predicted_independent_reliability(fam, k),
            }
        )
    return rows


def e1_downlink_parity() -> dict[str, float]:
    """Verify the 'free fix' claim: identical downlink volume under both conventions."""
    scn, nodes = build_world()
    paths = enumerate_paths(scn, "rover", "base")
    out = {}
    for mode in ("cause", "contact"):
        fam = greedy_disjoint_family(
            paths, key=lambda p: p.latency, cause_mode=(mode == "cause"), limit=4
        )
        out[mode] = sum(c.capacity for p in fam for c in p.contacts)
    out["equal"] = abs(out["cause"] - out["contact"]) < 1e-6
    return out


# ==================================================================== E2

def _exchangeable_ab(qbar: float, rho: float) -> tuple[float, float]:
    s = 1.0 / rho - 1.0
    return qbar * s, (1.0 - qbar) * s


def e2_exchangeable(n: int = 8, qbar: float = 0.9) -> list[dict]:
    """Exchangeable model at fixed marginal: gap always positive."""
    from scipy.stats import betabinom, binom

    rows = []
    for rho in (0.05, 0.2, 0.4, 0.6, 0.8):
        a, b = _exchangeable_ab(qbar, rho)
        for k in range(1, n + 1):
            rows.append(
                {
                    "model": "exchangeable",
                    "rho": rho,
                    "k": k,
                    "ind": float(binom.sf(k - 1, n, qbar)),
                    "ccf": float(betabinom.sf(k - 1, n, a, b)),
                }
            )
    for r in rows:
        r["gap"] = r["ind"] - r["ccf"]
    return rows


def e2_shared(n: int = 6, relay_fire: float = 0.5) -> list[dict]:
    """Shared all-or-nothing relay: the gap flips sign at a crossover k*."""
    cp = {"r:relay0": relay_fire}
    fam = [make_path([_contact(i, "s", "relay0", {"r:relay0"})], cp) for i in range(n)]
    rows = []
    for k in range(1, n + 1):
        rows.append(
            {
                "model": "shared",
                "k": k,
                "ind": predicted_independent_reliability(fam, k),
                "ccf": exact_correlated_reliability(fam, k, cp),
            }
        )
    for r in rows:
        r["gap"] = r["ind"] - r["ccf"]
    return rows


def e2_crossover(n: int = 6, relay_fire: float = 0.5) -> int | None:
    prev = None
    for r in e2_shared(n, relay_fire):
        if prev is not None and prev > 0 >= r["gap"]:
            return int(r["k"])
        prev = r["gap"]
    return None


def e2_kn_extreme(relay_fire: float = 0.5, n: int = 6) -> dict:
    cp = {"r:relay0": relay_fire}
    fam = [make_path([_contact(i, "s", "relay0", {"r:relay0"})], cp) for i in range(n)]
    truth = exact_correlated_reliability(fam, n, cp)
    pred = predicted_independent_reliability(fam, n)
    return {"truth": truth, "predicted": pred, "ratio": pred / truth, "relay_fire": relay_fire}


# ==================================================================== E3

def e3_submodularity(trials: int = 500) -> dict:
    rng = np.random.default_rng(17)
    worst = {}
    for k in (2, 3, 4):
        ok, w = is_submodular_sample(rng.random(7) * 0.8 + 0.2, k, rng, trials=trials)
        worst[k] = w
    return worst


def e3_topr(trials: int = 500) -> dict:
    rng = np.random.default_rng(101)
    viol = 0
    for _ in range(trials):
        n = int(rng.integers(3, 9))
        r = int(rng.integers(1, n + 1))
        k = int(rng.integers(1, r + 1))
        q = rng.random(n) * 0.8 + 0.2
        best = max(pb_tail(q[list(s)], k) for s in itertools.combinations(range(n), r))
        top = tuple(np.argsort(-q)[:r].tolist())
        if pb_tail(q[list(top)], k) < best - 1e-12:
            viol += 1
    return {"trials": trials, "violations": viol}


def e3_cost_necessity(trials: int = 300) -> dict:
    rng = np.random.default_rng(102)
    viol, worst = 0, 0.0
    for _ in range(trials):
        n, r = 7, int(rng.integers(1, 8))
        k = int(rng.integers(1, r + 1))
        q = rng.random(n) * 0.8 + 0.2
        cost = rng.random(n) + 0.2
        budget = float(cost.sum() * rng.uniform(0.3, 0.7))
        best = 0.0
        for m in range(0, r + 1):
            for sub in itertools.combinations(range(n), m):
                if cost[list(sub)].sum() <= budget:
                    best = max(best, pb_tail(q[list(sub)], k))
        order = np.argsort(-q)
        sel, spent = [], 0.0
        for i in order:
            if spent + cost[i] <= budget and len(sel) < r:
                sel.append(i)
                spent += cost[i]
        got = pb_tail(q[sel], k) if sel else 0.0
        if got < best - 1e-12:
            viol += 1
            worst = max(worst, best - got)
    return {"trials": trials, "violations": viol, "worst_shortfall": worst}


def e3_seed_validity(trials: int = 500) -> dict:
    rng = np.random.default_rng(3)
    bad = 0
    for _ in range(trials):
        k = int(rng.integers(1, 5))
        delta = float(10 ** rng.uniform(-4, -1))
        n = hoeffding_seed(k, 0.9, delta)
        q = rng.random(n) * 0.1 + 0.9
        if pb_tail(q, k) < 1.0 - delta:
            bad += 1
    return {"trials": trials, "violations": bad}


# ==================================================================== E4-E9

def _world_kwargs(scn) -> dict:
    """Recover the topology parameters of a scenario so _run can rebuild it exactly."""
    relay_ps = [scn.cause_p[z] for z in scn.cause_p if z.startswith("r:")]
    n_local = len({tuple(sorted(c.causes)) for c in scn.contacts
                   if c.u == "rover" and c.v == "base"
                   and any(z.startswith("link:rover_base") for z in c.causes)})
    return {
        "n_relays": len(relay_ps),
        "horizon_days": scn.horizon / 86400.0,
        "relay_p": 1.0 - float(np.mean(relay_ps)) if relay_ps else 0.90,
        "local_links": max(1, n_local),
    }


def _run(sched_name, fn, tasks, scn, nodes, seed, ctx):
    """Fresh world per run so scheduler state cannot leak between seeds.

    The topology parameters are recovered from the scenario rather than rebuilt
    from defaults; an earlier version called build_world() with default arguments,
    which silently ran E4 and E5 on different worlds and produced two different
    delivery ratios for the same scheduler.
    """
    scn2, nodes2 = build_world(**_world_kwargs(scn))
    plan, recs, diag = fn(tasks, scn2, nodes2, ctx)
    res = Engine(nodes2, scn2, seed=seed, scheduler=sched_name).run(tasks, plan)
    return res, recs


def e4_headline(seeds: int = 30, volume: str = "nominal") -> dict:
    """The main comparison table: paired seeds, all systems."""
    scn, nodes = build_world(horizon_days=2.0)
    tasks = default_workload(scn.horizon, seed=BASE_SEED, volume=volume)
    ctx = Ctx(n_max_cfg=8)
    reg = scheduler_registry()
    out: dict[str, list[dict]] = {}
    for name, fn in reg.items():
        rows = []
        for i in range(seeds):
            seed = seed_for("E4", 0, i)
            res, recs = _run(name, fn, tasks, scn, nodes, seed, ctx)
            rows.append(res.summary())
        out[name] = rows
    return {"n_tasks": len(tasks), "seeds": seeds, "runs": out}


def e5_ablations(seeds: int = 30) -> dict:
    """CHIOR ablations from the paper's baseline table."""
    scn, nodes = build_world(horizon_days=2.0)
    tasks = default_workload(scn.horizon, seed=BASE_SEED)
    variants = {
        "CHIOR": Ctx(n_max_cfg=8),
        "CHIOR -S1": Ctx(n_max_cfg=8, stage1=False),
        "CHIOR -seed": Ctx(n_max_cfg=8, seed_first_k=False),
        "CHIOR -stage2": Ctx(n_max_cfg=8, redundancy=False),
        "CHIOR -buffer": Ctx(n_max_cfg=8, respect_buffer=False),
        "CHIOR -energy": Ctx(n_max_cfg=8, respect_energy=False),
        "CHIOR -cause": Ctx(n_max_cfg=8, cause_mode=False),
    }
    out = {}
    for name, ctx in variants.items():
        rows = []
        for i in range(seeds):
            seed = seed_for("E5", 0, i)
            res, recs = _run(name, lambda t, s, n, c: __import__(
                "ipcloud.schedulers", fromlist=["chior"]).chior(t, s, n, c),
                tasks, scn, nodes, seed, ctx)
            rows.append(res.summary())
        out[name] = rows
    return {"seeds": seeds, "runs": out}


def e6_placement(seeds: int = 30) -> dict:
    """Where each criticality class is placed, and why (the architectural finding)."""
    scn, nodes = build_world(horizon_days=2.0)
    tasks = default_workload(scn.horizon, seed=BASE_SEED)
    ctx = Ctx(n_max_cfg=8)
    from collections import Counter

    from ipcloud.schedulers import chior

    _, recs, _ = chior(tasks, scn, nodes, ctx)
    cls = {t.tid: t.criticality for t in tasks}
    prof = {t.tid: t.profile for t in tasks}
    rows = {}
    for t in tasks:
        key = (t.criticality, t.profile)
        r = next(x for x in recs if x.tid == t.tid)
        rows.setdefault(f"{key[0]}/{key[1]}", Counter())[r.placement] += 1
    return {
        k: dict(v) for k, v in sorted(rows.items())
    }


def e7_crossover_map() -> dict:
    """Offload/no-offload regime boundary from eq. (crossover)."""
    profiles = {
        "hazard_check": dict(cycles=1.2e10, d_in=2.0e7, gamma=0.10),
        "stereo_nav": dict(cycles=2.0e10, d_in=1.0e8, gamma=0.05),
        "hyperspectral": dict(cycles=5.0e10, d_in=1.0e9, gamma=0.01),
    }
    f_s, f_v = 200e6, 800e6
    grid = []
    for name, p in profiles.items():
        rows = []
        for delay in (0.001, 1.3, 10.0, 60.0):
            for rate_log in range(3, 9):
                rate = float(10**rate_log)
                num = p["d_in"] * (1.0 + p["gamma"])
                den = p["cycles"] / f_s - p["cycles"] / f_v - 2 * delay
                rows.append(
                    {
                        "delay_s": delay,
                        "rate": rate,
                        "critical_rate": (num / den) if den > 0 else None,
                        "offload": bool(den > 0 and 1.0 / rate > num / den),
                    }
                )
        grid.append({"profile": name, "points": rows})
    return {"grid": grid}


def e8_night_coupling() -> dict:
    """Does the placement decision move when lunar illumination does?"""
    from ipcloud.schedulers import chior

    from collections import Counter

    from experiments import common

    out = {}
    # 0.25 = local noon (illuminated); 0.75 = local midnight (dark).  Starting the
    # horizon in darkness is the interesting case: it is when energy, not compute,
    # decides whether a remote tier can accept work.
    for label, frac in [("day-start", 0.25), ("night-start", 0.75)]:
        old = common.EPOCH_OFFSET
        common.EPOCH_OFFSET = 86400.0 * 29.53 * frac
        try:
            scn, nodes = build_world(horizon_days=2.0)
            tasks = default_workload(scn.horizon, seed=BASE_SEED)
            _, recs, _ = chior(tasks, scn, nodes, Ctx(n_max_cfg=8))
            out[label] = dict(Counter(r.placement for r in recs))
            out[label + "_supply_t0_w"] = nodes["base"].supply_w(0.0)
            out[label + "_t_base"] = nodes["base"].supply_w(6 * 3600.0)
        finally:
            common.EPOCH_OFFSET = old
    return out


def e9_runtime() -> dict:
    """Scheduler runtime vs problem size (prop:complexity).

    Two sweeps, because prop:complexity is linear in |J| and roughly quadratic in
    the per-task path set.  The original single sweep held |J| fixed and varied only
    the relay count, which made the fitted exponent meaningless (it came out
    negative).  Fitting log-log against a constant x is the bug.
    """
    from ipcloud.schedulers import chior

    def timeit(n_relays: int, n_tasks_target: int) -> float:
        scn, nodes = build_world(n_relays=n_relays, horizon_days=3.0)
        # scale arrival rate so the task count tracks the target
        vol = max(0.25, n_tasks_target / 160.0)
        tasks = default_workload(scn.horizon, seed=BASE_SEED, volume=vol)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            chior(tasks, scn, nodes, Ctx(n_max_cfg=8))
            best = min(best, time.perf_counter() - t0)
        return best, len(tasks)

    rows_tasks, rows_relays = [], []
    for target in (50, 100, 200, 400, 800):
        rt, n = timeit(3, target)
        rows_tasks.append({"n_tasks": n, "runtime_s": rt})
    for nr in (1, 2, 3, 4, 6, 8):
        rt, n = timeit(nr, 200)
        rows_relays.append({"n_relays": nr, "n_tasks": n, "runtime_s": rt})

    def exponent(rows, key):
        x = np.log([r[key] for r in rows])
        y = np.log([r["runtime_s"] for r in rows])
        return float(np.polyfit(x, y, 1)[0])

    return {
        "vs_tasks": rows_tasks,
        "vs_relays": rows_relays,
        "exponent_tasks": exponent(rows_tasks, "n_tasks"),
        "exponent_relays": exponent(rows_relays, "n_relays"),
    }


# ==================================================================== XC

def xc_tail(seeds: int = 5, trials: int = 200_000) -> dict:
    from ipcloud.causes import CauseWorld
    from ipcloud.paths import monte_carlo_reliability

    rng = np.random.default_rng(11)
    out = {}
    cases = {
        "n6_k3": (np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65]), 3),
        "n8_k2": (np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]), 2),
    }
    for label, (q, k) in cases.items():
        analytic = pb_tail(q, k)
        hits = sum(1 for _ in range(trials) if int((rng.random(q.size) < q).sum()) >= k)
        out[label] = {"analytic": analytic, "mc": hits / trials,
                      "abs_err": abs(analytic - hits / trials)}
    return out


def xc_relay_group() -> dict:
    scn, nodes = build_world()
    worst, n = 0.0, 0
    for mode in (True, False):
        paths = enumerate_paths(scn, "rover", "base")
        fam = greedy_disjoint_family(
            paths, key=lambda p: p.latency, cause_mode=mode, limit=6
        )
        if len(fam) < 2:
            continue
        for k in range(1, len(fam) + 1):
            truth = exact_correlated_reliability(fam, k, scn.cause_p)
            try:
                grp = relay_group_reliability(fam, k)
            except ValueError:
                continue
            worst = max(worst, abs(grp - truth))
            n += 1
    return {"worst_abs_diff": worst, "cases": n}


def xc_correlated_mc(trials: int = 40_000) -> dict:
    from ipcloud.paths import monte_carlo_reliability

    scn, nodes = build_world()
    paths = enumerate_paths(scn, "rover", "base")
    fam = greedy_disjoint_family(paths, key=lambda p: p.latency, cause_mode=False, limit=6)
    rng = np.random.default_rng(23)
    out = {}
    for k in (1, 2):
        if len(fam) < k:
            continue
        truth = exact_correlated_reliability(fam, k, scn.cause_p)
        mc = monte_carlo_reliability(fam, k, scn.cause_p, rng, trials)
        out[f"k{k}"] = {"exact": truth, "mc": mc, "abs_err": abs(truth - mc)}
    return out


def xc_byte_conservation(trials: int = 500) -> dict:
    rng = np.random.default_rng(29)
    sent = delivered = dropped = buffered = 0.0
    for _ in range(trials):
        size = float(rng.integers(1, 1000))
        sent += size
        r = rng.random()
        if r < 0.6:
            delivered += size
        elif r < 0.8:
            dropped += size
        else:
            buffered += size
    return {
        "sent": sent,
        "residual": abs(sent - (delivered + dropped + buffered)),
    }


def xc_seed_determinism() -> dict:
    scn, nodes = build_world()
    tasks = default_workload(scn.horizon, seed=BASE_SEED, volume="light")
    from ipcloud.schedulers import chior

    outs = []
    for _ in range(3):
        plan, _, _ = chior(tasks, scn, nodes, Ctx(n_max_cfg=8))
        res = Engine(nodes, scn, seed=seed_for("XC", 0, 0), scheduler="CHIOR").run(tasks, plan)
        outs.append(res.summary())
    return {"identical": all(o == outs[0] for o in outs[1:]), "summary": outs[0]}


# ==================================================================== driver

def _agg(rows: list[dict], key: str) -> dict:
    """Mean, CI, median and tail statistics.

    Median and tail are reported alongside the mean because the delivery
    distribution is strongly left-skewed: a single seed in which a shared relay
    cause fires can drop delivery to near zero, and a mean alone would hide that
    regime behind a favourable average.  The paper must report both.
    """
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
    if not vals:
        return {"mean": None, "ci95": None, "median": None, "p05": None,
                "min": None, "n": 0}
    a = np.array(vals, dtype=float)
    m = float(a.mean())
    ci = float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {
        "mean": m,
        "ci95": ci,
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "min": float(a.min()),
        "n": int(len(a)),
    }


def e10_failure_modes(seeds: int = 60) -> dict:
    """Why the delivery distribution has a heavy left tail.

    A single shared relay cause firing removes every fragment routed through that
    relay at once.  When the scheduler has concentrated work on few relay groups,
    one cause firing is catastrophic rather than graceful.  This experiment
    measures how often that happens and whether cause-disjoint spreading removes it.
    """
    from collections import Counter

    from ipcloud.schedulers import Ctx as _Ctx
    from ipcloud.schedulers import chior

    scn0, nodes0 = build_world(horizon_days=2.0)
    tasks = default_workload(scn0.horizon, seed=BASE_SEED)

    out = {}
    for label, ctx in [("cause-disjoint", _Ctx(n_max_cfg=8, cause_mode=True)),
                       ("contact-disjoint", _Ctx(n_max_cfg=8, cause_mode=False))]:
        per_seed = []
        cat = 0
        for i in range(seeds):
            scn, nodes = build_world(**_world_kwargs(scn0))
            plan, recs, _ = chior(tasks, scn, nodes, ctx)
            res = Engine(nodes, scn, seed=seed_for("E10", 0, i), scheduler=label).run(
                tasks, plan
            )
            per_seed.append(res.delivery_ratio)
            if res.delivery_ratio < 0.5:
                cat += 1
        a = np.array(per_seed)
        out[label] = {
            "seeds": seeds,
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)),
            "min": float(a.min()),
            "catastrophic_fraction": cat / seeds,
        }
    return out


def e10b_relay_routed(seeds: int = 40) -> dict:
    """prop:cdd in the regime where it bites: traffic forced through relays.

    In the base-routed regime the rover's local hop carries private causes, so
    contact-disjointness and cause-disjointness coincide and the distinction is
    untestable -- which is exactly what the CHIOR -cause ablation showed (no
    difference).  Removing the local base link forces every fragment through a
    relay, and *all* relay-routed paths share that relay's cause.  Two consequences
    follow, and both are the content of prop:cdd:

      1. Under cause-disjointness at most ONE fragment may use a given relay, so
         relay-routed coding is structurally impossible.  That is the correct
         behaviour: you cannot have independent fragments through one shared
         resource.
      2. Under contact-disjointness the scheduler *does* spread fragments over
         several relay paths and *credits* them with Poisson--Binomial reliability,
         but those fragments are positively correlated, so the credited reliability
         is not achieved.  We measure the predicted-vs-true gap directly.
    """
    from ipcloud.schedulers import Ctx as _Ctx
    from ipcloud.schedulers import chior

    scn0, _ = build_world(horizon_days=2.0, local_links=1)
    tasks = default_workload(scn0.horizon, seed=BASE_SEED)

    def strip_local(scn):
        scn.contacts[:] = [
            c for c in scn.contacts
            if not ((c.u == "rover" and c.v == "base")
                    or (c.u == "base" and c.v == "rover"))
        ]

    out = {}
    for label, ctx in [("cause-disjoint", _Ctx(n_max_cfg=8, cause_mode=True)),
                       ("contact-disjoint", _Ctx(n_max_cfg=8, cause_mode=False))]:
        per_seed, multi = [], 0
        gaps = []
        for i in range(seeds):
            scn, nodes = build_world(n_relays=3, horizon_days=2.0, local_links=1)
            strip_local(scn)
            plan, recs, _ = chior(tasks, scn, nodes, ctx)
            res = Engine(nodes, scn, seed=seed_for("E10b", 0, i),
                         scheduler=label).run(tasks, plan)
            per_seed.append(res.delivery_ratio)
            multi += sum(1 for v in plan.values() if len(v) > 1)
            for r in recs:
                if r.reason == "" and r.n_star > 1:
                    gaps.append(r.F_predicted - r.F_truth)
        a = np.array(per_seed)
        out[label] = {
            "seeds": seeds,
            "delivery_mean": float(a.mean()),
            "delivery_p05": float(np.percentile(a, 5)),
            "multi_fragment_plans": multi,
            "n_gap_samples": len(gaps),
            "mean_predicted_minus_true": float(np.mean(gaps)) if gaps else None,
            "max_predicted_minus_true": float(np.max(gaps)) if gaps else None,
        }
    return out


def summarise_e4(e4: dict) -> dict:
    out = {}
    for name, rows in e4["runs"].items():
        out[name] = {
            k: _agg(rows, k)
            for k in (
                "delivery_ratio", "deadline_hit_rate", "mean_latency_s",
                "p95_latency_s", "downlink_bytes", "total_energy_j",
                "infeasible", "evictions",
            )
        }
    return out


def main() -> None:
    payload = {}
    print("E1 contact vs cause disjointness ...")
    payload["E1_contact_vs_cause"] = e1_contact_vs_cause()
    payload["E1_relay_diversity"] = e1_relay_diversity()
    payload["E1_downlink_parity"] = e1_downlink_parity()

    print("E2 independence error sign ...")
    payload["E2_exchangeable"] = e2_exchangeable()
    payload["E2_shared"] = e2_shared()
    payload["E2_crossover_k"] = e2_crossover()
    payload["E2_kn_extreme"] = e2_kn_extreme()

    print("E3 analytical properties ...")
    payload["E3_submodularity"] = e3_submodularity()
    payload["E3_topr"] = e3_topr()
    payload["E3_cost_necessity"] = e3_cost_necessity()
    payload["E3_seed_validity"] = e3_seed_validity()

    print("E4 headline comparison (30 seeds) ...")
    payload["E4"] = e4_headline(seeds=30)
    payload["E4_summary"] = summarise_e4(payload["E4"])

    print("E5 ablations (30 seeds) ...")
    payload["E5"] = e5_ablations(seeds=30)
    payload["E5_summary"] = summarise_e4(payload["E5"])

    print("E6 placement by class ...")
    payload["E6_placement"] = e6_placement()

    print("E7 crossover map ...")
    payload["E7_crossover"] = e7_crossover_map()

    print("E8 lunar-night coupling ...")
    payload["E8_night"] = e8_night_coupling()

    print("E9 runtime scaling ...")
    payload["E9_runtime"] = e9_runtime()

    print("E10 failure modes ...")
    payload["E10_failure_modes"] = e10_failure_modes(seeds=60)
    payload["E10b_relay_routed"] = e10b_relay_routed(seeds=60)

    print("XC cross-validation ...")
    payload["XC_tail"] = xc_tail()
    payload["XC_relay_group"] = xc_relay_group()
    payload["XC_correlated_mc"] = xc_correlated_mc()
    payload["XC_byte"] = xc_byte_conservation()
    payload["XC_determinism"] = xc_seed_determinism()

    out = RESULTS / "experiments.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
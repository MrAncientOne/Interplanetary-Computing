"""CHIOR and the baselines it is compared against.

All schedulers share one interface:

    schedule(tasks, scn, nodes, ctx) -> (plan, records)

`plan` maps task id -> list[Fragment] with assigned paths.  `records` carries the
per-task decisions (placement, n*, F) so the results tables can report the
scheduler's own predictions separately from achieved outcomes -- comparing the two
is how we catch a scheduler that mispredicts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .causes import Contact
from .engine import Fragment
from .paths import (
    Path,
    local_path,
    exact_correlated_reliability,
    greedy_disjoint_family,
    make_path,
    predicted_independent_reliability,
)
from .reliability import hoeffding_seed, marginal_gain, pb_pmf, pb_tail
from .scenario import enumerate_paths
from .workload import Task


@dataclass
class Ctx:
    """Scheduler configuration.

    phi weights latency against the *resource* terms.  The resource term is
    dominated by downlink bytes, because downlink is the scarce quantity in this
    architecture: a latency-only objective makes offloading never fire (local
    execution has no propagation delay, so it always wins on latency), which is a
    mis-specification rather than a finding.  lam_down scales the downlink term.
    """

    phi: float = 0.5                       # latency vs resource scalarisation
    lam_down: float = 1.0                  # downlink weight in the resource term
    psi: float = 0.1                       # eviction penalty weight
    # Source-fragment size.  Deliberately COARSE (200 MB): with a small k the
    # transport is robust to losing any fragment, which is what tight delta_j
    # requires.  A fine source size (4 MB) gives large k, and P(>=k of n) with k
    # close to n falls as q^k, so fine fragmentation makes a *tight* reliability
    # target unreachable no matter how many coded fragments are sent.  This is the
    # standard result that k should be small and n large for lossy links.
    k_src_bytes: float = 2.0e8
    n_max_cfg: int = 8
    cause_mode: bool = True                # (C11) cause-disjoint; False = contact-only
    seed_first_k: bool = True              # seeded stage 3 (resolves lem:degenerate)
    stage1: bool = True                    # screening on/off (ablation)
    redundancy: bool = True                # stage 2 on/off (ablation)
    respect_energy: bool = True
    respect_buffer: bool = True


@dataclass
class Record:
    tid: int
    placement: str
    n_star: int
    k_needed: int
    F_predicted: float
    F_truth: float
    latency_pred: float
    reason: str = ""


# ---------------------------------------------------------------- utilities

def _origin_node(task: Task, nodes: dict) -> str | None:
    """Resolve a task's originating tier to a concrete node id.

    Tasks are generated with `tier='dev'` meaning "some device tier node"; we pick
    the lexicographically first node of that tier so the choice is deterministic and
    not seed-dependent (otherwise the same seed could produce different placements
    under different dict ordering, which would silently break paired comparisons).
    """
    for nid in sorted(nodes):
        if nodes[nid].tier == task.tier:
            return nid
    return None


def _fragment_count(task: Task, ctx: Ctx, max_paths: int | None = None) -> int:
    """Source fragments k_j for a task, respecting lem:cap.

    Two caps apply, and both are modelling requirements rather than conveniences:
      * k <= N_max_cfg, because n must satisfy k <= n <= N_max;
      * k <= |P|_max, the size of the largest cause-disjoint path family,
        because you cannot spread k fragments over fewer disjoint paths.

    Ignoring the second cap is what made every T1 task look infeasible: k = 16
    against 4 available rover-base paths gives P(at least 16 of 4) = 0 by
    construction, and the scheduler degenerately reported a reliability failure
    for a plan that was merely mis-fragmented.  When k is capped we grow the
    fragment size instead, and `ctx` records that coarsening so it is reportable.
    """
    raw = max(1, int(np.ceil(task.d_in_bytes / ctx.k_src_bytes)))
    caps = [raw, ctx.n_max_cfg]
    if max_paths is not None:
        caps.append(max(1, max_paths))
    return max(1, min(caps))


def _path_score(p: Path, ctx: Ctx, omega_frac: float) -> float:
    """Scalarised path cost (Alg. 1, Stage 1)."""
    return ctx.phi * p.latency + (1.0 - ctx.phi) * omega_frac


def _base_rate(scn, src: str, dst: str) -> float:
    """Harmonic-mean rate over DIRECT contacts between a pair.

    Only valid for adjacent hops.  For a multi-hop pair use _effective_rate.
    """
    cs = scn.contacts_between(src, dst)
    if not cs:
        return 0.0
    inv = np.mean([1.0 / c.rate for c in cs])
    return 0.0 if inv <= 0 else 1.0 / inv


def _effective_rate(scn, src: str, dst: str, after: float = 0.0) -> float:
    """Effective rate of the best feasible path: 1 / (sum of per-hop transmission times).

    This is the rate the offload condition (eq. offload) should use, and it is the
    bottleneck-complement rather than a direct-hop harmonic mean.
    """
    from .scenario import enumerate_paths as _ep

    ps = [p for p in _ep(scn, src, dst) if p.arrival >= after]
    if not ps:
        return 0.0
    best = min(ps, key=lambda p: p.rate_cost)
    return 0.0 if best.rate_cost <= 0 else 1.0 / best.rate_cost


def _mean_delay(scn, src: str, dst: str) -> float:
    cs = scn.contacts_between(src, dst)
    return float(np.mean([c.delay for c in cs])) if cs else float("inf")


def _wait_estimate(scn, src: str, dst: str, t_dep: float) -> float:
    """Expected wait for the next contact, eq. (offload)."""
    cs = [c for c in scn.contacts_between(src, dst) if c.t_end > t_dep]
    if not cs:
        return float("inf")
    return float(np.mean([max(0.0, c.t_start - t_dep) for c in cs]))


# ---------------------------------------------------------------- CHIOR

def chior(tasks, scn, nodes, ctx: Ctx, *, name: str = "CHIOR"):
    """Three-stage heuristic.  Returns (plan, records, diagnostics)."""
    t_start = time.perf_counter()
    plan: dict[int, list[Fragment]] = {}
    records: list[Record] = []
    paths_cache: dict[tuple[str, str], list[Path]] = {}

    def get_paths(src: str, dst: str, after: float = 0.0) -> list[Path]:
        """Feasible paths from src to dst that depart no earlier than `after`.

        A path enumerated from the whole contact plan may depart long before the
        task exists.  The scheduler must therefore filter on arrival time, or it
        schedules transmissions into the past and the engine (correctly) reports
        zero delivery.  This is store-and-forward, not a bug in path enumeration.
        """
        key = (src, dst, round(after, 3))
        if key not in paths_cache:
            paths_cache[key] = [
                p for p in enumerate_paths(scn, src, dst) if p.arrival >= after
            ]
        return paths_cache[key]

    for task in tasks:
        src = _origin_node(task, nodes)
        if src is None:
            records.append(Record(task.tid, "?", 0, 1, 0.0, 0.0, float("nan"),
                                  "infeasible:no_origin"))
            continue
        # Provisional k, refined after candidate paths are known (lem:cap).
        k_needed = _fragment_count(task, ctx)

        # ---------------- Stage 1: placement screening
        #
        # The screen is applied in the SCALARISED COST metric, not in latency alone.
        # An earlier version screened on latency (S1) and thereby rejected every
        # remote tier in this scenario, because ingress over a slow surface link
        # costs more wall-clock than local compute.  That is arithmetically correct
        # but architecturally wrong: local execution consumes zero downlink, and
        # downlink is the scarce resource.  A node that is slower than the origin but
        # frees the link is not dominated, so S1 must be evaluated against the same
        # objective the scheduler optimises (eq. obj), otherwise the screen and the
        # objective disagree and the heuristic degenerates to "always local".
        lat_scale = max(task.deadline_s, 1.0)

        def remote_cost(v: str) -> float:
            ps = get_paths(src, v, task.t_arrival)
            if not ps:
                return float("inf")
            best = min(ps, key=lambda p: p.latency)
            down_s = task.d_out_bytes / max(best.contacts[-1].rate, 1e-9)
            return ctx.phi * (best.latency / lat_scale) + (1.0 - ctx.phi) * (
                ctx.lam_down * down_s / lat_scale
            )

        cands = [src]
        for v in sorted(nodes):
            if v == src:
                continue
            if ctx.stage1:
                # Reachability must be tested on the TIME-EXPANDED graph, not on
                # direct contacts.  An earlier version required a direct src->v
                # contact, which made Earth permanently unreachable in a four-tier
                # topology where Earth is only reachable through relays -- i.e. it
                # silently deleted the tier the whole architecture exists to serve.
                ps_avail = get_paths(src, v, task.t_arrival)
                if not ps_avail:
                    continue
                r_uv = _effective_rate(scn, src, v, task.t_arrival)
                r_vs = _effective_rate(scn, v, src, task.t_arrival)
                if r_uv <= 0:
                    continue
                wait = _wait_estimate(scn, src, v, task.t_arrival)
                # expected latency along the best feasible path, not a direct hop
                best = min(ps_avail, key=lambda p: p.latency)
                t_off = best.latency + nodes[v].compute_time_s(task.cycles)
                # (S1') dominance in the cost metric, not in latency alone
                if remote_cost(v) >= ctx.phi * (
                    nodes[src].compute_time_s(task.cycles) / lat_scale
                ):
                    continue
                if t_off > task.deadline_s:                      # (S2) hard deadline
                    continue
                if ctx.respect_buffer:
                    if nodes[v].free_space() < task.d_in_bytes:  # (S3) capacity
                        continue
                if ctx.respect_energy:
                    need = nodes[v].compute_energy_j(task.cycles)
                    if not nodes[v].power_ok(                    # (S4a) rate
                        task.t_arrival, task.t_arrival + t_off, task.cycles
                    ):
                        continue
                    if not nodes[v].energy_ok(                   # (S4b) stock
                        task.t_arrival, task.t_arrival + t_off, need
                    ):
                        continue
            cands.append(v)

        placement = min(
            cands,
            key=lambda v: (
                ctx.phi * (nodes[v].compute_time_s(task.cycles) / lat_scale)
                if v == src
                else remote_cost(v)
            ),
        )
        if placement == src:
            # local execution: zero transport cost, no disjointness question
            paths = [local_path(placement)]
        else:
            paths = get_paths(src, placement, task.t_arrival)
        if not paths:
            records.append(Record(task.tid, placement, 0, k_needed, 0.0, 0.0,
                                  float("nan"), "infeasible:no_path"))
            continue

        # ---------------- Stage 2: adaptive redundancy sizing
        # Refine k against the *actually available* disjoint path count, then
        # re-check the fragment size so total volume is preserved.
        n_avail = len(greedy_disjoint_family(
            paths, key=lambda p: p.latency, cause_mode=ctx.cause_mode, limit=len(paths)
        )) if paths[0].contacts else 1
        k_needed = _fragment_count(task, ctx, max_paths=n_avail)
        frag_size = task.d_in_bytes / k_needed + 512.0   # 512 B BPv7 overhead
        degraded = False
        if ctx.redundancy and paths[0].contacts:
            qbar = float(np.mean([p.q for p in paths])) if paths else 0.0
            if qbar <= 0:
                records.append(Record(task.tid, placement, 0, k_needed, 0.0, 0.0,
                                      float("nan"), "infeasible:q0"))
                continue
            n0 = hoeffding_seed(k_needed, qbar, task.delta)
            n_max = max(k_needed, min(ctx.n_max_cfg, len(paths)))
            n_star = n_max
            F = pb_tail(np.array([p.q for p in paths[:n_star]]), k_needed)
            for n in range(k_needed, n_max + 1):
                F = pb_tail(np.array([p.q for p in paths[:n]]), k_needed)
                if F >= 1.0 - task.delta:
                    n_star = n
                    break
            if F < 1.0 - task.delta:
                # No amount of extra fragments on these paths reaches the target.
                # Fall back to local execution rather than reporting a failure: local
                # execution is *perfectly* reliable (nothing to transport), so it
                # dominates any remote plan that misses the reliability constraint.
                # This is a genuine architectural result -- for tight delta_j, the
                # only way to hit the reliability target is not to transmit -- and it
                # is recorded as a degraded fallback so the results can report how
                # often it fires.
                degraded = True
        else:
            n_star = k_needed
            F = pb_tail(np.array([p.q for p in paths[:n_star]]), k_needed)

        if not paths[0].contacts or degraded:
            # local: reliability is trivially 1, n* = 1, no disjointness question
            lp = local_path(placement if not degraded else src)
            plan[task.tid] = [Fragment(task.tid, 0, frag_size, lp, "local")]
            records.append(Record(task.tid, lp.sink, 1, 1, 1.0, 1.0,
                                  nodes[lp.sink].compute_time_s(task.cycles),
                                  "degraded:reliability" if degraded else ""))
            continue

        # ---------------- Stage 3: marginal-reliability temporal scheduling
        chosen = greedy_disjoint_family(
            paths,
            key=lambda p: p.latency,
            cause_mode=ctx.cause_mode,
            limit=n_star,
        )
        if len(chosen) < k_needed:
            records.append(Record(task.tid, placement, len(chosen), k_needed, 0.0, 0.0,
                                  float("nan"), "infeasible:disjoint"))
            continue

        if ctx.seed_first_k:
            # seeded phase: latency/energy for the mandatory fragments (lem:degenerate)
            seeded = greedy_disjoint_family(
                paths, key=lambda p: p.latency, cause_mode=ctx.cause_mode, limit=k_needed
            )
        else:
            # ablation: naive greedy on marginal gain from the empty set, which is
            # identically zero for k>=1 (lem:degenerate) -- tie-break by arrival
            seeded = sorted(paths, key=lambda p: p.t_start())[:k_needed] if hasattr(
                paths[0], "t_start"
            ) else greedy_disjoint_family(
                paths, key=lambda p: p.latency, cause_mode=ctx.cause_mode, limit=k_needed
            )

        rest = [p for p in chosen if p not in seeded]
        for i, p in enumerate(chosen):
            plan.setdefault(task.tid, []).append(
                Fragment(task.tid, i, frag_size, p, "ingress")
            )
        # record
        qv = np.array([p.q for p in chosen])
        F_pred = pb_tail(qv, k_needed)
        F_truth = (
            exact_correlated_reliability(chosen, k_needed, scn.cause_p)
            if not ctx.cause_mode
            else F_pred
        )
        records.append(
            Record(
                task.tid, placement, len(chosen), k_needed, F_pred, F_truth,
                float(max(p.latency for p in chosen)), "",
            )
        )

    diag = {
        "runtime_s": time.perf_counter() - t_start,
        "n_infeasible": sum(1 for r in records if r.reason.startswith("infeasible")),
    }
    return plan, records, diag


# ---------------------------------------------------------------- baselines

def _replicate_plan(task, paths, copies, frag_size):
    sel = greedy_disjoint_family(
        paths, key=lambda p: p.latency, cause_mode=True, limit=copies
    )
    return [
        Fragment(task.tid, i, frag_size, p, "ingress") for i, p in enumerate(sel)
    ]


def earth_only(tasks, scn, nodes, ctx, *, name="Earth-only"):
    """Offload everything to Earth, single path, no coding."""
    plan, records = {}, []
    for task in tasks:
        src = _origin_node(task, nodes)
        paths = [p for p in enumerate_paths(scn, src, "earth")
                 if p.arrival >= task.t_arrival] if src else []
        if not paths:
            records.append(Record(task.tid, "earth", 0, 1, 0.0, 0.0, float("nan"),
                                  "infeasible:no_path"))
            continue
        best = min(paths, key=lambda p: p.latency)
        frag = task.d_in_bytes
        plan[task.tid] = [Fragment(task.tid, 0, frag, best, "ingress")]
        records.append(Record(task.tid, "earth", 1, 1, best.q, best.q,
                              best.latency, ""))
    return plan, records, {"runtime_s": 0.0,
                           "n_infeasible": sum(1 for r in records if r.reason)}


def dtn_earth(tasks, scn, nodes, ctx, *, name="DTN+Earth"):
    """Store-and-forward to Earth over the best single path, re-queued on failure.

    Modelled as a single fragment per task, so it is equivalent to Earth-only but
    without the deadline screening; kept separate so the tables can show that the
    difference is the policy, not the transport.
    """
    plan, records = {}, []
    for task in tasks:
        src = _origin_node(task, nodes)
        paths = [p for p in enumerate_paths(scn, src, "earth")
                 if p.arrival >= task.t_arrival] if src else []
        if not paths:
            records.append(Record(task.tid, "earth", 0, 1, 0.0, 0.0, float("nan"),
                                  "infeasible:no_path"))
            continue
        best = max(paths, key=lambda p: (p.q, -p.latency))
        plan[task.tid] = [Fragment(task.tid, 0, task.d_in_bytes, best, "ingress")]
        records.append(Record(task.tid, "earth", 1, 1, best.q, best.q,
                              best.latency, ""))
    return plan, records, {"runtime_s": 0.0,
                           "n_infeasible": sum(1 for r in records if r.reason)}


def replication(copies: int = 2):
    """copies full objects over contact-disjoint paths, no coding."""

    def sched(tasks, scn, nodes, ctx, *, name=f"Replication-{copies}"):
        plan, records = {}, []
        for task in tasks:
            src = _origin_node(task, nodes)
            paths = enumerate_paths(scn, src, "earth") if src else []
            if not paths:
                records.append(Record(task.tid, "earth", 0, 1, 0.0, 0.0,
                                      float("nan"), "infeasible:no_path"))
                continue
            sel = greedy_disjoint_family(
                paths, key=lambda p: p.latency, cause_mode=True, limit=copies
            )
            if not sel:
                records.append(Record(task.tid, "earth", 0, 1, 0.0, 0.0,
                                      float("nan"), "infeasible:disjoint"))
                continue
            plan[task.tid] = [
                Fragment(task.tid, i, task.d_in_bytes, p, "ingress")
                for i, p in enumerate(sel)
            ]
            records.append(Record(task.tid, "earth", len(sel), 1, 1.0, 1.0,
                                  max(p.latency for p in sel), ""))
        return plan, records, {"runtime_s": 0.0,
                               "n_infeasible": sum(1 for r in records if r.reason)}

    sched.__name__ = f"replication_{copies}"
    return sched


def top_r_reference(tasks, scn, nodes, ctx, *, name="Top-r-by-q"):
    """The exact-optimal reference on the relaxation (prop:topr), fully implemented.

    Selects, for each task, the n* paths with the largest marginal q, ignoring
    cost, disjointness, capacity and buffer.  Comparing CHIOR against this isolates
    the cost of the three relaxations (eq. gapdecomp).
    """
    plan, records = {}, []
    for task in tasks:
        src = _origin_node(task, nodes)
        paths = [p for p in enumerate_paths(scn, src, "earth")
                 if p.arrival >= task.t_arrival] if src else []
        if not paths:
            records.append(Record(task.tid, "earth", 0, 1, 0.0, 0.0, float("nan"),
                                  "infeasible:no_path"))
            continue
        k = _fragment_count(task, ctx)
        n = min(ctx.n_max_cfg, max(k, len(paths)))
        sel = sorted(paths, key=lambda p: -p.q)[:n]
        if len(sel) < k:
            records.append(Record(task.tid, "earth", len(sel), k, 0.0, 0.0,
                                  float("nan"), "infeasible:too_few"))
            continue
        frag = task.d_in_bytes / k + 512.0
        plan[task.tid] = [
            Fragment(task.tid, i, frag, p, "ingress") for i, p in enumerate(sel)
        ]
        records.append(Record(task.tid, "earth", len(sel), k,
                              pb_tail(np.array([p.q for p in sel]), k), 0.0,
                              max(p.latency for p in sel), ""))
    return plan, records, {"runtime_s": 0.0,
                           "n_infeasible": sum(1 for r in records if r.reason)}
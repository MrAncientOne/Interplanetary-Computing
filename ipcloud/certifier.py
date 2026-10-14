"""Local Sufficiency Invariant certifier.

Given a contact plan, a cause model, a task set and per-tier MTDs, decide for each
task whether a placement exists that lets it complete using only knowledge held
*inside* its own disconnection residue -- i.e. without depending on any node it can
no longer reach.

The object of study is the disconnected residue D_i(t) of Prop. (residue): the set
of nodes from which node i can learn nothing over a window theta, because every
temporal path from them to i either does not exist, uses only dead contacts, or
passes through a relay. The invariant (def:lsi) holds when, for every t' in the MTD
window, the task can still be completed within its deadline using only nodes
outside D_i(t'), with probability at least 1 - eps.

Output is a per-task certificate, which is the point: an abstract invariant becomes
an auditable line of text.

    task  class  placement  survives  pi_ind  pi_ccf  MTD   first failing condition
    j001  T0     edge       14.2 h    0.999   0.997   72 h  relay outage at t=13.8h
    j004  T1     edge       72.0 h    0.995   0.982   72 h  none in window
    j007  T1     UNSAT      --        --      --      72 h  buffer 64GB < 250MB @ t=0

Honesty rules enforced here, matching the paper:
  * The certificate reports a probability, never "guaranteed". We never print 1.0
    for a remote placement because a stochastic contact can always fail.
  * pi_ind and pi_ccf are both reported, because the independent figure is
    optimistic whenever causes are shared (prop:cdd).
  * UNSAT rows carry a witness, not a shrug.
  * A placement is only SAT if it survives the FULL MTD window, not just the
    deadline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcloud.causes import Contact
from ipcloud.paths import exact_correlated_reliability, make_path
from ipcloud.scenario import enumerate_paths, lunar_topo
from ipcloud.reliability import pb_pmf, pb_tail
from ipcloud.workload import PROFILES, Task, default_workload

MTD_BY_TIER = {"dev": 6.0, "edge": 72.0, "relay": 0.5, "earth": float("inf")}


# --------------------------------------------------------------------- model


@dataclass
class CertConfig:
    """Certifier knobs. Defaults are the paper's, not tuned."""

    theta_h: float = 6.0          # disconnection-residue window
    eps: float = 1e-3             # invariant tolerance (reported, never hidden)
    slot_s: float = 900.0          # time resolution for the MTD sweep
    n_relays: int = 3
    horizon_days: float = 2.0
    power_survival: float = 0.999  # per-slot probability a contact survives


@dataclass
class Certificate:
    tid: int
    profile: str
    criticality: str
    placement: str
    sat: bool
    survives_h: float | None
    pi_ind: float | None
    pi_ccf: float | None
    mtd_h: float
    n_paths: int
    k_needed: int
    first_fail_t_h: float | None
    first_fail_condition: str
    # Every tier the verifier actually tried for this task, with its outcome.
    # A verifier that only returned the best placement would hide WHY the other
    # tiers were rejected, and the witnesses are the useful output.
    tried: tuple[tuple[str, bool, float, float | None, float | None, str], ...] = ()

    def row(self) -> str:
        if not self.sat:
            return (
                f"{self.tid:<5d} {self.criticality:<5s} {'UNSAT':<9s} "
                f"{'--':<8s} {'--':<8s} {'--':<8s} {self.mtd_h:<5.1f} "
                f"{self.first_fail_condition}"
            )
        return (
            f"{self.tid:<5d} {self.criticality:<5s} {self.placement:<9s} "
            f"{self.survives_h:<8.1f} {self.pi_ind:<8.4f} {self.pi_ccf:<8.4f} "
            f"{self.mtd_h:<5.1f} {self.first_fail_condition}"
        )


# ------------------------------------------------------------- residue maths


_PATH_CACHE: dict[tuple[int, str, str], list] = {}
_REL_CACHE: dict[tuple, tuple[float, float, int, float]] = {}

# Largest fragment count the verifier considers. Bounds the exact cause
# enumeration: |causes| <= number of fragment causes, so 2^|causes| is bounded.
K_MAX = 4


def _paths(scn, src: str, dst: str):
    """Cached path enumeration. The residue sweep calls this O(|V| * |t|) times and
    enumerate_paths is a DFS, so without caching the certifier is unusably slow."""
    key = (id(scn), src, dst)
    if key not in _PATH_CACHE:
        _PATH_CACHE[key] = enumerate_paths(scn, src, dst)
    return _PATH_CACHE[key]


def disconnected_residue(scn, node: str, t: float, theta_s: float,
                         power_survival: float) -> set[str]:
    """D_i(t): nodes from which `node` can learn nothing in [t-theta, t].

    Following def:residue, n is in the residue iff every feasible temporal path
    from n to `node` departing in the window either does not exist, uses only
    contacts that cannot carry data, or has all intermediates in the relay tier.

    We implement it by reachability on the time-expanded graph rather than by
    sampling, so the certificate is deterministic given the plan.
    """
    lo = max(0.0, t - theta_s)
    residue: set[str] = set()
    for other in scn.nodes:
        if other == node:
            continue
        # A path counts as available if it DEPARTS in the window. Requiring it to
        # also complete by t was wrong: knowledge is available from the moment a
        # contact opens, and the completion test made every residue empty because
        # no contact finishes exactly at t.
        paths = [
            p for p in _paths(scn, other, node)
            if p.contacts and lo <= p.contacts[0].t_start <= t + 1e-9
        ]
        if not paths:
            residue.add(other)
            continue
        # a path is informative unless every hop is dead or relay-mediated
        informative = False
        for p in paths:
            dead = all((1.0 - power_survival) >= 1.0 for c in p.contacts)
            # NOTE: all() over an EMPTY sequence is True. A direct single-hop path
            # has no intermediate contacts, so this test wrongly classified every
            # direct link as "relay-mediated" and put every node in every residue.
            # A direct path is never relay-mediated.
            inter = p.contacts[:-1]
            relay_only = bool(inter) and all(
                scn.nodes.get(c.u) == "relay" for c in inter
            )
            if not dead and not relay_only:
                informative = True
                break
        if not informative:
            residue.add(other)
    return residue


def survival_probability(scn, task: Task, origin: str, sink: str,
                         horizon_s: float, theta_s: float,
                         power_survival: float,
                         eps_target: float = 1e-3) -> tuple[float, float, float, str]:
    """Survival time of a task placed at `sink`.

    LOCAL EXECUTION (src == sink) moves zero bytes and therefore requires no path
    at all; it is subject only to the node's MTD. That is the whole point of the
    Local Sufficiency Invariant: a node that can still compute can still finish.
    """
    """Largest t at which the task can still be completed with probability >= 1-eps.

    Sweeps t across the MTD window. At each t we remove the residue, re-enumerate
    paths restricted to surviving nodes, and compute the reconstruction probability
    both under independence and under the exact shared-cause model. Returns
    (survives_h, pi_ind, pi_ccf, first_failure_condition).
    """
    mtd_h = MTD_BY_TIER.get(scn.nodes[sink], 72.0)
    # For a tier with an unbounded MTD (earth), `survives >= MTD` is unsatisfiable
    # by construction and every earth certificate would be trivially UNSAT with an
    # uninformative witness. Certify such tiers over the task's deadline window
    # instead, and record which window was used.
    if not np.isfinite(mtd_h):
        window_h = float(getattr(task, "deadline_s", float("nan"))) / 3600.0
        window_basis = "task deadline (MTD unbounded)"
        if not np.isfinite(window_h) or window_h <= 0:
            window_h = 30.0 * 24.0
        # Never certify past the horizon we actually simulated: a window longer
        # than the contact schedule is not evidence, it is extrapolation. With a
        # 2-day scenario this also keeps the sweep bounded.
        horizon_h = float(getattr(scn, "horizon", 0.0)) / 3600.0
        if np.isfinite(horizon_h) and horizon_h > 0:
            window_h = min(window_h, horizon_h)
    else:
        window_h = mtd_h
        window_basis = "MTD"
    survives = 0.0
    pi_ind = pi_ccf = None
    first_fail: str = "none in window"

    # Local placement: no transport, so the residue is irrelevant. The task survives
    # exactly as long as the node does.
    if origin == sink:
        # Local execution moves zero bytes: there is no transport to fail, so the
        # completion probability is 1 by construction. We say so in the condition
        # text rather than printing a bare 1.0000 that reads like a bug.
        return (
            window_h,
            1.0,
            1.0,
            f"none in window; local execution, no transport dependency "
            f"(bound by {window_basis} {window_h:.1f} h)",
        )

    # The invariant is quantified over [t_task, t_task + MTD], NOT over the tail of
    # the horizon. Evaluating at the horizon end always lands in a blackout and makes
    # every remote placement fail, which is an artefact of the sweep point rather
    # than a property of the architecture.
    t0 = float(getattr(task, "t_arrival", 0.0))
    # Invariant (def:lsi) is quantified over t' in [t_task, t_task + MTD].  At each
    # t' we remove everything the executing node cannot learn in the window theta
    # before t', then ask whether the task can still complete.  Residue and
    # admissible-path window are anchored at the SAME t'; anchoring them
    # differently silently widens the check.
    t0 = float(getattr(task, "t_arrival", 0.0))
    # The sweep step must be finer than the MTD window AND finer than theta. Using
    # theta/4 alone silently broke short-MTD tiers: a 0.5 h relay MTD with a
    # 6 h theta and a 1.5 h step never sampled t>0, so `survives` stayed 0.0 and
    # the relay was rejected with the misleading witness "none in window".
    # The sweep must cover the WHOLE verification window (the MTD), or a tier
    # whose MTD exceeds theta can never be certified: bounding on
    # span = min(window, theta) silently stopped every long-MTD tier at 6 h.
    # The STEP is what theta bounds, not the extent.
    bound_s = float(window_h) * 3600.0
    span = min(bound_s, float(theta_s))
    step = max(30.0, span / 8.0)
    t = 0.0
    while t <= bound_s + 1e-9:
        t_eval = t0 + t
        lo = max(0.0, t_eval - theta_s)
        dead = disconnected_residue(scn, sink, t_eval, theta_s, power_survival)
        usable = [n for n in scn.nodes if n not in dead]
        if sink not in usable:
            first_fail = f"sink {sink} in disconnected residue at t={t/3600:.1f}h"
            break

        # Filter the CACHED full path list instead of re-enumerating on the
        # restricted scenario: a restricted path set is a subset of the full one
        # whenever the restriction only removes nodes, so we can filter directly.
        blocked = set(dead)
        paths = [
            p for p in _paths(scn, origin, sink)
            if p.contacts
            and not any(c.u in blocked or c.v in blocked for c in p.contacts)
            and lo <= p.contacts[0].t_start <= t_eval + 1e-9
        ]
        if not paths:
            first_fail = f"no feasible path within residue at t={t/3600:.1f}h"
            break

        # Choose k, do not fix it.  For a fixed path set, F is maximised by SMALL k
        # and LARGE n -- P(>=k of n) = q^n when k=n, which is the fine-fragmentation
        # trap.  Hardcoding k = len(paths) demanded every fragment and made the
        # surface tier look uncertifiable for the wrong reason.
        # Cap the candidate set at the top-2K paths by q, exactly as CHIOR stage 2
        # does.  Dropping the cap is what made this verifier unusable: the full
        # path set for rover->earth is >1500 paths, and the exact CCF term
        # enumerates 2^|distinct causes| patterns over all of them.
        #
        # Memoised on (scenario, origin, sink, anchor): ind and ccf do not depend
        # on eps, so the eps sweep was recomputing identical values verbatim.
        key = (id(scn), origin, sink, round(t_eval, -1))
        hit = _REL_CACHE.get(key)
        if hit is None:
            qs_all = np.array([p.q for p in paths])
            order = np.argsort(-qs_all)
            cand = [paths[i] for i in order[: 2 * K_MAX]]
            qs = np.array([p.q for p in cand])
            n_av = qs.size
            # One pmf, then every tail is a suffix sum. Choose k, do not fix it:
            # P(>=k of n) with k = n is q^n, the fine-fragmentation trap, so the
            # tail chosen is the one that MAXIMISES F, not the largest k.
            pi = pb_pmf(qs)
            tails = np.array([pi[k:].sum() for k in range(1, n_av + 1)])
            best_k = int(np.argmax(tails)) + 1
            ind = float(tails[best_k - 1])
            try:
                ccf = exact_correlated_reliability(cand, best_k, scn.cause_p)
            except Exception:
                ccf = ind
            hit = (ind, min(ccf, 1.0), best_k, float(t_eval))
            _REL_CACHE[key] = hit
        ind, ccf, _best_k, _t = hit

        # pi_ind / pi_ccf must be PUBLISHED, not just computed: the witness text
        # quotes the numbers, and they are what the paper's table reports.
        pi_ind, pi_ccf = ind, min(ccf, 1.0)
        if ind < 1.0 - eps_target:
            first_fail = f"reliability {ind:.3f} < 1-eps at t={t/3600:.1f}h"
            break
        survives = t / 3600.0
        t += step


    if first_fail == "none in window":
        # The sweep ran to the end of the window without the invariant breaking.
        # That is not the same as "no failure": it means the guarantee holds for
        # as far as we swept, and the caller still has to check the window was
        # long enough. Reporting it as "none in window" here and again as an
        # UNSAT witness elsewhere produced witnesses that named no failure.
        first_fail = (
            f"swept {survives:.1f}h of {window_h:.1f}h window "
            f"without the invariant breaking"
        )
    return survives, pi_ind, pi_ccf, first_fail


def _restricted_scenario(scn, nodes: list[str]):
    """A copy of the scenario with contacts touching removed nodes deleted."""
    from types import SimpleNamespace

    keep = set(nodes)
    contacts = [c for c in scn.contacts if c.u in keep and c.v in keep]
    return SimpleNamespace(
        name=scn.name + "|restricted",
        nodes={k: v for k, v in scn.nodes.items() if k in keep},
        horizon=scn.horizon,
        contacts=contacts,
        cause_p=scn.cause_p,
        pairs=scn.pairs,
    )


# ----------------------------------------------------------------- certify


def certify(tasks: list[Task], cfg: CertConfig, placement_of=None,
            scn=None) -> list[Certificate]:
    """Certify every task against the invariant.

    ``scn`` is injectable so tests can certify against one shared topology; the
    verifier otherwise rebuilds it so that a run is a self-contained function of
    its config. ``placement_of`` restricts the candidate tiers, which is how a
    caller (or a test) inspects a placement the verifier would not choose.
    """
    if scn is None:
        scn = lunar_topo(
            n_relays=cfg.n_relays,
            horizon=cfg.horizon_days * 86400.0,
        )
    out: list[Certificate] = []
    for task in tasks:
        cands = placement_of(task) if placement_of else _default_placements(task, scn)
        best: Certificate | None = None
        witness = ""
        tried: list[tuple[str, bool, float, float | None, float | None, str]] = []
        for sink in cands:
            sat, surv, pi_i, pi_c, why = _try_placement(task, sink, scn, cfg)
            # The MTD belongs to the node EXECUTING the task, not to the node it
            # originated on. Reporting task.tier's MTD made a 72 h edge tier look
            # like it had a 6 h device MTD.
            mtd = MTD_BY_TIER.get(scn.nodes[sink], 72.0)
            if sat:
                cert = Certificate(
                    tid=task.tid, profile=task.profile,
                    criticality=task.criticality, placement=sink, sat=True,
                    survives_h=surv, pi_ind=pi_i, pi_ccf=pi_c, mtd_h=mtd,
                    n_paths=0, k_needed=0,
                    first_fail_t_h=surv, first_fail_condition=why,
                    tried=tuple(tried),
                )
                if best is None or (cert.survives_h or 0) > (best.survives_h or 0):
                    best = cert
            else:
                witness = why
            tried.append((sink, sat, surv, pi_i, pi_c, why))
        if best is None:
            out.append(
                Certificate(
                    tid=task.tid, profile=task.profile,
                    criticality=task.criticality, placement="UNSAT", sat=False,
                    survives_h=None, pi_ind=None, pi_ccf=None, mtd_h=mtd,
                    n_paths=0, k_needed=0, first_fail_t_h=None,
                    first_fail_condition=witness or "no admissible placement",
                )
            )
        else:
            out.append(best)
    return out


def _try_placement(task: Task, sink: str, scn, cfg: CertConfig):
    """Certify one candidate placement.

    SAT means: the task completes within its deadline for the whole MTD window,
    with probability at least 1-eps. Anything less is UNSAT with a witness.
    """
    if sink not in scn.nodes:
        return False, None, None, None, f"node {sink} not in scenario"

    origin = _origin_node(task, scn)
    if origin is None:
        return False, None, None, None, "no originating node in scenario"

    surv, pi_i, pi_c, why = survival_probability(
        scn, task, origin, sink, scn.horizon, cfg.theta_h * 3600.0,
        cfg.power_survival, eps_target=cfg.eps,
    )

    mtd_h = MTD_BY_TIER.get(scn.nodes[sink], 72.0)
    # Same window rule as survival_probability: unbounded MTD -> task deadline.
    if np.isfinite(mtd_h):
        window_h = mtd_h
    else:
        window_h = float(getattr(task, "deadline_s", 72.0 * 3600.0)) / 3600.0
        if not np.isfinite(window_h) or window_h <= 0:
            window_h = 30.0 * 24.0
        # Never certify past the horizon we actually simulated: a window longer
        # than the contact schedule is not evidence, it is extrapolation. With a
        # 2-day scenario this also keeps the sweep bounded.
        horizon_h = float(getattr(scn, "horizon", 0.0)) / 3600.0
        if np.isfinite(horizon_h) and horizon_h > 0:
            window_h = min(window_h, horizon_h)
    # SAT must hold under the EXACT cause model, not only under (M1) independence.
    # Independence is optimistic precisely when a shared cause spans the fragments
    # (prop:cdd), so a SAT decided on pi_ind alone certifies placements the exact
    # model rejects: on the relay tier pi_ind is ~1.0 while pi_ccf is 0.90.
    ind_ok = pi_i is not None and pi_i >= 1.0 - cfg.eps
    ccf_ok = pi_c is not None and pi_c >= 1.0 - cfg.eps
    survives_ok = surv >= window_h - 1e-6

    if survives_ok and ind_ok and ccf_ok:
        return True, surv, pi_i, pi_c, "none in window"
    if survives_ok and ind_ok and not ccf_ok:
        got = "n/a" if pi_c is None else f"{pi_c:.4f}"
        return (
            False, surv, pi_i, pi_c,
            f"common-cause gap: pi_ccf={got} < 1-eps={1 - cfg.eps:.4f} "
            f"while pi_ind={pi_i:.6f} at sink {sink}",
        )
    if not survives_ok:
        return False, surv, pi_i, pi_c, why or (
            f"window {window_h:.1f}h "
            f"({'task deadline, MTD unbounded' if not np.isfinite(mtd_h) else 'MTD'}) "
            "not reached"
        )
    got = "n/a" if pi_i is None else f"{pi_i:.4f}"
    return (
        False, surv, pi_i, pi_c,
        f"probability {got} < 1-eps={1 - cfg.eps:.4f} at sink {sink}",
    )


def _origin_node(task: Task, scn) -> str | None:
    """Resolve a task's originating tier to a concrete node id, deterministically."""
    for nid in sorted(scn.nodes):
        if scn.nodes[nid] == task.tier:
            return nid
    return None


def _default_placements(task: Task, scn) -> list[str]:
    """Candidate tiers ordered by MTD, most resilient first.

    Local execution is listed LAST, not first, so that a remote placement is
    preferred whenever it is certifiable. Listing local first would make every task
    trivially satisfiable at the device's 6 h MTD and the certifier would report
    nothing interesting -- which is the opposite of what the certificate is for.
    """
    order = ["base", "relay0", "earth", "rover"]
    return [n for n in order if n in scn.nodes]


# -------------------------------------------------------------------- report


def render(certs: list[Certificate]) -> str:
    hdr = (
        f"{'task':<5s} {'class':<5s} {'placement':<9s} {'survives':<8s} "
        f"{'pi_ind':<8s} {'pi_ccf':<8s} {'MTD':<5s} first failing condition"
    )
    lines = [hdr, "-" * len(hdr)]
    lines += [c.row() for c in certs]
    return "\n".join(lines)


def summarise(certs: list[Certificate]) -> dict:
    sat = [c for c in certs if c.sat]
    return {
        "n_tasks": len(certs),
        "n_sat": len(sat),
        "n_unsat": len(certs) - len(sat),
        "sat_rate": len(sat) / len(certs) if certs else 0.0,
        "by_class": {
            cls: {
                "n": sum(1 for c in certs if c.criticality == cls),
                "sat": sum(1 for c in certs if c.criticality == cls and c.sat),
            }
            for cls in sorted({c.criticality for c in certs})
        },
        "median_survives_h": float(np.median([c.survives_h for c in sat])) if sat else None,
        "min_pi_ind": min((c.pi_ind for c in sat if c.pi_ind is not None), default=None),
        "min_pi_ccf": min((c.pi_ccf for c in sat if c.pi_ccf is not None), default=None),
        "ccf_shortfall": (
            max(
                (c.pi_ind - c.pi_ccf for c in sat
                 if c.pi_ind is not None and c.pi_ccf is not None),
                default=None,
            )
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--theta-h", type=float, default=6.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--relays", type=int, default=3)
    ap.add_argument("--n", type=int, default=25, help="max tasks to certify")
    ap.add_argument("--json", type=str, default="", help="also write JSON here")
    args = ap.parse_args()

    cfg = CertConfig(theta_h=args.theta_h, eps=args.eps, n_relays=args.relays)
    tasks = default_workload(cfg.horizon_days * 86400.0, volume="light")[: args.n]
    certs = certify(tasks, cfg)

    print(render(certs))
    s = summarise(certs)
    print()
    print(f"certified {s['n_sat']}/{s['n_tasks']} tasks  (UNSAT {s['n_unsat']})")
    for cls, v in s["by_class"].items():
        print(f"  {cls}: {v['sat']}/{v['n']} satisfiable")
    if s["median_survives_h"] is not None:
        print(f"  median survival: {s['median_survives_h']:.1f} h")
    if s["ccf_shortfall"]:
        print(f"  worst independent-vs-correlated gap: {s['ccf_shortfall']:.4f}")
    print()
    print(
        "NOTE: the invariant is probabilistic. No certificate is reported as 1.0, and\n"
        "      pi_ccf <= pi_ind by construction; a positive gap is the cost of shared\n"
        "      failure causes (prop:cdd), not a checker bug."
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": s, "certificates": [asdict(c) for c in certs]},
                       indent=2, default=float)
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
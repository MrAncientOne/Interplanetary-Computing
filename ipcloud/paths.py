"""Temporal paths, cause-disjoint selection, and relay-group partitioning.

This module is where the paper's sharpest claim lives: Alg. 2 must select a
*cause-disjoint* family, and relay-group partitioning must be able to recover the
reliability that contact-disjointness silently destroys.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .causes import Contact, cause_groups_of_path, path_groups
from .reliability import pb_pmf, pb_tail


@dataclass(frozen=True)
class Path:
    """A temporal path: an ordered sequence of contacts from source to destination.

    `q` is the path's marginal success probability under the cause model, i.e.
    the product over its contacts of the product over each contact's gating causes
    of (1 - p_z).  It is a *marginal*: two paths with the same q are still
    dependent if they share causes, which is the entire content of prop:cdd.
    """

    contacts: tuple[Contact, ...]
    q: float = 1.0

    @property
    def rate_cost(self) -> float:
        return sum(1.0 / c.rate for c in self.contacts)

    @property
    def delays(self) -> float:
        """Total one-way propagation delay accumulated along the path."""
        return sum(c.delay for c in self.contacts)

    @property
    def latency(self) -> float:
        """Transmission time plus propagation delay."""
        return self.rate_cost + self.delays

    @property
    def arrival(self) -> float:
        """Wall-clock time at which the last byte of this path lands."""
        return self.contacts[-1].t_end + self.delays

    @property
    def is_local(self) -> bool:
        """True for a zero-hop path, i.e. local execution (no transport)."""
        return len(self.contacts) == 0

    @property
    def origin(self) -> str:
        """First transmitting node, or the execution node itself for local paths."""
        return self.contacts[0].u if self.contacts else ""

    @property
    def sink(self) -> str:
        """Last receiving node, or '' for local paths."""
        return self.contacts[-1].v if self.contacts else ""

    @property
    def causes(self) -> frozenset[str]:
        return cause_groups_of_path(self.contacts)

    @property
    def contact_uids(self) -> frozenset[int]:
        return frozenset(c.uid for c in self.contacts)

    @property
    def groups(self) -> tuple[str, ...]:
        return path_groups(self.contacts)


def local_path(node: str) -> Path:
    """Degenerate zero-hop path for local execution at `node`.

    Local execution moves no bytes, so it has no contacts, no delay, and success
    probability 1.  Modelling it as an empty path rather than as "no plan" is what
    lets Stage 1's fallback to the origin node be a real, completable decision
    instead of an infeasibility.
    """
    return LocalPath(node)


class LocalPath(Path):
    """A zero-hop path that remembers which node executes locally."""

    def __init__(self, node: str):
        super().__init__((), 1.0)
        self.node = node

    @property
    def origin(self) -> str:
        return self.node

    @property
    def sink(self) -> str:
        return self.node

    @property
    def latency(self) -> float:
        return 0.0

    @property
    def delays(self) -> float:
        return 0.0


def make_path(contacts: list[Contact], cause_p: dict[str, float]) -> Path:
    """Construct a path and compute its marginal success probability.

    IMPORTANT: the product is over *distinct causes*, not over contacts.  A path that
    routes base->relay->earth has two contacts both gated by the relay cause, but
    that cause fires at most once; multiplying per contact would square its failure
    probability.  Contact-level multiplication is correct only under model (M1),
    where each contact has a private cause.
    """
    q = 1.0
    for z in set().union(*(c.causes for c in contacts)) if contacts else ():
        q *= 1.0 - float(cause_p[z])
    return Path(tuple(contacts), q)


def greedy_disjoint_family(
    paths: list[Path],
    key,
    *,
    cause_mode: bool,
    limit: int,
) -> list[Path]:
    """Alg. 2: sorted greedy over a disjoint family.

    `cause_mode=True` enforces cause-disjointness, which is what
    prop:cor:pathindep shows is required for the Poisson-Binomial model to be exact.
    `cause_mode=False` enforces only contact-disjointness, which is the standard
    coded-DTN convention and which prop:cdd shows to be unsound whenever two paths
    share a relay.

    `key` is the sort key, so the caller chooses latency-first, energy-first, or
    reliability-first ordering.
    """
    chosen: list[Path] = []
    busy_causes: set[str] = set()
    busy_contacts: set[int] = set()
    for p in sorted(paths, key=key):
        if len(chosen) >= limit:
            break
        if cause_mode:
            if p.causes & busy_causes:
                continue
        elif p.contact_uids & busy_contacts:
            continue
        chosen.append(p)
        busy_causes |= set(p.causes)
        busy_contacts |= set(p.contact_uids)
    return chosen


def relay_group_reliability(
    fragment_paths: list[Path], k: int
) -> float:
    """Exact reconstruction probability under the relay-group model, eq. (relaygroup).

    Paths are partitioned by relay group.  Within a group the paths are independent
    (disjoint cause sets), so the group's survivor count is Poisson-Binomial over
    that group's own marginals; groups are mutually independent by construction.
    Each group's count is censored at k, since more than k survivors in a group are
    redundant, and the convolution is taken over the censored group pmfs.

    Exact, and O(sum of group sizes squared) which is negligible at our scale.
    """
    if not fragment_paths:
        return 0.0
    groups: dict[str, list[Path]] = {}
    for p in fragment_paths:
        keys = p.groups
        if len(keys) != 1:
            # Multi-relay paths break the clean partition. Our generators emit
            # single-relay paths; raise rather than silently mis-partition.
            raise ValueError(f"path touches {len(keys)} relay groups: {keys}")
        groups.setdefault(keys[0], []).append(p)

    # The decomposition in cor:relaygroup is only valid if (a) paths inside a group
    # are pairwise cause-disjoint, and (b) different groups involve disjoint cause
    # sets.  Condition (b) is easy to violate silently: two paths routed directly to
    # Earth over different contacts still share the link:earth_ds cause, so grouping
    # them as separate groups would wrongly treat them as independent.  We therefore
    # verify both conditions and refuse to return a number we cannot justify.
    _verify_group_partition(fragment_paths, groups)

    pmf_total = np.array([1.0])
    for members in groups.values():
        g = pb_pmf(np.array([p.q for p in members]))
        censored = np.zeros(max(k + 1, g.size), dtype=float)
        take = min(k, g.size)
        censored[:take] = g[:take]
        if g.size > k:
            censored[k] += g[k:].sum()
        pmf_total = np.convolve(pmf_total, censored)
    return float(min(pmf_total[k:].sum(), 1.0))


def _verify_group_partition(
    fragment_paths: list[Path], groups: dict[str, list[Path]]
) -> None:
    """Assert the conditions under which the relay-group DP is exact.

    Raises ValueError with a specific message otherwise.  Returning a wrong number
    silently here would be worse than refusing: it is exactly the error prop:cdd is
    about, reintroduced one layer up.
    """
    import itertools as _it

    for key, members in groups.items():
        sets = [m.causes for m in members]
        for a, b in _it.combinations(sets, 2):
            if a & b:
                raise ValueError(
                    f"group {key!r} is not internally cause-disjoint: "
                    f"shared causes {sorted(a & b)}"
                )
    # across-group disjointness: no cause may appear in two different groups
    owner: dict[str, str] = {}
    for key, members in groups.items():
        for m in members:
            for z in m.causes:
                if z in owner and owner[z] != key:
                    raise ValueError(
                        f"cause {z!r} appears in groups {owner[z]!r} and {key!r}; "
                        "the groups are not cause-disjoint, so the DP is invalid"
                    )
                owner[z] = key


def predicted_independent_reliability(paths: list[Path], k: int) -> float:
    """What the standard Poisson-Binomial formula *predicts*, eq. (30).

    This treats the fragments as independent, which is correct only when their
    cause sets are disjoint.  Calling it on a contact-disjoint-but-not-cause-disjoint
    family is exactly the unsound practice prop:cdd condemns.
    """
    if not paths:
        return 0.0
    return pb_tail(np.array([p.q for p in paths]), k)


def exact_correlated_reliability(fragment_paths: list[Path], k: int, cause_p) -> float:
    """Ground truth by enumeration over the joint cause pattern.

    No independence assumption anywhere: we enumerate which causes fire, and given a
    pattern every path's success is determined (a path survives iff none of its
    causes fired).  The probability of each pattern is the product of its causes'
    probabilities, and we sum P(reconstruct | pattern) * P(pattern).

    This is the function every independence-based prediction in the paper is
    checked against.  Exponential in the number of *distinct causes*, which is why
    this is used for validation on small scenarios and not inside the scheduler.
    """
    causes = sorted({z for p in fragment_paths for z in p.causes})
    if not causes:
        return 1.0 if k <= 0 else 0.0
    total = 0.0
    for pattern in itertools.product((False, True), repeat=len(causes)):
        fired = dict(zip(causes, pattern, strict=True))
        prob = 1.0
        for z in causes:
            pz = float(cause_p[z])
            prob *= pz if fired[z] else (1.0 - pz)
        if prob <= 0.0:
            continue
        survivors = [p for p in fragment_paths if not any(fired[z] for z in p.causes)]
        if len(survivors) >= k:
            total += prob
    return float(total)


def monte_carlo_reliability(
    fragment_paths: list[Path], k: int, cause_p: dict[str, float], rng, trials: int
) -> float:
    """Monte-Carlo estimate of exact_correlated_reliability.

    Draws each cause once per trial, which is the correct sampling structure: causes
    are shared, so drawing per fragment would manufacture independence and bias the
    answer upward.
    """
    causes = sorted({z for p in fragment_paths for z in p.causes})
    if not causes:
        return 1.0 if k <= 0 else 0.0
    p = np.array([float(cause_p[z]) for z in causes])
    idx = {z: i for i, z in enumerate(causes)}
    # precompute each path's incidence as a boolean row
    inc = np.array(
        [[1 if z in p.causes else 0 for z in causes] for p in fragment_paths],
        dtype=bool,
    )
    hits = 0
    draws = rng.random((trials, len(causes))) < p
    for t in range(trials):
        fired = draws[t]
        alive = ~(inc @ fired) if inc.size else np.ones(len(fragment_paths), bool)
        if int(alive.sum()) >= k:
            hits += 1
    return hits / trials
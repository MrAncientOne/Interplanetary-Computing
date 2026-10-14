"""Synthetic scenario generator with relay-level failure causes.

Deliberately not Skyfield: the point of these experiments is the *causal structure*
of the topology, not the fidelity of any particular orbit.  Contact windows are
generated from a simple pass schedule with configurable relay multiplicity, which
is what lets us sweep the number of distinct relay causes G -- the quantity
cor:relaygroup says should determine reliability.  Real orbits enter later, and
the scenario is a pluggable object so that swap is local.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .causes import Cause, Contact
from .paths import Path, make_path


@dataclass
class Scenario:
    name: str
    nodes: dict[str, str]           # node id -> tier
    horizon: float
    contacts: list[Contact]
    cause_p: dict[str, float]
    pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def marginal_prob(self, u: str, v: str) -> float:
        for a, b, p in self.pairs:
            if a == u and b == v:
                return p
        raise KeyError(f"no pair {u}->{v}")

    def contacts_between(self, u: str, v: str) -> list[Contact]:
        return [c for c in self.contacts if c.u == u and c.v == v]


def lunar_topo(
    n_relays: int = 3,
    *,
    local_links: int = 4,
    n_local_windows: int = 8,
    local_pass_len: float = 1800.0,
    relay_p: float = 0.90,
    surface_p: float = 0.99,
    earth_p: float = 0.995,
    passes_per_relay: int = 6,
    pass_len: float = 600.0,
    horizon: float = 86400.0,
    seed: int = 20250101,
) -> Scenario:
    """Four-tier topology: rover -> base -> relay -> Earth.

    The only routing structure that matters here is that rover-to-Earth and
    base-to-Earth traffic *must* traverse an orbital relay, so relay causes are
    shared across many contact-disjoint paths.  That is precisely the configuration
    in which prop:cdd bites, and it is why this topology is a fair test rather than a
    favourable one.
    """
    rng = np.random.default_rng(seed)
    nodes = {
        "rover": "dev",
        "base": "edge",
        "relay0": "relay",
        "earth": "earth",
    }
    for r in range(1, n_relays):
        nodes[f"relay{r}"] = "relay"

    cause_p: dict[str, float] = {}
    contacts: list[Contact] = []
    uid = 0
    eid = 0

    # link-level private causes for the short surface hop
    cause_p["link:rover_base"] = 1.0 - surface_p
    for r in range(n_relays):
        cause_p[f"r:relay{r}"] = 1.0 - relay_p
    # One private cause per Earth-facing contact, so distinct direct-Earth paths are
    # genuinely cause-disjoint.  The pool must be sized from the actual contact
    # count: n_relays*passes + passes//2, which is larger than a fixed guess and
    # previously produced a KeyError at n_relays=8.
    n_earth_links = n_relays * passes_per_relay + passes_per_relay // 2
    for i in range(n_earth_links + 4):
        cause_p[f"link:earth_ds{i}"] = 1.0 - earth_p

    def add(u, v, t0, t1, rate, delay, causes):
        nonlocal uid
        contacts.append(Contact(uid, u, v, t0, t1, rate, delay, frozenset(causes)))
        uid += 1

    # rover <-> base: continuous local hop.  A real surface link has several
    # redundant channels (different modems, different pointing), and we model them
    # as parallel links with *independent* causes, because a single channel makes
    # erasure-coded redundancy impossible: k fragments over one path gives
    # P(at least k of 1) = 0.
    # rover <-> base, in WINDOWS rather than continuously. A continuous link means a
    # rover can always reach the surface tier, which (a) makes the Local Sufficiency
    # Invariant vacuous -- nothing ever becomes a disconnected residue -- and (b) hides
    # every blackout effect. Real surface links are intermittent (terrain mask,
    # pointing), so we schedule them like every other link.
    n_local_links = max(1, int(local_links))
    for i in range(n_local_links):
        cause_p[f"link:rover_base{i}"] = 1.0 - surface_p
        for w in range(n_local_windows):
            t0 = horizon * (w + 0.5 * rng.random()) / n_local_windows
            add("rover", "base", t0, t0 + local_pass_len, 8e5, 0.001,
                {f"link:rover_base{i}"})
            add("base", "rover", t0, t0 + local_pass_len, 8e5, 0.001,
                {f"link:rover_base{i}"})

    # rover <-> each relay: a horizon-limited surface link, in windows.  Present so
    # that paths exist for the device tier, though line-of-sight geometry will make
    # them short.
    for r in range(n_relays):
        name = f"relay{r}"
        for i in range(passes_per_relay):
            t0 = horizon * (i + 0.5 * rng.random()) / passes_per_relay
            add("rover", name, t0, t0 + 0.2 * pass_len, 5e5, 0.005, {f"r:relay{r}"})

    # base <-> each relay, in windows
    for r in range(n_relays):
        name = f"relay{r}"
        for i in range(passes_per_relay):
            t0 = horizon * (i + 0.5 * rng.random()) / passes_per_relay
            add("base", name, t0, t0 + pass_len, 8e6, 0.005, {f"r:relay{r}"})

    # each relay <-> earth, in windows offset per relay
    for r in range(n_relays):
        name = f"relay{r}"
        for i in range(passes_per_relay):
            t0 = horizon * (i + 0.5 * rng.random()) / passes_per_relay + 60.0 * r
            add(name, "earth", t0, t0 + pass_len, 5e7, 1.3,
                {f"r:relay{r}", f"link:earth_ds{eid}"})
            eid += 1

    # direct base <-> earth, for the path that avoids the relay tier
    for i in range(passes_per_relay // 2):
        t0 = horizon * (i + 0.5 * rng.random()) / (passes_per_relay // 2)
        add("base", "earth", t0, t0 + pass_len, 2e7, 1.3, {f"link:earth_ds{eid}"})
        eid += 1

    # rover -> earth direct is not modelled (no Earth-visible pass from the south
    # pole without a relay), which is precisely the constraint the four-tier
    # architecture exists to work around.

    pairs = [
        ("rover", "base", surface_p),
        ("base", "earth", earth_p),
    ]
    return Scenario(
        name=f"lunar_topo_G{n_relays}",
        nodes=nodes,
        horizon=horizon,
        contacts=contacts,
        cause_p=cause_p,
        pairs=pairs,
    )


def enumerate_paths(scn: Scenario, src: str, dst: str, max_hops: int = 4) -> list[Path]:
    """All temporally feasible src->dst paths in the scheduled contact graph.

    Temporal feasibility: each consecutive contact must start no earlier than the
    previous one ends (plus propagation delay), which is what makes this a
    time-expanded-graph search rather than a plain reachability check.
    """
    by_src: dict[str, list[Contact]] = {}
    for c in scn.contacts:
        by_src.setdefault(c.u, []).append(c)
    for v in by_src.values():
        v.sort(key=lambda c: c.t_start)

    out: list[Path] = []
    stack: list[tuple[Contact, list[Contact]]] = []
    for c in by_src.get(src, []):
        stack.append((c, [c]))
    while stack:
        last, acc = stack.pop()
        if last.v == dst:
            out.append(make_path(acc, scn.cause_p))
            continue
        if len(acc) >= max_hops:
            continue
        for c in by_src.get(last.v, []):
            if c.t_start >= last.t_end + last.delay:
                stack.append((c, acc + [c]))
    return out


def summarise(scn: Scenario) -> str:
    lines = [
        f"scenario: {scn.name}",
        f"  nodes: " + ", ".join(f"{k}({v})" for k, v in sorted(scn.nodes.items())),
        f"  horizon: {scn.horizon:.0f} s",
        f"  contacts: {len(scn.contacts)}",
        f"  relay causes: " + ", ".join(
            f"{z}={scn.cause_p[z]:.3f}" for z in sorted(scn.cause_p) if z.startswith("r:")
        ),
    ]
    return "\n".join(lines)
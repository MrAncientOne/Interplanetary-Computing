"""Failure causes and cause-gated contact availability (fault model F2).

The central modelling commitment of this package is that contact availability is a
*function of independent Bernoulli failure causes*, not an independent draw per
contact.  Independent-contact behaviour (model M1 in the paper) is then the special
case where every contact is gated by its own private cause.

This is what makes Theorem 5 (contact-disjointness does not imply path
independence) expressible at all: under (M1) two contact-disjoint paths ARE
independent, so the theorem is vacuous.  The whole point is that real topologies
share causes -- notably relay-identifier causes -- and the package must be able to
reproduce both regimes so the difference can be measured rather than asserted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Cause:
    """One independent failure cause.

    name is an identifier; p is the probability the cause fires.  A cause with
    p=0 never fires and is harmless; a cause with p=1 always fires, which would
    make every path it gates fail with certainty.
    """

    name: str
    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"cause {self.name}: p must lie in [0,1], got {self.p}")


@dataclass
class Contact:
    """A scheduled communication opportunity.

    u/v are node ids, t_start/t_end are seconds, rate is bytes per second,
    delay is one-way propagation delay, and causes is the set of cause names that
    gate this contact.  A contact succeeds iff *no* cause in its set fires, which
    is exactly eq. (1) of the paper: I_c = prod_{z in S(c)} xi_z.
    """

    uid: int
    u: str
    v: str
    t_start: float
    t_end: float
    rate: float
    delay: float
    causes: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError(f"contact {self.uid}: empty window")
        if self.rate <= 0:
            raise ValueError(f"contact {self.uid}: non-positive rate")

    @property
    def capacity(self) -> float:
        """Bytes transferable over the window at line rate."""
        return self.rate * (self.t_end - self.t_start)

    def base_p(self, cause_p: dict[str, float]) -> float:
        """Marginal success probability of this contact, used for sizing only.

        This is the probability the contact succeeds averaged over causes, and it
        is *not* the joint probability structure -- that is the whole point of the
        cause model.  It is used only where a scalar per-contact probability is
        genuinely needed (e.g. the closed-form seed in Prop. 21), never where a
        joint quantity is required.
        """
        p_ok = 1.0
        for z in self.causes:
            p_ok *= 1.0 - cause_p.get(z, 0.0)
        return p_ok


class CauseWorld:
    """A sampled realisation of every cause in the fault model.

    Draw once per (seed, horizon) and reuse across contacts and fragments: this is
    what induces the correlation that prop:cdd is about.  Drawing per contact
    instead would silently give you (M1) and the effect would vanish.
    """

    def __init__(self, causes: dict[str, float], rng: np.random.Generator):
        self._p = dict(causes)
        self._rng = rng
        self._fired: dict[str, bool] | None = None

    def sample(self) -> "CauseWorld":
        """Draw an independent Bernoulli for every cause."""
        names = sorted(self._p)
        draws = self._rng.random(len(names))
        self._fired = {
            z: bool(d < self._p[z]) for z, d in zip(names, draws, strict=True)
        }
        return self

    def fired(self, cause: str) -> bool:
        if self._fired is None:
            raise RuntimeError("CauseWorld.sample() must be called before use")
        return self._fired.get(cause, False)

    def contact_ok(self, contact: Contact) -> bool:
        """True iff no cause gating this contact has fired."""
        return all(not self.fired(z) for z in contact.causes)

    def path_ok(self, contacts) -> bool:
        """True iff every contact on the path succeeds."""
        return all(self.contact_ok(c) for c in contacts)


def relay_grouping(contacts: list[Contact]) -> dict[str, frozenset[str]]:
    """Partition contacts by their relay cause, for the cor:relaygroup experiment.

    A contact tagged with a relay cause 'r:relayA' belongs to group 'relayA'.  A
    contact with no relay cause forms its own singleton group, because it is gated
    by a private cause and therefore independent of everything else.
    """
    groups: dict[str, set[str]] = {}
    for c in contacts:
        relay_causes = {z for z in c.causes if z.startswith("r:")}
        if len(relay_causes) == 1:
            key = next(iter(relay_causes))
        else:
            key = f"priv:{c.uid}"
        groups.setdefault(key, set()).update(c.causes)
    return {k: frozenset(v) for k, v in groups.items()}


def cause_groups_of_path(path: list[Contact]) -> frozenset[str]:
    """Cause set of a path: the union of its contacts' gating causes."""
    out: set[str] = set()
    for c in path:
        out.update(c.causes)
    return frozenset(out)


def path_groups(path: list[Contact]) -> tuple[str, ...]:
    """Which relay group(s) a path touches.

    Returns a tuple of group keys; a path touching two relay groups is counted in
    both, which is why the group-count model in cor:relaygroup needs care with
    multi-relay paths.  For the experiments we generate single-relay paths so this
    returns a singleton.
    """
    keys = set()
    for c in path:
        relay_causes = {z for z in c.causes if z.startswith("r:")}
        if len(relay_causes) == 1:
            keys.add(next(iter(relay_causes)))
        else:
            keys.add(f"priv:{c.uid}")
    return tuple(sorted(keys))


def disjoint_contact(paths: list[list[Contact]]) -> bool:
    """True iff the paths share no contact (the *insufficient* condition)."""
    for a, b in itertools.combinations(paths, 2):
        if {c.uid for c in a} & {c.uid for c in b}:
            return False
    return True


def disjoint_causes(paths: list[list[Contact]]) -> bool:
    """True iff the paths share no failure cause (the *sufficient* condition)."""
    sets = [cause_groups_of_path(p) for p in paths]
    for a, b in itertools.combinations(sets, 2):
        if a & b:
            return False
    return True
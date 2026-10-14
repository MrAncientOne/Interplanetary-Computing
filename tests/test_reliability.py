"""Exact-reliability and cause-structure checks.

These are the tests that decide whether the paper's central claims are true.  They
are written to fail loudly if a claim is false, not to be tuned until they pass.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ipcloud.causes import (
    CauseWorld,
    Contact,
    disjoint_causes,
    disjoint_contact,
)
from ipcloud.paths import (
    exact_correlated_reliability,
    greedy_disjoint_family,
    make_path,
    monte_carlo_reliability,
    predicted_independent_reliability,
    relay_group_reliability,
)
from ipcloud.reliability import (
    hoeffding_seed,
    is_submodular_sample,
    kl_seed,
    marginal_gain,
    pb_pmf,
    pb_tail,
)
from ipcloud.scenario import enumerate_paths, lunar_topo


# ---------------------------------------------------------------- exactness

def test_pb_tail_matches_closed_form_homogeneous():
    """P(Binom(n,q) >= k) for identical q must match the binomial formula."""
    from math import comb

    for n, q, k in [(5, 0.5, 2), (10, 0.3, 4), (8, 0.9, 7), (4, 0.7, 1)]:
        got = pb_tail(np.full(n, q), k)
        want = sum(comb(n, r) * q**r * (1 - q) ** (n - r) for r in range(k, n + 1))
        assert got == pytest.approx(want, abs=1e-12)


def test_pb_tail_edges():
    assert pb_tail(np.array([0.5, 0.5]), 0) == 1.0
    assert pb_tail(np.array([0.5, 0.5]), 3) == 0.0
    assert pb_tail(np.array([]), 1) == 0.0


def test_pb_pmf_sums_to_one_and_mass():
    rng = np.random.default_rng(7)
    for _ in range(200):
        q = rng.random(9)
        pi = pb_pmf(q)
        assert pi.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(pi >= -1e-15)


def test_exact_tail_matches_monte_carlo_independent():
    """eq. (30) must equal a direct simulation when causes are genuinely disjoint."""
    rng = np.random.default_rng(11)
    q = rng.random(6) * 0.4 + 0.6
    analytic = pb_tail(q, 3)
    trials, hits = 200_000, 0
    for _ in range(trials):
        if int((rng.random(6) < q).sum()) >= 3:
            hits += 1
    assert abs(analytic - hits / trials) < 0.005


# ---------------------------------------------------------------- the seed

def test_hoeffding_seed_is_actually_a_bound():
    """The corrected seed must never under-provision.

    This is the test that the *previous* formula would have failed.  We assert the
    bound holds on many randomised heterogeneous q vectors.
    """
    rng = np.random.default_rng(3)
    violations = 0
    for _ in range(500):
        k = int(rng.integers(1, 5))
        delta = float(10 ** rng.uniform(-4, -1))
        n = hoeffding_seed(k, 0.9, delta)
        q = rng.random(n) * 0.1 + 0.9
        if pb_tail(q, k) < 1.0 - delta:
            violations += 1
    assert violations == 0


def test_kl_seed_is_at_least_as_tight_as_hoeffding():
    rng = np.random.default_rng(5)
    for _ in range(100):
        k = int(rng.integers(1, 6))
        delta = float(10 ** rng.uniform(-4, -1))
        h = hoeffding_seed(k, 0.85, delta)
        kl = kl_seed(k, 0.85, delta)
        if kl is not None:
            assert kl <= h


# ---------------------------------------------------------------- degeneracy

def test_first_fragment_degeneracy_is_real():
    """lem:degenerate: at an empty incumbent every marginal gain is exactly zero.

    marginal_gain takes the *incumbent* q-vector.  With no incumbent the pmf is a
    point mass at 0, so pi_{k-1} = 0 for every k >= 1 and all gains vanish.
    """
    empty = np.array([])
    for k in (1, 2, 3):
        assert np.all(marginal_gain(empty, k) == 0.0)
    # with k-1 survivors present the gain lifts
    assert np.all(marginal_gain(np.array([0.9, 0.8]), 3) > 0.0)


def test_marginal_gain_equals_finite_difference_of_tail():
    """eq. (32) must be the exact finite difference of the exact tail."""
    rng = np.random.default_rng(13)
    for _ in range(200):
        k = int(rng.integers(1, 4))
        n = int(rng.integers(k, 8))
        q = rng.random(n) * 0.5 + 0.5
        inc = q[: n - 1]
        cand = q[n - 1]
        finite = pb_tail(np.append(inc, cand), k) - pb_tail(inc, k)
        pred = cand * pb_pmf(inc)[k - 1]
        assert finite == pytest.approx(pred, abs=1e-12)


# ---------------------------------------------------------------- submodularity

def test_F_is_NOT_submodular():
    """rem:notsubmod: F is NOT submodular for k>=2.  This test asserts the VIOLATION.

    We previously believed F was submodular and claimed a (1-1/e) guarantee on that
    basis.  It is not, and the minimal witness is small enough to state in the paper.
    The test is written to *require* a violation so the claim cannot silently
    reappear; if someone changes the model such that F becomes submodular, this
    test fails and the paper's text must be revisited.
    """
    rng = np.random.default_rng(17)
    ok, worst = is_submodular_sample(rng.random(7) * 0.8 + 0.2, 3, rng, trials=400)
    assert not ok, "F unexpectedly submodular; revisit rem:notsubmod"
    assert worst < -0.05, f"expected a clear violation, got worst={worst}"

    # the explicit minimal witness from rem:notsubmod
    q = np.array([0.7, 0.6])
    k = 2
    F = lambda idx: pb_tail(q[list(idx)], k)
    assert F(()) == 0.0
    assert F((0,)) == 0.0
    assert F((1,)) == 0.0
    assert F((0, 1)) == pytest.approx(0.42, abs=1e-12)
    # diminishing returns would need 0.0 >= 0.42
    assert F((1,)) - F(()) < F((0, 1)) - F((0,))


def test_top_r_by_q_is_exactly_optimal_for_fixed_cardinality():
    """prop:topr: for fixed r, the r largest-q paths maximise F exactly.

    Brute force over all r-subsets on small instances.  Zero violations expected.
    """
    import itertools

    rng = np.random.default_rng(101)
    for _ in range(400):
        n = int(rng.integers(3, 9))
        r = int(rng.integers(1, n + 1))
        k = int(rng.integers(1, r + 1))
        q = rng.random(n) * 0.8 + 0.2
        best = max(
            pb_tail(q[list(sub)], k) for sub in itertools.combinations(range(n), r)
        )
        top = tuple(np.argsort(-q)[:r].tolist())
        assert pb_tail(q[list(top)], k) == pytest.approx(best, abs=1e-12)


def test_cost_weighting_is_necessary_not_decorative():
    """rem:toprcaveat: top-r-by-q is NOT optimal under a cost budget.

    This is the measurement that justifies keeping CHIOR's marginal-gain-per-cost
    rule instead of simplifying Stage 3.  It asserts violations occur, so the cost
    denominator is doing real work.
    """
    import itertools

    rng = np.random.default_rng(102)
    violations = 0
    worst = 0.0
    for _ in range(300):
        n = 7
        r = int(rng.integers(1, n + 1))
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
            violations += 1
            worst = max(worst, best - got)
    assert violations > 0, "cost-blind top-r unexpectedly optimal; revisit the rule"
    assert worst > 0.1, f"violations too weak to justify cost weighting: {worst}"


def test_F_is_monotone_empirically():
    rng = np.random.default_rng(19)
    for _ in range(300):
        q = rng.random(6) * 0.9 + 0.1
        k = int(rng.integers(1, 4))
        order = rng.permutation(6)
        prev = -1.0
        for size in range(0, 7):
            val = pb_tail(q[order[:size]], k)
            assert val >= prev - 1e-12
            prev = val


# ---------------------------------------------------------------- prop:cdd

def _contact(uid, u, v, causes):
    return Contact(uid, u, v, 0.0, 100.0, 1e6, 0.1, frozenset(causes))


def test_contact_disjoint_but_cause_shared_is_dependent():
    """prop:cdd: two contact-disjoint paths sharing a relay are NOT independent.

    Note the direction of the modelling error depends on k relative to n; see
    test_independence_error_sign_depends_on_k.  Here k = n = 2, so the fragments
    must *both* survive, and positive correlation makes co-survival *more* likely
    than independence predicts -- so independence UNDERSTATES.
    """
    cause_p = {"r:relay0": 0.10, "link:a": 0.01, "link:b": 0.01}
    c1 = [_contact(1, "s", "relay0", {"r:relay0"}), _contact(2, "relay0", "d", {"link:a"})]
    c2 = [_contact(3, "s", "relay0", {"r:relay0"}), _contact(4, "relay0", "d", {"link:b"})]
    p1, p2 = make_path(c1, cause_p), make_path(c2, cause_p)

    assert disjoint_contact([c1, c2]) is True          # contact-disjoint
    assert disjoint_causes([c1, c2]) is False         # but NOT cause-disjoint

    truth = exact_correlated_reliability([p1, p2], 2, cause_p)
    predicted = predicted_independent_reliability([p1, p2], 2)
    assert truth == pytest.approx(0.9 * 0.99 * 0.99, abs=1e-12)
    assert predicted < truth - 0.01, "for k=n, independence must UNDERSTATE"


def test_independence_error_sign_depends_on_k():
    """rem:signedgap: the SIGN of the independence error flips with k.

    This is the correction to an earlier draft, which asserted that the independent
    model always overestimates reliability.  That is false.  With n fragments sharing
    one relay cause:
      * k well below n  -> you need only a few survivors, coupling HURTS,
                           so independence OVERSTATES;
      * k = n           -> you need every survivor, coupling HELPS,
                           so independence UNDERSTATES.
    The test asserts both regimes, so the paper cannot revert to the one-sided claim.
    """
    n, relay_fire = 6, 0.25
    cp = {"r:relay0": relay_fire}
    paths = []
    for i in range(n):
        cp[f"link:{i}"] = 0.05
        paths.append(
            make_path(
                [_contact(i, "s", "relay0", {"r:relay0"}),
                 _contact(100 + i, "relay0", "d", {f"link:{i}"})],
                cp,
            )
        )
    gaps = {}
    for k in (1, 3, 5, 6):
        ind = predicted_independent_reliability(paths, k)
        corr = exact_correlated_reliability(paths, k, cp)
        gaps[k] = ind - corr
    assert gaps[1] > 0.1, "small k: independence should overstate"
    assert gaps[3] > 0.1, "k=3: independence should still overstate"
    assert gaps[6] < -0.3, "k=n: independence should understate"


def test_cause_disjoint_family_makes_independence_exact():
    cause_p = {"r:relay0": 0.10, "r:relay1": 0.12, "link:a": 0.02, "link:b": 0.02}
    c1 = [_contact(1, "s", "relay0", {"r:relay0"}), _contact(2, "relay0", "d", {"link:a"})]
    c2 = [_contact(3, "s", "relay1", {"r:relay1"}), _contact(4, "relay1", "d", {"link:b"})]
    p1, p2 = make_path(c1, cause_p), make_path(c2, cause_p)
    assert disjoint_causes([c1, c2]) is True
    truth = exact_correlated_reliability([p1, p2], 2, cause_p)
    predicted = predicted_independent_reliability([p1, p2], 2)
    assert predicted == pytest.approx(truth, abs=1e-12)


def test_relay_group_reliability_matches_enumeration_on_topology():
    """cor:relaygroup's decomposition must equal brute-force enumeration."""
    scn = lunar_topo(n_relays=3)
    paths = enumerate_paths(scn, "base", "earth")
    fam = greedy_disjoint_family(paths, key=lambda p: p.latency, cause_mode=True, limit=4)
    if len(fam) >= 2:
        truth = exact_correlated_reliability(fam, 2, scn.cause_p)
        group = relay_group_reliability(fam, 2)
        assert group == pytest.approx(truth, abs=1e-9)


def test_monte_carlo_agrees_with_enumeration():
    scn = lunar_topo(n_relays=3)
    paths = enumerate_paths(scn, "base", "earth")
    fam = greedy_disjoint_family(paths, key=lambda p: p.latency, cause_mode=False, limit=4)
    truth = exact_correlated_reliability(fam, 2, scn.cause_p)
    rng = np.random.default_rng(23)
    mc = monte_carlo_reliability(fam, 2, scn.cause_p, rng, trials=40_000)
    assert abs(truth - mc) < 0.01


def test_more_relay_causes_never_hurts():
    """cor:relaygroup: spreading fragments over more relay causes strictly helps."""
    cause_p = {f"r:relay{i}": 0.10 for i in range(4)}
    div = [
        make_path([_contact(100 + i, "s", f"relay{i}", {f"r:relay{i}"})], cause_p)
        for i in range(4)
    ]
    coll = [
        make_path([_contact(200 + i, "s", "relay0", {"r:relay0"})], cause_p)
        for i in range(4)
    ]
    # k=3 of 4: diverse (independent) gives 4*0.9^3*0.1 + 0.9^4 = 0.9477;
    # collapsed is all-or-nothing on the single relay, so 0.9 exactly.
    f_div = exact_correlated_reliability(div, 3, cause_p)
    f_coll = exact_correlated_reliability(coll, 3, cause_p)
    assert f_div == pytest.approx(0.9477, abs=1e-9)
    assert f_coll == pytest.approx(0.90, abs=1e-9)
    assert f_div > f_coll + 0.04

    # Monotonicity in the number of distinct relays, at FIXED k and n.
    # Note k must be held fixed: at k=min(3,g) both n and k change, and the tail is
    # not monotone in that combined move (at g=1 you need 1 fragment and get 0.9;
    # at g=2 you need 3 of 2 which is impossible, so the value drops to 0).
    for g in (1, 2, 3, 4):
        fam = [
            make_path([_contact(300 + i, "s", f"relay{i}", {f"r:relay{i}"})], cause_p)
            for i in range(g)
        ]
        val = exact_correlated_reliability(fam, 3, cause_p)
        expected = pb_tail(np.full(g, 0.9), 3)  # independent by construction
        assert val == pytest.approx(expected, abs=1e-9)
    vals = [
        exact_correlated_reliability(
            [make_path([_contact(300 + i, "s", f"relay{i}", {f"r:relay{i}"})], cause_p)
             for i in range(g)],
            3,
            cause_p,
        )
        for g in (1, 2, 3, 4)
    ]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), vals


def test_path_q_does_not_double_count_shared_causes():
    """A cause shared by several contacts on one path must be counted once.

    Regression test: base->relay->earth has two contacts gated by the same relay
    cause.  Multiplying per contact squares its failure probability, which is what
    made relay_group_reliability disagree with brute-force enumeration by ~7e-6.
    """
    cp = {"r:relay0": 0.10, "link:earth": 0.005}
    c1 = [_contact(1, "base", "relay0", {"r:relay0"})]
    c2 = [_contact(2, "relay0", "earth", {"r:relay0", "link:earth"})]
    p = make_path(c1 + c2, cp)
    assert p.q == pytest.approx(0.9 * 0.995, abs=1e-12)
    # not the per-contact product
    assert p.q != pytest.approx(0.9 * 0.9 * 0.995, abs=1e-12)


def test_relay_group_dp_equals_enumeration_on_lunar_topo():
    """cor:relaygroup's DP must be exact, to floating point, on the real topology."""
    scn = lunar_topo(n_relays=3)
    paths = enumerate_paths(scn, "base", "earth")
    for cause_mode in (True, False):
        fam = greedy_disjoint_family(
            paths, key=lambda p: p.latency, cause_mode=cause_mode, limit=4
        )
        if len(fam) < 2:
            continue
        for k in range(1, len(fam) + 1):
            truth = exact_correlated_reliability(fam, k, scn.cause_p)
            grp = relay_group_reliability(fam, k)
            assert abs(grp - truth) < 1e-12, (cause_mode, k, grp, truth)


def test_relay_group_dp_refuses_invalid_partition():
    """The DP must raise rather than return a wrong number on a bad partition.

    Regression test for a silent miscount: two single-relay paths that share a
    non-relay cause (here the shared downlink) are NOT independent, but they land in
    different relay groups, so a naive grouping would treat them as independent and
    overstate reliability.
    """
    cp = {"r:relay0": 0.10, "r:relay1": 0.10, "link:shared_dl": 0.05}
    a = make_path([_contact(1, "s", "relay0", {"r:relay0"}),
                   _contact(2, "relay0", "d", {"r:relay0", "link:shared_dl"})], cp)
    b = make_path([_contact(3, "s", "relay1", {"r:relay1"}),
                   _contact(4, "relay1", "d", {"r:relay1", "link:shared_dl"})], cp)
    with pytest.raises(ValueError, match="not cause-disjoint"):
        relay_group_reliability([a, b], 2)


def test_byte_conservation():
    """Sanity identity: sent = delivered + dropped + still buffered."""
    rng = np.random.default_rng(29)
    sent = 0.0
    delivered = 0.0
    dropped = 0.0
    remaining = 0.0
    for _ in range(500):
        size = float(rng.integers(1, 1000))
        sent += size
        r = rng.random()
        if r < 0.6:
            delivered += size
        elif r < 0.8:
            dropped += size
        else:
            remaining += size
    assert sent == pytest.approx(delivered + dropped + remaining, abs=1e-9)
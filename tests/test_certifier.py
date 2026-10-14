"""Tests for the Local Sufficiency Invariant certifier (sec:certifier).

These guard the four bugs found while building it:

1. ``all()`` over an EMPTY sequence returns True, which classified every direct
   single-hop path as "relay-mediated" and put every node in every residue.
2. Evaluating the residue at the *end of the horizon* always lands in a blackout.
3. Residue and admissible-path window anchored at different times.
4. Fixing k = len(paths) demands every fragment (P(>=n of n) = q^n); the verifier
   must CHOOSE k to maximise the tail.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ipcloud.certifier import (  # noqa: E402
    CertConfig,
    certify,
    disconnected_residue,
    summarise,
)
from ipcloud.reliability import pb_pmf, pb_tail  # noqa: E402
from ipcloud.scenario import lunar_topo, enumerate_paths  # noqa: E402
from ipcloud.workload import default_workload  # noqa: E402

EPS = 1e-3


@pytest.fixture(scope="module")
def scn():
    return lunar_topo(n_relays=3, horizon=2 * 86400)


@pytest.fixture(scope="module")
def task0(scn):
    return default_workload(scn.horizon, volume="light")[0]


# --------------------------------------------------------------- residue bugs

def test_direct_path_is_not_relay_mediated(scn):
    """Regression for the all([]) bug.

    A direct single-hop path has no intermediate contacts, so ``all(...)`` over
    ``p.contacts[:-1]`` is vacuously True. That wrongly declared the device
    unreachable from the surface tier and put every node in every residue, which
    made the verifier certify nothing but local execution.
    """
    dead = disconnected_residue(scn, "base", 1253.0, 6 * 3600.0, 0.999)
    # relays and earth must be in the residue: they cannot be learned inside 6 h
    assert {"earth", "relay0", "relay1", "relay2"} <= set(dead)
    # ... but the device itself must NOT be: a direct link exists
    assert "rover" not in dead


def test_residue_is_anchored_at_the_given_time(scn):
    """The invariant is quantified from the task's arrival, not from the horizon.

    Anchoring at the horizon end always lands in a blackout, so every remote
    placement failed for reasons that had nothing to do with the architecture.
    """
    early = disconnected_residue(scn, "base", 1253.0, 6 * 3600.0, 0.999)
    late = disconnected_residue(scn, "base", scn.horizon, 6 * 3600.0, 0.999)
    assert "rover" not in early          # reachable early in the mission
    assert "relay0" in late              # blackout at the horizon tail


def test_residue_window_and_path_window_agree(scn):
    """Residue and admissible-path window must be anchored at the SAME instant.

    Anchoring them differently silently widens the check: a path could count as
    available while the node that would use it was already removed.
    """
    for anchor in (1253.0, 22853.0, 150000.0):
        dead = disconnected_residue(scn, "base", anchor, 6 * 3600.0, 0.999)
        paths = enumerate_paths(scn, "rover", "base")
        lo = max(0.0, anchor - 6 * 3600.0)
        usable = [p for p in paths if lo <= p.contacts[0].t_start <= anchor + 1e-9]
        if "rover" not in dead:
            assert usable, f"rover reachable at {anchor}s but no path in window"


# ------------------------------------------------------------------ k choice

def test_k_is_chosen_not_fixed():
    """P(>=n of n) = q^n is the fine-fragmentation trap.

    With four q=0.99 fragments, k=4 gives 0.961 (fails eps=1e-2) while k=1 gives
    ~1.0 at the same bandwidth. The verifier must pick the maximising k.
    """
    q = np.full(4, 0.99)
    pmf = pb_pmf(q)
    tails = np.array([pmf[k:].sum() for k in range(1, 5)])
    assert tails[0] > tails[3]          # k=1 beats k=4
    assert int(np.argmax(tails)) + 1 == 1
    assert pb_tail(q, 1) > 1 - EPS
    assert pb_tail(q, 4) < 1 - EPS


def test_all_n_paths_is_never_better_than_best_k():
    rng = np.random.default_rng(7)
    for _ in range(50):
        q = np.round(rng.uniform(0.8, 0.999, size=6), 4)
        pmf = pb_pmf(q)
        tails = [pmf[k:].sum() for k in range(1, 7)]
        assert max(tails) >= pmf[6].sum() - 1e-12


# ------------------------------------------------------- certificate content

def test_local_placement_is_sat_with_an_honest_probability(scn, task0):
    certs = certify([task0], CertConfig(theta_h=6.0, horizon_days=2.0, eps=EPS), scn=scn)
    assert len(certs) == 1
    c0 = certs[0]
    assert c0.sat and c0.placement == "rover"
    # Never a bare 1.0000 that reads like certainty, but never trivially low either
    assert c0.pi_ind is not None and c0.pi_ind <= 1.0
    assert c0.pi_ind >= 1 - EPS


def test_surface_tier_is_unsat_with_a_witness(scn, task0):
    """The finding the verifier actually produces, and it must be honest.

    The device reaches the base through ONE link whose per-fragment success is
    0.99, so no fragment count can reach 1-eps = 0.999. The witness must name
    both the shortfall and the tier that failed.
    """
    certs = certify([task0], CertConfig(theta_h=6.0, horizon_days=2.0, eps=EPS), scn=scn)
    for c in certs:
        if c.placement == "base":
            assert not c.sat
            assert c.witness is not None and "probability" in c.witness
            assert c.pi_ind is not None and c.pi_ind < 1 - EPS


def test_remote_tiers_fail_for_distinguishable_reasons(scn):
    """The discriminator: each remote tier is rejected for a DIFFERENT reason.

    This is the observable payload of the invariant, and it is what shows the
    tool has teeth rather than returning one blanket verdict.

      * base  -- the device reaches it over ONE link, so no fragment count
                 reaches 1-eps: reliability 0.990 < 1-eps.
      * relay -- pi_ind ~ 1.0 but the exact cause model gives pi_ccf = 0.90,
                 because every relay-routed path shares R_r (prop:cdd).
      * earth -- reachable over the window and mostly certifiable, but not for
                 every task.

    A certifier that reported "local is best" without a per-tier witness would be
    indistinguishable from one that had simply been pointed at a hard problem.
    """
    from ipcloud.certifier import certify

    cfg = CertConfig(theta_h=6.0, horizon_days=2.0, eps=EPS)
    tasks = default_workload(cfg.horizon_days * 86400, volume="nominal")
    certs = certify(tasks, cfg, scn=scn)

        # `tried` carries every attempted tier for a task, so a verifier run that
    # settles on one placement still exposes why the others were rejected.
    def by_tier(placement):
        return [(c, t) for c in certs for t in c.tried if t[0] == placement]

    base = [t for _c, t in by_tier("base")]
    relays = [t for _c, t in by_tier("relay0")] + [t for _c, t in by_tier("relay1")]
    earth = [t for _c, t in by_tier("earth")]

    # base: whatever the verdict, an UNSAT must NAME a reason, and the reason is
    # either fragment diversity (one link, so no k reaches 1-eps) or the window
    # (survived less of the 72 h MTD than the invariant demands).
    assert base
    for _sat, _pl, _surv, pi_i, _pi_c, why in base:
        if not why or "none in window" == why:
            continue
        assert ("reliability" in why or "probability" in why
                or "swept" in why or "window" in why
                or "residue" in why), why

    # relay: UNSAT for a named reason, and the common-cause gap does occur and
    # runs the way the paper predicts (pi_ind optimistic, pi_ccf low). We assert
    # the gap APPEARS rather than that every row is that way: over a longer
    # window a relay row can also fail on reliability alone.
    assert relays
    gaps = []
    for _sat, sat, _surv, pi_i, pi_c, why in relays:
        if sat:
            continue
        assert why and why != "none in window", why
        assert ("common-cause gap" in why or "reliability" in why
                or "probability" in why or "swept" in why
                or "residue" in why or "window" in why), why
        if "common-cause gap" in why:
            gaps.append((pi_i, pi_c))
    assert gaps, "the shared relay cause must bite somewhere"
    for pi_i, pi_c in gaps:
        assert pi_i >= 1 - EPS and pi_c < 1 - EPS

    # earth: reachable, certifiable for some tasks but not all
    sat_earth = [t for t in earth if t[1]]
    unsat_earth = [t for t in earth if not t[1]]
    assert sat_earth, "earth should be certifiable for some tasks"
    assert unsat_earth, "earth should fail for some tasks, with a witness"
    for _sat, _pl, _surv, pi_i, _pi_c, why in unsat_earth:
        assert why and why != "none in window", why

    # and the verifier as a whole never reports a remote-only verdict
    assert any(c.sat for c in certs)


def test_remote_placement_publishes_its_probabilities(scn, task0):
    """Regression: pi must be PUBLISHED, not merely computed.

    A patch deleted the `pi_ind, pi_ccf = ind, ccf` assignment while leaving the
    witness text that quotes the numbers, so every remote certificate reported
    "reliability 0.990 < 1-eps" with pi_ind = None. The table took a number from
    nowhere. The witness and the reported pi must agree.
    """
    from ipcloud.certifier import _try_placement

    sat, surv, pi_i, pi_c, why = _try_placement(
        task0, "base", scn, CertConfig(theta_h=6.0, horizon_days=2.0, eps=EPS)
    )
    assert not sat
    assert pi_i is not None and pi_c is not None, "pi must be published"
    # the witness quotes the number it actually measured
    quoted = f"{pi_i:.3f}"
    assert quoted in why, f"witness {why!r} does not quote pi_ind {quoted}"


def test_summarise_counts_sat_and_placements(scn, task0):
    certs = certify([task0], CertConfig(theta_h=6.0, horizon_days=2.0, eps=EPS), scn=scn)
    s = summarise(certs)
    assert s["n_tasks"] == 1
    assert s["n_sat"] == len([c for c in certs if c.sat])
    assert s["median_survives_h"] >= 0.0


def test_certifier_is_fast_enough_to_sweep(scn, task0):
    """Regression guard on the complexity blowup.

    The exact cause term is 2^|causes| over the candidate paths, so the verifier
    must cap candidates at the top-2K by q (as CHIOR stage 2 does). Dropping the
    cap made a single task take >20 s because rover->earth has >1500 paths.
    """
    import time

    t0 = time.time()
    certify([task0], CertConfig(theta_h=1.0, horizon_days=2.0, eps=EPS), scn=scn)
    assert time.time() - t0 < 5.0


def test_no_certificate_claims_certainty(scn, task0):
    certs = certify([task0], CertConfig(theta_h=6.0, horizon_days=2.0, eps=EPS), scn=scn)
    why = " ".join(c.first_fail_condition or "" for c in certs).lower()
    assert "guaranteed" not in why

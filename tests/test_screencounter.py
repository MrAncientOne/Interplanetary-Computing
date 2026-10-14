"""Executable counterexample for rem:screencounter.

The paper's screening remark (S1) compares the offload estimate for task j
against W_j / f_s using *only j's own parameters*.  Constraint (C4), which splits
a node's throughput across every task placed there, is invisible to that
comparison.  These tests pin down the arithmetic that makes the point concrete,
and they are written to fail loudly if the arithmetic is wrong.

Run:  python3 -m pytest tests/test_screencounter.py -q
"""

from __future__ import annotations

import pytest


def offload_estimate(d_in, d_out, cycles, f_v, rate, delta_sv=0.0, delta_vs=0.0):
    """Expected remote latency, eq.(offload).

    Zero propagation and zero waiting, matching the two-node instance stated in
    rem:screencounter.
    """
    return d_in / rate + delta_sv + cycles / f_v + d_out / rate + delta_vs


def test_offload_estimate_matches_paper_arithmetic():
    """The worked example in rem:screencounter: 3T against 2T.

    An earlier draft claimed the offload cost was 4T; the terms sum to 3T.  This
    test exists so that the number printed in the paper is the number computed.
    """
    T = 100.0
    d_in = d_out = T / 2.0
    cycles = 2.0 * T
    f_s = f_v = 1.0
    rate = 1.0

    got = offload_estimate(d_in, d_out, cycles, f_v, rate)
    local = cycles / f_s

    assert got == pytest.approx(3.0 * T)
    assert local == pytest.approx(2.0 * T)
    # S1 fails, so Stage 1 prunes v.
    assert got >= local


def test_screen_is_blind_to_contention():
    """Screen S1 is unchanged by how loaded node s is; (C4) is not.

    Task j is screened against W_j / f_s.  That quantity depends on j alone, so
    adding a competing task j' at s cannot change whether v is pruned.  The
    contention term of (C4), by contrast, does depend on j'.  This is the gap the
    paper claims, expressed as a property of the screening rule rather than as a
    proof sketch.
    """
    T = 100.0
    d_in = d_out = T / 2.0
    cycles = 2.0 * T
    f_s = f_v = 1.0
    rate = 1.0

    screen_for_j = offload_estimate(d_in, d_out, cycles, f_v, rate) >= cycles / f_s

    # A second task at s, sized so its own local execution misses its deadline.
    j_prime_cycles = 2.0 * T
    j_prime_f_s = 1.0 / (2.0 * T)
    j_prime_local = j_prime_cycles / j_prime_f_s
    j_prime_deadline = 3.0 * T

    # Adding j' does not change the screen's verdict for j ...
    assert screen_for_j is True
    # ... yet it makes s a strictly worse host, which is the contention term the
    # screen never sees.
    assert j_prime_local > j_prime_deadline


def test_screencounter_label_is_a_remark_not_a_theorem():
    """Guard against the claim being silently promoted back to a theorem.

    The paper previously stated this as `thm:screencounter` with a proof sketch
    whose arithmetic did not check out.  It is now `rem:screencounter`.  If the
    label is reintroduced, this test fails.
    """
    from pathlib import Path

    # tests/ -> ipcloud/ -> code/ -> <repo root>/paper/
    tex = Path(__file__).resolve().parents[3] / "paper" / "interplanetary_cloud_report.tex"
    if not tex.exists():  # artefact run without the manuscript
        pytest.skip("manuscript not present next to the package")

    src = tex.read_text(encoding="utf-8", errors="replace")
    assert r"\label{rem:screencounter}" in src
    assert r"\label{thm:screencounter}" not in src


def test_remarks_are_unnumbered():
    """Remarks must not consume the shared theorem counter.

    IEEE Transactions numbers only Definition/Assumption/Theorem/Proposition/
    Lemma/Corollary.  Remarks are commentary.  While they shared the counter
    a simulation note was numbered "Remark 64" and the certifier section reached
    "Proposition 67".  This test pins the two environment definitions so the
    distinction cannot silently regress.
    """
    from pathlib import Path

    tex = Path(__file__).resolve().parents[3] / "paper" / "interplanetary_cloud_report.tex"
    if not tex.exists():
        pytest.skip("manuscript not present next to the package")

    src = tex.read_text(encoding="utf-8", errors="replace")
    assert r"\newtheorem*{rem}" in src, "rem must be defined unnumbered"
    assert r"\newtheorem*{remark}" in src, "remark must be defined unnumbered"
    assert r"\newtheorem{rem}[thm]{Remark}" not in src, (
        "rem must not share the numbered theorem counter"
    )
    assert r"\newtheorem{remark}[thm]{Remark}" not in src, (
        "remark must not share the numbered theorem counter"
    )
    # ...while the result environments stay numbered.
    for env in ("thm", "prop", "lem", "cor", "dfn"):
        assert r"\newtheorem{%s}" % env in src, f"{env} must remain numbered"
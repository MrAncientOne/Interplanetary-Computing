"""Poisson-Binomial tails, exact marginal gain, and the seed formula.

Everything here is exact (O(n) dynamic program), not bounded.  The paper argues in
Prop. 20 and the accompanying remark that a bound-based formulation is needlessly
lossy and that the earlier Chernoff seed was in fact invalid as written, so this
module deliberately contains no approximation for the tail itself.

Three quantities matter for the heuristic and all three are checked against
Monte-Carlo in tests/test_reliability.py:
  * pb_tail(q, k)        -- P(K >= k), eq. (30)
  * marginal_gain(q, k)  -- q_{m'} * pi_{k-1}, eq. (32)
  * hoeffding_seed(k, qbar, delta) -- closed-form seed, eq. (34)
"""

from __future__ import annotations

import numpy as np


def pb_pmf(q: np.ndarray) -> np.ndarray:
    """Poisson-Binomial pmf by convolution, eq. (31).

    q is the vector of per-fragment success probabilities.  Returns an array of
    length len(q)+1 whose r-th entry is P(K = r).  O(n) with a single pass.
    """
    q = np.asarray(q, dtype=float)
    if q.size == 0:
        return np.array([1.0])
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    pi = np.zeros(q.size + 1, dtype=float)
    pi[0] = 1.0
    for m, qm in enumerate(q):
        # in-place downward update keeps this O(n) overall
        pi[1 : m + 2] = pi[1 : m + 2] * (1.0 - qm) + pi[0 : m + 1] * qm
        pi[0] *= 1.0 - qm
    return pi


def pb_tail(q: np.ndarray, k: int) -> float:
    """P(K >= k), eq. (30).  Exact.

    For k <= 0 this is 1 and for k > len(q) it is 0, matching the convention that a
    requirement larger than the number of fragments is unsatisfiable.  These edge
    cases matter: the degenerate first-fragment regime of Lemma 12 lives exactly
    here.
    """
    q = np.asarray(q, dtype=float)
    n = q.size
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    pi = pb_pmf(q)
    return float(pi[k:].sum())


def marginal_gain(q: np.ndarray, k: int) -> np.ndarray:
    """Delta F of adding each candidate fragment, eq. (32).

    Returns the vector of gains q_m * pi_{k-1}, where pi is the pmf of the
    *incumbent* set.  This is the utility CHIOR's Stage 3 maximises.

    At an empty incumbent the gain is identically zero for k >= 1, which is the
    degeneracy of Lemma 12 and is why Stage 3 has a separate seeded phase.  This is
    asserted in test_reliability.py so the degeneracy cannot silently disappear if
    the DP is refactored.
    """
    q = np.asarray(q, dtype=float)
    pi = pb_pmf(q)
    if k <= 0:
        return np.zeros_like(q)
    if k - 1 >= pi.size:
        return np.zeros_like(q)
    return q * pi[k - 1]


def hoeffding_seed(k: int, qbar: float, delta: float) -> int:
    """Closed-form redundancy seed, eq. (34).

    Smallest n satisfying qbar*y^2 - sqrt(L/2)*y - k >= 0 with y = sqrt(n) and
    L = ln(1/delta).  Returns at least k, since n >= k is a hard modelling
    requirement independent of reliability.
    """
    if not 0.0 < qbar <= 1.0:
        raise ValueError("qbar must lie in (0,1]")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if k <= 0:
        raise ValueError("k must be positive")
    L = np.log(1.0 / delta)
    y = (np.sqrt(L / 2.0) + np.sqrt(L / 2.0 + 4.0 * qbar * k)) / (2.0 * qbar)
    return max(k, int(np.ceil(y**2)))


def kl_seed(k: int, qbar: float, delta: float, n_max: int = 512) -> int | None:
    """Multiplicative-Chernoff (KL) seed: smallest n with n*D(k/n||qbar) >= L.

    Stronger than hoeffding_seed.  Returns None if no n <= n_max satisfies it, which
    the caller must handle rather than silently clamping.
    """
    L = np.log(1.0 / delta)
    best: int | None = None
    for n in range(max(k, 1), n_max + 1):
        q = k / n
        # KL divergence D(q||qbar), with the convention 0*log 0 = 0 and
        # D -> +inf as q -> qbar.  Handled explicitly to avoid log(0) warnings.
        if q >= qbar:
            # q == qbar gives D = 0; q > qbar is not a valid lower-tail point,
            # so the Chernoff bound does not apply and we stop searching.
            break
        if q <= 0.0:
            d = np.inf
        elif q >= 1.0:
            d = np.inf
        else:
            d = q * np.log(q / qbar) + (1.0 - q) * np.log((1.0 - q) / (1.0 - qbar))
        if n * d >= L:
            best = n
            break
    return best


def minimum_n_for_reliability(
    q_candidates: np.ndarray, k: int, delta: float, n_max: int
) -> tuple[int, float]:
    """Smallest n whose exact tail reaches 1-delta, and the achieved probability.

    Scans n upward using the exact pmf, which is what Stage 2 does in the paper.
    Because q_candidates is a fixed *ordered* list and we take prefixes, this is
    monotone in n by prop:mono and the first success is the minimum.
    """
    for n in range(k, min(n_max, q_candidates.size) + 1):
        f = pb_tail(q_candidates[:n], k)
        if f >= 1.0 - delta:
            return n, f
    n = min(n_max, q_candidates.size)
    return n, pb_tail(q_candidates[:n], k)


def is_submodular_sample(
    q: np.ndarray, k: int, rng: np.random.Generator, trials: int = 400
) -> tuple[bool, float]:
    """Empirical check of prop:submod by brute force over small sets.

    g(A) = P(at least k of A succeed) under independent per-path success
    probabilities q.  We sample random nested triples A subset B and check
    g(A + R) - g(A) >= g(B + R) - g(B).  Returns (all_passed, worst_violation).
    """
    worst = 0.0
    n = q.size
    for _ in range(trials):
        order = rng.permutation(n)
        a_size = int(rng.integers(0, n))
        b_size = int(rng.integers(a_size, n))
        A = set(order[:a_size].tolist())
        B = set(order[:b_size].tolist())
        remaining = [i for i in range(n) if i not in B]
        if not remaining:
            continue
        r = int(rng.choice(remaining))
        gA = pb_tail(q[sorted(A)] if A else np.array([]), k)
        gAu = pb_tail(q[sorted(A | {r})], k)
        gB = pb_tail(q[sorted(B)], k)
        gBu = pb_tail(q[sorted(B | {r})], k)
        violation = (gBu - gB) - (gAu - gA)
        worst = min(worst, violation)
    return worst > -1e-12, worst
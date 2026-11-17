"""Backward recurrences F_n(k) = G_n(k) + H_n(k) F_n(k+1) and their continuum limits.

Shared library for the experiments, figures and animation of the project
"Pointwise convergence conjecture for recurrence-defined functions" (MATHDB 354674).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


# ---------------------------------------------------------------------------
# Generic solver
# ---------------------------------------------------------------------------
def solve_backward(G, H, mu):
    """Return F[0..K] with F[K] = mu and F[k] = G[k] + H[k] F[k+1] for k < K.

    G and H are sequences of length K (indices 0..K-1). Works with floats or Fractions.
    """
    K = len(G)
    F = [None] * (K + 1)
    F[K] = mu
    for k in range(K - 1, -1, -1):
        F[k] = G[k] + H[k] * F[k + 1]
    return F


def sample(F, n, x):
    """f_n(x) = F_n(floor(n x)) for an array of x in [0, 1)."""
    x = np.atleast_1d(x)
    idx = np.floor(n * x + 1e-12).astype(int)
    return np.asarray(F, dtype=float)[idx]


# ---------------------------------------------------------------------------
# The counterexample (Principia solution to MATHDB 354674)
# ---------------------------------------------------------------------------
def spike_index(n):
    return n // 2 + 1


def counterexample_coeffs(n, exact=False, spike_at=None):
    """H_n(k) = (n-k)/(n-k+1); a unit spike in G_n at j_n = floor(n/2)+1 (or at `spike_at`)."""
    num = Fraction if exact else float
    H = [num(n - k) / num(n - k + 1) for k in range(n)]
    j = spike_index(n) if spike_at is None else spike_at
    G = [num(0)] * n
    if n >= 4 and 0 <= j < n:
        G[j] = num(1)
    return G, H


def counterexample_closed_form(n, k):
    """F_n(k) = ceil(n/2)/(n-k+1) for k <= j_n, and 0 beyond (exact)."""
    if n >= 4 and k <= spike_index(n):
        return Fraction((n + 1) // 2, n - k + 1)
    return Fraction(0)


def counterexample_limit(x):
    """Pointwise limit of f_n: 1/(2(1-x)) on (0, 1/2], 0 on (1/2, 1)."""
    x = np.asarray(x, dtype=float)
    return np.where(x <= 0.5, 0.5 / (1.0 - x), 0.0)


# ---------------------------------------------------------------------------
# Measure-valued limit (Theorem 2 of the paper)
# ---------------------------------------------------------------------------
def propagator_from_antiderivative(Hint):
    """Phi(x, s) = exp(-(Hint(s) - Hint(x))) for an antiderivative Hint of h."""
    return lambda x, s: np.exp(-(Hint(s) - Hint(x)))


def measure_limit(x, Phi, density=None, atoms=(), Theta=0.0, b=1.0, quad_pts=4000):
    """f(x) = int_{[x,b]} Phi(x,s) dnu(s) + Theta Phi(x,b), nu = density ds + sum a_i delta_{c_i}."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        val = Theta * Phi(xi, b)
        if density is not None and xi < b:
            # Gauss-Legendre quadrature on [xi, b]
            nodes, weights = np.polynomial.legendre.leggauss(64)
            # composite: split [xi, b] into panels for robustness
            panels = np.linspace(xi, b, 33)
            for a0, a1 in zip(panels[:-1], panels[1:]):
                s = 0.5 * (a1 - a0) * nodes + 0.5 * (a1 + a0)
                val += 0.5 * (a1 - a0) * np.sum(weights * Phi(xi, s) * density(s))
        for (c, a) in atoms:
            if xi <= c <= b:
                val += a * Phi(xi, c)
        out[i] = val
    return out

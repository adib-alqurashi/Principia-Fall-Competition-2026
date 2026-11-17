"""All numerical experiments and figures for the project (one run produces everything).

Usage:  python3 experiments.py  [--outdir ../figures]
Outputs:
  figures/paper/*.pdf   vector figures for the paper (light theme)
  figures/light/*.png   300-dpi light figures
  figures/dark/*.png    300-dpi dark figures for slides
  results.json          every number quoted in the paper
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recurrence import (  # noqa: E402
    solve_backward, sample, spike_index, counterexample_coeffs,
    counterexample_closed_form, counterexample_limit, measure_limit,
)

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
THEMES = {
    "light": dict(bg="#FFFFFF", fg="#1B1F2A", muted="#6B7280", grid="#E5E7EB",
                  blue="#2F5BD3", coral="#D64545", teal="#12857A", amber="#C98A0B",
                  violet="#7C5CD6", spine="#9CA3AF"),
    "dark": dict(bg="#0B1220", fg="#E6EAF2", muted="#8A96AD", grid="#24304A",
                 blue="#5B8DEF", coral="#FF6B6B", teal="#2EC4B6", amber="#F4B942",
                 violet="#A78BFA", spine="#3A4766"),
}


def set_theme(name):
    t = THEMES[name]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "xtick.color": t["fg"],
        "ytick.color": t["fg"], "axes.edgecolor": t["spine"], "grid.color": t["grid"],
        "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10,
        "axes.labelsize": 9.5, "legend.fontsize": 8.2, "legend.frameon": False,
        "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "lines.solid_capstyle": "round",
    })
    return t


def save(fig, name, outdir, theme):
    if theme == "light":
        os.makedirs(os.path.join(outdir, "paper"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "light"), exist_ok=True)
        fig.savefig(os.path.join(outdir, "paper", name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "light", name + ".png"), dpi=300, bbox_inches="tight")
    else:
        os.makedirs(os.path.join(outdir, "dark"), exist_ok=True)
        fig.savefig(os.path.join(outdir, "dark", name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def step_xy(F, n):
    """Coordinates of the step function x -> F[floor(n x)] on [0, 1)."""
    xs, ys = [], []
    for k in range(n):
        xs += [k / n, (k + 1) / n]
        ys += [float(F[k]), float(F[k])]
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# E1  exact verification of the closed form
# ---------------------------------------------------------------------------
def e1_exact(nmax=400):
    for n in range(4, nmax + 1):
        G, H = counterexample_coeffs(n, exact=True)
        F = solve_backward(G, H, Fraction(0))
        for k in range(n + 1):
            assert F[k] == counterexample_closed_form(n, k), (n, k)
            assert 0 <= F[k] <= 1
        for k in range(n):
            assert 0 < H[k] < 1
    return {"closed_form_verified_exactly_for_n": [4, nmax]}


# ---------------------------------------------------------------------------
# E2  convergence rate of the counterexample to its measure-valued limit
# ---------------------------------------------------------------------------
def e2_rates():
    ns = [2 ** p for p in range(3, 18)]
    xs = np.concatenate([np.linspace(0.05, 0.45, 401), np.linspace(0.55, 0.95, 401)])
    errs, f_quarter = [], []
    for n in ns:
        G, H = counterexample_coeffs(n)
        F = solve_backward(G, H, 0.0)
        fn = sample(F, n, xs)
        errs.append(float(np.max(np.abs(fn - counterexample_limit(xs)))))
        f_quarter.append(float(sample(F, n, 0.25)[0]))
    slope = np.polyfit(np.log(ns[4:]), np.log(errs[4:]), 1)[0]
    return {"ns": ns, "sup_err_away_from_half": errs, "f_n_quarter": f_quarter,
            "fitted_slope": float(slope)}


# ---------------------------------------------------------------------------
# E3  a generic test of Theorem 2: smooth forcing + two moving spikes + perturbed h_n
# ---------------------------------------------------------------------------
A_SMOOTH = 0.6


def h_fun(x):
    return 1.0 + A_SMOOTH * np.sin(2 * np.pi * x)


def h_int(x):
    return x - A_SMOOTH / (2 * np.pi) * np.cos(2 * np.pi * x)


def g_fun(x):
    return 1.0 + 2.0 * x * (1 - x)


ATOMS = [(0.3, 0.8), (0.7, 0.5)]
THETA = 0.4


def e3_coeffs(n):
    k = np.arange(n)
    hn = h_fun(k / n) + np.sin(37.0 * k) / math.sqrt(n)      # uniform O(n^{-1/2}) perturbation
    H = 1.0 - hn / n
    G = g_fun(k / n) / n
    for (c, a) in ATOMS:
        G[int(math.floor(c * n)) + 1] += a                  # spikes that move with n
    return G, H


def phi_e3(x, s):
    return np.exp(-(h_int(s) - h_int(x)))


def e3_general():
    ns = [2 ** p for p in range(4, 18)]
    xs = np.linspace(0.02, 0.98, 961)
    mask = np.ones_like(xs, bool)
    for (c, _) in ATOMS:
        mask &= np.abs(xs - c) > 0.02
    xs = xs[mask]
    lim = measure_limit(xs, phi_e3, density=g_fun, atoms=ATOMS, Theta=THETA)
    naive = measure_limit(xs, phi_e3, density=g_fun, atoms=(), Theta=THETA)
    errs = []
    for n in ns:
        G, H = e3_coeffs(n)
        F = solve_backward(list(G), list(H), THETA)
        errs.append(float(np.max(np.abs(sample(F, n, xs) - lim))))
    slope = np.polyfit(np.log(ns[4:]), np.log(errs[4:]), 1)[0]
    gap = float(np.max(np.abs(lim - naive)))
    return {"ns": ns, "sup_err": errs, "fitted_slope": float(slope),
            "max_gap_limit_vs_naive_ode": gap}


# ---------------------------------------------------------------------------
# E4  smoothed spikes: width w -> 0 recovers the jump
# ---------------------------------------------------------------------------
def phi_ce(x, s):  # counterexample propagator (1-s)/(1-x)
    return (1.0 - s) / (1.0 - x)


def smoothed_limit(xs, w):
    dens = lambda s: np.where(np.abs(s - 0.5) < w / 2, 1.0 / w, 0.0)  # noqa: E731
    out = []
    for x in xs:
        lo, hi = max(x, 0.5 - w / 2), 0.5 + w / 2
        if lo >= hi:
            out.append(0.0)
            continue
        # exact integral of (1-s)/(1-x) * 1/w over [lo, hi]
        out.append(((hi - lo) - (hi ** 2 - lo ** 2) / 2) / (w * (1 - x)))
    return np.array(out)


def e4_smoothing():
    ws = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01]
    # discrete check for one width
    n, w = 4096, 0.1
    k = np.arange(n)
    G = np.where(np.abs((k + 0.5) / n - 0.5) < w / 2, 1.0 / (w * n), 0.0)
    _, H = counterexample_coeffs(n)
    F = solve_backward(list(G), H, 0.0)
    xs = np.linspace(0.02, 0.95, 500)
    err = float(np.max(np.abs(sample(F, n, xs) - smoothed_limit(xs, w))))
    # uniform-integrability diagnostic: largest mass in a window of half-width delta
    deltas = np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    n = 4096
    Gs, _ = counterexample_coeffs(n)
    Gs = np.array(Gs)
    pos = np.arange(n) / n

    def window_mass(Garr, d):
        cs = np.concatenate([[0.0], np.cumsum(Garr)])
        best = 0.0
        for c in np.linspace(0, 1, 401):
            lo, hi = np.searchsorted(pos, c - d), np.searchsorted(pos, c + d, side="right")
            best = max(best, cs[hi] - cs[lo])
        return best

    Gsmooth = np.array([g_fun(kk / n) / n for kk in range(n)])
    wm_spike = [float(window_mass(Gs, d)) for d in deltas]
    wm_smooth = [float(window_mass(Gsmooth, d)) for d in deltas]
    return {"widths": ws, "discrete_check_w0.1_n4096_sup_err": err,
            "deltas": deltas.tolist(), "window_mass_spike": wm_spike,
            "window_mass_smooth": wm_smooth}


# ---------------------------------------------------------------------------
# E5  fixed vs moving spikes
# ---------------------------------------------------------------------------
def e5_fixed_vs_moving():
    ns = [2 ** p for p in range(3, 17)]
    out = {"ns": ns, "moving_half": [], "fixed_index_3": [], "sqrt_index": [], "moving_third": []}
    for n in ns:
        for key, j in (("moving_half", spike_index(n)), ("fixed_index_3", 3),
                       ("sqrt_index", int(math.isqrt(n))), ("moving_third", n // 3 + 1)):
            G, H = counterexample_coeffs(n, spike_at=j)
            F = solve_backward(G, H, 0.0)
            out[key].append(float(sample(F, n, 0.25)[0]))
    out["limits"] = {"moving_half": 2 / 3, "moving_third": (1 - 1 / 3) / (1 - 0.25),
                     "fixed_index_3": 0.0, "sqrt_index": 0.0}
    return out


# ---------------------------------------------------------------------------
# E6  ResNet backpropagation = backward recurrence; auxiliary loss = spike
# ---------------------------------------------------------------------------
A_, B_, Z0, TARGET, C_AUX, W_AUX = 1.3, 0.4, 0.5, 0.2, 0.0, 1.0


def f_res(z, a=A_, b=B_):
    return np.tanh(a * z + b)


def fz_res(z, a=A_, b=B_):
    return a * (1 - np.tanh(a * z + b) ** 2)


def fa_res(z, a=A_, b=B_):
    return z * (1 - np.tanh(a * z + b) ** 2)


def resnet_loss(n, a=A_, b=B_):
    z = np.empty(n + 1)
    z[0] = Z0
    for k in range(n):
        z[k + 1] = z[k] + f_res(z[k], a, b) / n
    j = n // 2
    return 0.5 * (z[n] - TARGET) ** 2 + 0.5 * W_AUX * (z[j] - C_AUX) ** 2, z


def resnet_backprop(n, a=A_, b=B_):
    """Adjoint lam_k = dL/dz_k via the backward recurrence; returns lam and dL/da."""
    _, z = resnet_loss(n, a, b)
    j = n // 2
    G = np.zeros(n)
    G[j] = W_AUX * (z[j] - C_AUX)                      # the spike: an O(1) auxiliary loss
    H = 1.0 + fz_res(z[:n], a, b) / n                 # h_n = -f_z(z_k)
    lam = np.array(solve_backward(list(G), list(H), z[n] - TARGET))
    dLda = float(np.sum(lam[1:] * fa_res(z[:n], a, b) / n))
    return lam, dLda, z


def resnet_continuous(jump=True):
    """Continuous adjoint (neural-ODE limit). With jump=True the adjoint jumps at t=1/2."""
    sol = solve_ivp(lambda t, z: f_res(z), (0, 1), [Z0], rtol=1e-12, atol=1e-14, dense_output=True)
    z = lambda t: sol.sol(t)[0]  # noqa: E731
    rhs = lambda t, y: [-fz_res(z(t)) * y[0], -y[0] * fa_res(z(t))]  # noqa: E731
    # integrate backward from 1 to 1/2: y = (lam, -int_t^1 lam f_a)
    s1 = solve_ivp(rhs, (1, 0.5), [z(1.0) - TARGET, 0.0], rtol=1e-12, atol=1e-14, dense_output=True)
    lam_half = s1.y[0, -1] + (W_AUX * (z(0.5) - C_AUX) if jump else 0.0)
    s2 = solve_ivp(rhs, (0.5, 0), [lam_half, s1.y[1, -1]], rtol=1e-12, atol=1e-14, dense_output=True)
    grad = s2.y[1, -1]

    def lam(t):
        t = np.atleast_1d(t)
        return np.where(t >= 0.5, s1.sol(np.clip(t, 0.5, 1))[0], s2.sol(np.clip(t, 0, 0.5))[0])

    return lam, float(grad), z


def e6_resnet():
    # finite-difference check of the backward recurrence (n = 64)
    n = 64
    _, g_bp, _ = resnet_backprop(n)
    eps = 1e-6
    g_fd = (resnet_loss(n, A_ + eps)[0] - resnet_loss(n, A_ - eps)[0]) / (2 * eps)
    _, g_jump, _ = resnet_continuous(jump=True)
    _, g_naive, _ = resnet_continuous(jump=False)
    ns = [2 ** p for p in range(3, 15)]
    err_jump, err_naive, grads = [], [], []
    for n in ns:
        _, g, _ = resnet_backprop(n)
        grads.append(g)
        err_jump.append(abs(g - g_jump))
        err_naive.append(abs(g - g_naive))
    slope = np.polyfit(np.log(ns[3:]), np.log(err_jump[3:]), 1)[0]
    return {"fd_check_n64": {"backprop": g_bp, "finite_difference": float(g_fd),
                             "abs_diff": abs(g_bp - g_fd)},
            "grad_continuous_with_jump": g_jump, "grad_continuous_naive": g_naive,
            "ns": ns, "grad_discrete": grads, "err_vs_jump": err_jump,
            "err_vs_naive": err_naive, "fitted_slope_jump": float(slope)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_counterexample(outdir, theme):
    t = set_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.28})
    # (a) coefficients
    ax = axes[0]
    for n, alpha in ((8, 0.45), (32, 0.7), (128, 1.0)):
        j = spike_index(n)
        ax.bar(j / n, n, width=1 / n, align="edge", color=t["coral"], alpha=alpha,
               edgecolor="none", label=rf"$g_n$, $n={n}$")
    xs = np.linspace(0, 0.9, 400)
    ax.plot(xs, 1 / (1 - xs), color=t["teal"], lw=1.8, label=r"$h(x)=1/(1-x)$")
    _, H = counterexample_coeffs(128)
    xx, yy = step_xy([128 * (1 - Hk) for Hk in H] + [0], 128)
    m = xx <= 0.9
    ax.plot(xx[m], yy[m], color=t["blue"], lw=1.0, alpha=0.9, label=r"$h_n$, $n=128$")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 300)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel(r"$x$")
    ax.set_title(r"(a) scaled coefficients: $g_n\to 0$ pointwise", loc="left")
    ax.text(0.44, 150, "each spike\nhas area 1", ha="right", va="center", color=t["coral"], fontsize=8.2)
    ax.legend(loc="upper right", handlelength=1.4, fontsize=7.6)
    # (b) f_n
    ax = axes[1]
    for n, alpha in ((16, 0.35), (64, 0.6), (512, 1.0)):
        G, H = counterexample_coeffs(n)
        F = solve_backward(G, H, 0.0)
        xx, yy = step_xy(F, n)
        ax.plot(xx, yy, color=t["blue"], lw=1.3, alpha=alpha, label=rf"$f_n$, $n={n}$")
    xl = np.linspace(0, 0.5, 200)
    ax.plot(xl, 0.5 / (1 - xl), color=t["teal"], lw=2.0, ls=(0, (4, 2)), label="pointwise limit")
    ax.plot([0.5, 1], [0, 0], color=t["teal"], lw=2.0, ls=(0, (4, 2)))
    ax.plot([0, 1], [0, 0], color=t["amber"], lw=1.6, zorder=0, label=r"ODE prediction $y\equiv 0$")
    ax.scatter([0.5], [1.0], s=18, color=t["teal"], zorder=5)
    ax.annotate("jump of size 1", xy=(0.505, 0.85), xytext=(0.62, 0.93), color=t["coral"],
                arrowprops=dict(arrowstyle="->", color=t["coral"], lw=1.0), fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(r"$x$")
    ax.set_title(r"(b) $f_n(x)=F_n(\lfloor nx\rfloor)$", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), handlelength=1.4, fontsize=7.6)
    save(fig, "fig1_counterexample", outdir, theme)


def fig_rates(outdir, theme, R):
    t = set_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ns = np.array(R["e2"]["ns"])
    ax.loglog(ns, R["e2"]["sup_err_away_from_half"], "o-", color=t["blue"], ms=3, lw=1.4,
              label=rf"counterexample (slope ${R['e2']['fitted_slope']:.2f}$)")
    ns3 = np.array(R["e3"]["ns"])
    ax.loglog(ns3, R["e3"]["sup_err"], "s-", color=t["violet"], ms=3, lw=1.4,
              label=rf"generic test (slope ${R['e3']['fitted_slope']:.2f}$)")
    ax.loglog(ns, 2.0 / ns, color=t["muted"], lw=0.9, ls=":", label=r"$O(n^{-1})$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("sup error away from atoms")
    ax.set_title("(a) convergence to the measure limit", loc="left")
    ax.legend(loc="upper right", fontsize=7.6)
    ax = axes[1]
    E = R["e5"]
    ns5 = np.array(E["ns"])
    for key, col, lab in (("moving_half", t["coral"], r"spike at $\lfloor n/2\rfloor+1$"),
                          ("moving_third", t["teal"], r"spike at $\lfloor n/3\rfloor+1$"),
                          ("sqrt_index", t["amber"], r"spike at $\lfloor\sqrt{n}\rfloor$"),
                          ("fixed_index_3", t["blue"], r"spike at $k=3$")):
        ax.semilogx(ns5, E[key], "o-", ms=3, lw=1.4, color=col, label=lab)
    for v in (2 / 3, 8 / 9):
        ax.axhline(v, color=t["muted"], lw=0.8, ls=":")
    ax.text(ns5[-1], 2 / 3 + 0.03, r"$2/3$", ha="right", fontsize=8, color=t["muted"])
    ax.text(ns5[-1], 8 / 9 + 0.03, r"$8/9$", ha="right", fontsize=8, color=t["muted"])
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$f_n(1/4)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) only spikes that move with $n$ survive", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=7.6)
    save(fig, "fig2_rates_and_spikes", outdir, theme)


def fig_general(outdir, theme):
    t = set_theme(theme)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    xs = np.linspace(0.0, 0.999, 1500)
    for n, alpha in ((24, 0.35), (96, 0.6), (1536, 1.0)):
        G, H = e3_coeffs(n)
        F = solve_backward(list(G), list(H), THETA)
        xx, yy = step_xy(F, n)
        ax.plot(xx, yy, color=t["blue"], lw=1.2, alpha=alpha, label=rf"$f_n$, $n={n}$")
    ax.plot(xs, measure_limit(xs, phi_e3, density=g_fun, atoms=ATOMS, Theta=THETA),
            color=t["teal"], lw=2.0, ls=(0, (4, 2)), label=r"measure-valued limit (with atoms)")
    ax.plot(xs, measure_limit(xs, phi_e3, density=g_fun, atoms=(), Theta=THETA),
            color=t["amber"], lw=1.6, label=r"ODE $y'=-g+hy$ (pointwise prediction)")
    for (c, a) in ATOMS:
        ax.axvline(c, color=t["coral"], lw=0.8, ls=":")
        ax.text(c + 0.008, 1.62, rf"atom ${a}$ at ${c}$", color=t["coral"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$x$")
    ax.set_title(r"Smooth forcing, perturbed $h_n$, and two moving spikes: the limit is the measure-driven equation",
                 loc="left", fontsize=9)
    ax.legend(loc="lower left", ncol=2)
    save(fig, "fig3_general_theorem", outdir, theme)


def fig_smoothing(outdir, theme, R):
    t = set_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.3, "width_ratios": [1.35, 1]})
    ax = axes[0]
    xs = np.linspace(0.0, 0.98, 1200)
    cmap = plt.get_cmap("viridis" if theme == "light" else "plasma")
    ws = R["e4"]["widths"]
    for i, w in enumerate(ws):
        ax.plot(xs, smoothed_limit(xs, w), color=cmap(0.15 + 0.7 * i / (len(ws) - 1)), lw=1.4,
                label=rf"$w={w}$")
    ax.plot(xs, counterexample_limit(xs), color=t["coral"], lw=1.8, ls=(0, (4, 2)), label=r"$w\to 0$ (jump)")
    ax.set_xlabel(r"$x$")
    ax.set_title(r"(a) spreading the unit mass over width $w$", loc="left")
    ax.legend(loc="upper right", ncol=1, fontsize=7.4)
    ax.set_xlim(0, 1)
    ax = axes[1]
    d = np.array(R["e4"]["deltas"])
    ax.loglog(d, R["e4"]["window_mass_spike"], "o-", color=t["coral"], ms=3, lw=1.4, label="spike (counterexample)")
    ax.loglog(d, R["e4"]["window_mass_smooth"], "s-", color=t["teal"], ms=3, lw=1.4, label=r"smooth $g$")
    ax.set_xlabel(r"window half-width $\delta$")
    ax.set_ylabel(r"$\max_x \nu_n([x-\delta,x+\delta])$")
    ax.set_title("(b) the mass test that pointwise checks miss", loc="left")
    ax.legend(loc="lower right")
    save(fig, "fig4_smoothing_and_mass", outdir, theme)


def fig_adjoint(outdir, theme, R):
    t = set_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    for n, alpha, mk in ((16, 0.45, "o"), (64, 0.7, "s"), (512, 1.0, None)):
        lam, _, _ = resnet_backprop(n)
        tt = np.arange(n + 1) / n
        if mk:
            ax.plot(tt, lam, mk, ms=2.6, color=t["blue"], alpha=alpha, label=rf"backprop $\lambda_k$, $n={n}$")
        else:
            ax.plot(tt, lam, color=t["blue"], lw=1.2, alpha=alpha, label=rf"backprop $\lambda_k$, $n={n}$")
    lam_j, _, _ = resnet_continuous(jump=True)
    lam_n, _, _ = resnet_continuous(jump=False)
    tt = np.linspace(0, 1, 800)
    l1 = lam_j(tt)
    l1[np.argmin(np.abs(tt - 0.5))] = np.nan
    ax.plot(tt, l1, color=t["teal"], lw=2.0, ls=(0, (4, 2)), label="adjoint ODE with jump", zorder=6)
    ax.plot(tt, lam_n(tt), color=t["amber"], lw=1.6, label="naive adjoint ODE")
    ax.set_xlabel(r"depth $t=k/n$")
    ax.set_ylabel(r"$\lambda=\partial L/\partial z$")
    ax.set_title("(a) backprop through a ResNet", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=7.4)
    ax = axes[1]
    ns = np.array(R["e6"]["ns"])
    ax.loglog(ns, R["e6"]["err_vs_jump"], "o-", color=t["teal"], ms=3, lw=1.4,
              label="vs. adjoint with jump")
    ax.loglog(ns, R["e6"]["err_vs_naive"], "s-", color=t["amber"], ms=3, lw=1.4, label="vs. naive adjoint")
    ax.set_xlabel(r"depth $n$")
    ax.set_ylabel(r"$|\partial L/\partial a$ error$|$")
    ax.set_title(rf"(b) gradient error (slope ${R['e6']['fitted_slope_jump']:.2f}$ vs. jump)", loc="left")
    ax.legend(loc="lower left", fontsize=7.4)
    save(fig, "fig5_resnet_adjoint", outdir, theme)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "figures"))
    args = ap.parse_args()
    R = {"e1": e1_exact(), "e2": e2_rates(), "e3": e3_general(), "e4": e4_smoothing(),
         "e5": e5_fixed_vs_moving(), "e6": e6_resnet()}
    with open(os.path.join(here, "..", "results.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    for theme in ("light", "dark"):
        fig_counterexample(args.outdir, theme)
        fig_rates(args.outdir, theme, R)
        fig_general(args.outdir, theme)
        fig_smoothing(args.outdir, theme, R)
        fig_adjoint(args.outdir, theme, R)
    print(json.dumps({k: (v if k != "e2" else {kk: v[kk] for kk in ("fitted_slope",)}) for k, v in R.items()},
                     indent=1)[:6000])


if __name__ == "__main__":
    main()

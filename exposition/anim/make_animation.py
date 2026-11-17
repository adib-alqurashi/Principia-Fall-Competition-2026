"""Exposition animation for MATHDB 354674.

Renders a ~93 s 1920x1080 explainer video with matplotlib and pipes raw frames
into ffmpeg. Usage:
    python3 make_animation.py video OUT.mp4 [--workers N]
    python3 make_animation.py stills OUTDIR
    python3 make_animation.py frame T OUT.png      (debug: one frame at time T)
"""
import os
import sys
import math
import subprocess
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Ellipse, FancyBboxPatch

# --------------------------------------------------------------------------- style
FONT_DIR = "/usr/share/fonts/opentype/inter"
for _f in ["Inter-Regular.otf", "Inter-Medium.otf", "Inter-SemiBold.otf",
           "Inter-Bold.otf", "Inter-Italic.otf", "InterDisplay-SemiBold.otf",
           "InterDisplay-Bold.otf", "InterDisplay-Medium.otf"]:
    p = os.path.join(FONT_DIR, _f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)

BG = "#0B1220"
PANEL = "#111A2E"
GRID = "#24304A"
TEXT = "#E6EAF2"
MUTED = "#8A96AD"
BLUE = "#5B8DEF"
CORAL = "#FF6B6B"
TEAL = "#2EC4B6"
AMBER = "#F4B942"
VIOLET = "#A78BFA"

W, H, DPI, FPS = 1920, 1080, 100, 30

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "text.color": TEXT,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.linewidth": 1.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 10,
    "ytick.major.pad": 10,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.formatter.use_mathtext": True,
})

TITLE_PT = 34     # ~47 px
BODY_PT = 22      # ~31 px
SMALL_PT = 21     # ~29 px
MATH_PT = 26
LW = 3.6          # ~5 px

# --------------------------------------------------------------------------- timeline
SCENES = [  # (name, duration in seconds)
    ("title", 5.5),
    ("recurrence", 12.5),
    ("coefficients", 14.0),
    ("convergence", 21.0),
    ("fixed_moving", 12.0),
    ("measure", 12.0),
    ("sciml", 11.5),
    ("end", 5.5),
]
STARTS = np.cumsum([0.0] + [d for _, d in SCENES])
TOTAL = float(STARTS[-1])
FADE = 0.45


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(u):
    u = clamp(u)
    return clamp(u * u * u * (u * (6 * u - 15) + 10))  # smootherstep


def prog(t, a, b):
    if b <= a:
        return 1.0 if t >= b else 0.0
    return ease((t - a) / (b - a))


def log_interp(lo, hi, p):
    return lo * (hi / lo) ** p


# --------------------------------------------------------------------------- math
def F_main(n):
    """Closed form of the counterexample: F_n(k) = ceil(n/2)/(n-k+1) for k <= j_n, else 0."""
    k = np.arange(n + 1, dtype=float)
    j = n // 2 + 1
    c = (n + 1) // 2
    F = np.where(k <= j, c / (n - k + 1), 0.0)
    return F


def F_main_recurrence(n):
    """Same values computed by running the recurrence backward (used for checks)."""
    F = np.zeros(n + 1)
    j = n // 2 + 1
    for k in range(n - 1, -1, -1):
        Hk = (n - k) / (n - k + 1)
        Gk = 1.0 if (n >= 4 and k == j) else 0.0
        F[k] = Gk + Hk * F[k + 1]
    return F


def F_fixed(n, m0=3):
    k = np.arange(n + 1, dtype=float)
    return np.where(k <= m0, (n - m0 + 1) / (n - k + 1), 0.0)


def f_limit_xy():
    x1 = np.linspace(0.0, 0.5, 300)
    return x1, 1.0 / (2.0 * (1.0 - x1))


def f_smoothed(x, w):
    """ODE solution y' = -g_w + y/(1-x), y bounded at 1, g_w = 1/w on |s-1/2|<w/2."""
    a, b = 0.5 - w / 2, 0.5 + w / 2
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    left = x <= a
    mid = (x > a) & (x < b)
    out[left] = 0.5 / (1.0 - x[left])
    xm = x[mid]
    out[mid] = ((b - xm) - (b * b - xm * xm) / 2.0) / (w * (1.0 - xm))
    return out


def sciml_adjoint(n, spike=True, z0=0.2, wt=1.0, c=-0.5):
    """1-D ResNet z_{k+1} = z_k + f(z_k)/n, f = tanh(1.5 z + 0.4).
    Loss L = 0.5*z_n^2 + wt*0.5*(z_j - c)^2 with j = n//2. Returns t_k, lambda_k."""
    f = lambda z: np.tanh(1.5 * z + 0.4)
    fp = lambda z: 1.5 / np.cosh(1.5 * z + 0.4) ** 2
    z = np.empty(n + 1)
    z[0] = z0
    for k in range(n):
        z[k + 1] = z[k] + f(z[k]) / n
    lam = np.empty(n + 1)
    lam[n] = z[n]
    j = n // 2
    for k in range(n - 1, -1, -1):
        lam[k] = lam[k + 1] * (1.0 + fp(z[k]) / n)
        if spike and k == j:
            lam[k] += wt * (z[j] - c)
    return np.arange(n + 1) / n, lam


# precomputed curves for the SciML scene
T_FINE, LAM_TRUE = sciml_adjoint(20000, spike=True)
_, LAM_NAIVE = sciml_adjoint(20000, spike=False)
T_40, LAM_40 = sciml_adjoint(40, spike=True)


# --------------------------------------------------------------------------- helpers
def style_ax(ax, alpha=1.0, grid=True):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_alpha(alpha)
    ax.tick_params(colors=MUTED, labelsize=21)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(alpha)
    if grid:
        ax.grid(True, color=GRID, lw=1.2, alpha=0.55 * alpha)
        ax.set_axisbelow(True)
    ax.patch.set_alpha(alpha)


def header(fig, num, text, alpha=1.0):
    fig.text(0.06, 0.905, f"{num:02d}", fontsize=24, color=VIOLET, alpha=alpha,
             fontweight="semibold", va="center")
    fig.text(0.098, 0.905, text, fontsize=TITLE_PT, color=TEXT, alpha=alpha,
             family=["Inter Display", "Inter", "DejaVu Sans"], fontweight="semibold",
             va="center")


def progress_bar(fig, t):
    frac = clamp(t / TOTAL)
    fig.add_artist(Line2D([0, 1], [0.004, 0.004], transform=fig.transFigure,
                          color=GRID, lw=4, solid_capstyle="butt"))
    fig.add_artist(Line2D([0, frac], [0.004, 0.004], transform=fig.transFigure,
                          color=VIOLET, lw=4, alpha=0.85, solid_capstyle="butt"))


def fade_overlay(fig, alpha):
    if alpha > 0.002:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 facecolor=BG, edgecolor="none", alpha=clamp(alpha),
                                 zorder=1000))


def step_xy(F):
    n = len(F) - 1
    return np.arange(n + 1) / n, F


def draw_limit(ax, alpha=1.0, lw=LW, dots=True, z=4):
    x1, y1 = f_limit_xy()
    ax.plot(x1, y1, color=TEAL, lw=lw, ls=(0, (5, 3)), alpha=alpha, zorder=z)
    ax.plot([0.5, 1.0], [0, 0], color=TEAL, lw=lw, ls=(0, (5, 3)), alpha=alpha, zorder=z)
    if dots:
        ax.scatter([0.5], [1.0], s=170, color=TEAL, zorder=z + 2, alpha=alpha, lw=0)
        ax.scatter([0.5], [0.0], s=170, facecolor=PANEL, edgecolor=TEAL, lw=3,
                   zorder=z + 2, alpha=alpha)


def nice_n(t, a, b, lo, hi):
    return int(round(log_interp(lo, hi, prog(t, a, b))))


# --------------------------------------------------------------------------- scenes
def scene_title(fig, t, d):
    # faint background motif: the jump limit and a coarse f_n
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 4.2)
    ax.axis("off")
    a_bg = 0.16 * prog(t, 0.0, 1.5)
    F = F_main(28)
    x, y = step_xy(F)
    ax.step(x, y, where="post", color=BLUE, lw=5, alpha=a_bg)
    draw_limit(ax, alpha=a_bg * 0.9, lw=4, dots=False)

    pa = prog(t, 0.25, 1.35)
    fig.text(0.5, 0.585 - 0.02 * (1 - pa), "When a discretized ODE\nforgets its equation",
             ha="center", va="center", fontsize=58, linespacing=1.12,
             family=["Inter Display", "Inter"], fontweight="bold", color=TEXT, alpha=pa)
    pl = prog(t, 1.0, 2.1)
    L = 0.30 * pl
    fig.add_artist(Line2D([0.5 - L / 2, 0.5 + L / 2], [0.425, 0.425],
                          transform=fig.transFigure, color=CORAL, lw=5,
                          solid_capstyle="round", alpha=pl))
    ps = prog(t, 0.9, 2.0)
    fig.text(0.5, 0.36, "A backward recurrence, a moving spike, and a jump",
             ha="center", va="center", fontsize=26, color=MUTED, alpha=ps)
    fig.text(0.5, 0.305, "MATHDB 354674", ha="center", va="center", fontsize=22,
             color=TEAL, alpha=ps, fontweight="semibold")


def scene_recurrence(fig, t, d):
    n, j = 12, 12 // 2 + 1
    F = F_main(n)
    header(fig, 1, "The recurrence, run backward", alpha=prog(t, 0.0, 0.6))
    pf = prog(t, 0.3, 1.1)
    fig.text(0.06, 0.80, r"$F_n(k)\;=\;G_n(k)\;+\;H_n(k)\,F_n(k+1),\qquad F_n(n)=0$",
             fontsize=31, alpha=pf, va="center")
    fig.text(0.06, 0.735,
             r"$H_n(k)=(n-k)/(n-k+1)<1$" + " shrinks values;   " +
             r"$G_n(k)=1$" + " only at " + r"$k=j_n=\lfloor n/2\rfloor+1$",
             fontsize=BODY_PT, color=MUTED, alpha=pf, va="center")

    ax = fig.add_axes([0.075, 0.13, 0.60, 0.52])
    style_ax(ax, alpha=prog(t, 0.4, 1.2))
    ax.set_xlim(-0.6, n + 0.6)
    ax.set_ylim(-0.06, 1.28)
    ax.set_xticks(range(0, n + 1, 1))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("k   (time step / layer index)", fontsize=SMALL_PT, color=MUTED, labelpad=12)

    # reveal schedule: k = n, n-1, ..., 0 between t0 and t1
    t0, t1 = 1.6, 9.4
    per = (t1 - t0) / (n + 1)
    revealed = []
    for idx, k in enumerate(range(n, -1, -1)):
        tk = t0 + idx * per
        p = prog(t, tk, tk + 0.45)
        if p > 0:
            revealed.append((k, p))
    # backward arrow
    pa = prog(t, 1.0, 1.8) * (1 - prog(t, 10.0, 10.8))
    if pa > 0:
        ax.annotate("", xy=(0.3, 1.2), xytext=(n - 0.3, 1.2),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.45",
                                    color=VIOLET, lw=3, alpha=pa))
        ax.text(n / 2, 1.235, "computed from right to left", ha="center", va="bottom",
                fontsize=SMALL_PT, color=VIOLET, alpha=pa)
    # step curve through revealed points
    if revealed:
        ks = sorted(k for k, _ in revealed)
        kmin = ks[0]
        xs = np.arange(kmin, n + 1)
        ax.step(xs, F[kmin:], where="post", color=BLUE, lw=2.2, alpha=0.45, zorder=2)
    current = None
    for k, p in revealed:
        val = F[k] * (0.85 + 0.15 * p) if k < j else F[k]
        is_spike = k == j
        col = CORAL if is_spike else BLUE
        ax.plot([k, k], [0, val], color=col, lw=3.2, alpha=p, zorder=3)
        ax.scatter([k], [val], s=(60 + 140 * p), color=col, zorder=5, lw=0, alpha=p)
        if p < 1:
            current = k
        if is_spike:
            ring = prog(t, t0 + (n - j) * per, t0 + (n - j) * per + 1.0)
            if 0 < ring < 1:
                ax.scatter([k], [val], s=200 + 2400 * ring, facecolor="none",
                           edgecolor=CORAL, lw=3, alpha=(1 - ring), zorder=4)
            ax.text(k + 0.25, val + 0.07, "+1 spike", color=CORAL, fontsize=SMALL_PT,
                    fontweight="semibold", alpha=p, va="bottom")
    if current is None and revealed:
        current = min(k for k, _ in revealed) if t < t1 + 0.5 else None

    # right panel: live computation
    xr = 0.715
    pr = prog(t, 1.2, 1.9) * (1 - prog(t, t1 + 0.3, t1 + 0.8))
    fig.text(xr, 0.62, "live step", fontsize=SMALL_PT, color=MUTED, alpha=pr)
    if current is not None and t < t1 + 0.6:
        k = current
        if k == n:
            line1 = rf"$F({n}) = 0$"
            line2 = "terminal value"
        else:
            Gk = 1 if k == j else 0
            line1 = rf"$F({k}) = {Gk} + \frac{{{n - k}}}{{{n - k + 1}}}\cdot{F[k + 1]:.3f}$"
            line2 = rf"$= {F[k]:.3f}$"
        fig.text(xr, 0.55, line1, fontsize=30, alpha=pr, va="center")
        fig.text(xr, 0.47, line2, fontsize=30, alpha=pr, va="center",
                 color=CORAL if k == j else TEXT)
    pc = prog(t, 9.8, 10.9)
    if pc > 0:
        fig.text(xr, 0.62, "closed form", fontsize=SMALL_PT, color=MUTED, alpha=pc)
        fig.text(xr, 0.535, r"$F_n(k)=\frac{\lceil n/2\rceil}{n-k+1}\ \ (k\leq j_n)$",
                 fontsize=33, alpha=pc, va="center")
        fig.text(xr, 0.435, r"$F_n(k)=0\ \ (k>j_n)$", fontsize=33, alpha=pc, va="center")
        fig.text(xr, 0.345, "machine-checked in Lean", fontsize=SMALL_PT, color=TEAL,
                 alpha=pc, va="center")
    fig.text(0.075, 0.665, rf"$n={n},\ \ j_n={j}$", fontsize=24, color=MUTED,
             alpha=prog(t, 0.6, 1.3), va="center")


def scene_coefficients(fig, t, d):
    header(fig, 2, "Both coefficients converge pointwise", alpha=prog(t, 0, 0.6))
    n = nice_n(t, 1.2, 10.0, 4, 60)
    j = n // 2 + 1
    pin = prog(t, 0.3, 1.1)

    # --- left: g_n spike
    ax = fig.add_axes([0.07, 0.235, 0.40, 0.52])
    style_ax(ax, alpha=pin)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 66)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "¼", "½", "¾", "1"])
    ax.set_yticks([0, 20, 40, 60])
    ax.text(0.0, 1.035, r"$g_n(x)=n\,G_n(\lfloor nx\rfloor)$", transform=ax.transAxes,
            fontsize=MATH_PT, alpha=pin, va="bottom")
    x0, x1 = j / n, (j + 1) / n
    ax.add_patch(Rectangle((x0, 0), x1 - x0, n, facecolor=CORAL, edgecolor=CORAL,
                           lw=2, alpha=0.9 * pin, zorder=3))
    ax.plot([0, x0, x0, x1, x1, 1], [0, 0, n, n, 0, 0], color=CORAL, lw=2.6,
            alpha=pin, zorder=4)
    # probe at x = 0.3
    ax.axvline(0.3, color=MUTED, lw=1.8, ls=(0, (2, 3)), alpha=0.8 * pin)
    ax.scatter([0.3], [0], s=150, color=TEXT, zorder=6, alpha=pin, clip_on=False)
    ax.text(0.285, 4.0, r"$g_n(0.3)=0$", fontsize=24, color=TEXT, alpha=pin, ha="right")
    # live numbers
    tx = 0.60
    ax.text(tx, 0.90, rf"height $= {n}$", transform=ax.transAxes, fontsize=24, alpha=pin)
    ax.text(tx, 0.81, rf"width $= 1/{n}$", transform=ax.transAxes, fontsize=24, alpha=pin)
    ax.text(tx, 0.72, r"area $= 1$", transform=ax.transAxes, fontsize=24, color=CORAL,
            fontweight="semibold", alpha=pin)

    # --- right: h_n -> 1/(1-x)
    ax2 = fig.add_axes([0.56, 0.235, 0.40, 0.52])
    style_ax(ax2, alpha=pin)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 12)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_xticklabels(["0", "¼", "½", "¾", "1"])
    ax2.set_yticks([0, 4, 8, 12])
    ax2.text(0.0, 1.035, r"$h_n(x)=n\,(1-H_n(\lfloor nx\rfloor))\ \to\ 1/(1-x)$",
             transform=ax2.transAxes, fontsize=MATH_PT, alpha=pin, va="bottom")
    xs = np.linspace(0, 0.93, 400)
    ax2.plot(xs, 1 / (1 - xs), color=TEAL, lw=LW, ls=(0, (5, 3)), alpha=pin, zorder=3)
    k = np.arange(0, n)
    hx = k / n
    hv = n / (n - k + 1.0)
    ax2.step(np.append(hx, 1.0), np.append(hv, hv[-1]), where="post", color=BLUE,
             lw=3.0, alpha=pin, zorder=4)
    ax2.text(0.05, 0.86, rf"$n={n}$", transform=ax2.transAxes, fontsize=28,
             color=BLUE, alpha=pin, fontweight="semibold")

    pcap = prog(t, 10.4, 11.6)
    fig.text(0.07, 0.115,
             r"Pointwise limits:   $g\equiv 0,\ \ h(x)=1/(1-x)$   $\Rightarrow$   limiting ODE   $y'=y/(1-x)$",
             fontsize=BODY_PT + 1, color=TEXT, alpha=pcap, va="center")
    fig.text(0.07, 0.055,
             r"Its only solution continuous at $x=1$ is  $y\equiv 0$:  the limit the conjecture predicts.",
             fontsize=BODY_PT + 1, color=AMBER, alpha=prog(t, 11.2, 12.3), va="center")


def scene_convergence(fig, t, d):
    header(fig, 3, "What the recurrence actually converges to", alpha=prog(t, 0, 0.6))
    pin = prog(t, 0.2, 1.0)
    ax = fig.add_axes([0.075, 0.13, 0.60, 0.64])
    style_ax(ax, alpha=pin)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "¼", "½", "¾", "1"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("x", fontsize=SMALL_PT, color=MUTED, labelpad=8)

    # naive prediction
    pam = prog(t, 0.8, 2.2)
    ax.plot([0, pam], [0, 0], color=AMBER, lw=LW + 0.6, alpha=pin, zorder=3,
            solid_capstyle="round")
    ax.text(0.62, 0.05, r"conjecture predicts  $y\equiv 0$", color=AMBER,
            fontsize=BODY_PT, alpha=pam, va="bottom")

    # f_n growing
    pn = prog(t, 2.8, 15.0)
    n = int(round(log_interp(4, 2000, pn)))
    show_fn = prog(t, 2.6, 3.2)
    reveal = prog(t, 15.4, 17.0)
    F = F_main(n)
    x, y = step_xy(F)
    ax.step(x, y, where="post", color=BLUE, lw=LW, alpha=show_fn * (1 - 0.35 * reveal),
            zorder=5)
    # probe at x = 1/4
    k4 = n // 4
    v4 = F[k4]
    ax.axvline(0.25, color=MUTED, lw=1.6, ls=(0, (2, 3)), alpha=0.8 * show_fn)
    ax.scatter([0.25], [v4], s=190, color=BLUE, edgecolor=TEXT, lw=2, zorder=8,
               alpha=show_fn)
    ax.scatter([0.25], [0], s=150, color=AMBER, zorder=8, alpha=show_fn)

    # true limit + jump
    if reveal > 0:
        draw_limit(ax, alpha=reveal, z=6)
        pj = prog(t, 16.6, 18.0)
        ax.annotate("", xy=(0.5, 0.0), xytext=(0.5, 1.0),
                    arrowprops=dict(arrowstyle="<|-|>,head_length=0.7,head_width=0.35",
                                    color=CORAL, lw=3, alpha=pj, shrinkA=8, shrinkB=8),
                    zorder=9)
        ax.text(0.515, 0.52, "jump = 1", color=CORAL, fontsize=BODY_PT + 2,
                fontweight="semibold", alpha=pj, va="center")

    # right panel
    xr = 0.715
    fig.text(xr, 0.71, rf"$n = {n}$", fontsize=44, color=BLUE, alpha=show_fn, va="center")
    fig.text(xr, 0.605, rf"$f_n(1/4) = {v4:.4f}$", fontsize=31, alpha=show_fn, va="center")
    fig.text(xr, 0.535, r"$\to\ 2/3$" + "   (not 0)", fontsize=28,
             alpha=prog(t, 9.0, 10.5), va="center", color=TEXT)
    fig.text(xr, 0.47, "predicted by the conjecture:  0", fontsize=BODY_PT, color=AMBER,
             alpha=show_fn, va="center")
    if reveal > 0:
        fig.text(xr, 0.37, "true limit", fontsize=SMALL_PT, color=TEAL, alpha=reveal,
                 va="center")
        fig.text(xr, 0.30, r"$f(x)=\frac{1}{2(1-x)}$", fontsize=31, alpha=reveal, va="center")
        fig.text(xr + 0.155, 0.30, r"$(x\leq 1/2)$", fontsize=31, alpha=reveal, va="center")
        fig.text(xr, 0.225, r"$f(x)=0$", fontsize=31, alpha=reveal, va="center")
        fig.text(xr + 0.155, 0.225, r"$(x>1/2)$", fontsize=31, alpha=reveal, va="center")
        pm = prog(t, 18.0, 19.2)
        fig.text(xr, 0.13, "The limit exists, but it is not\nthe ODE solution.",
                 fontsize=BODY_PT, color=CORAL, alpha=pm, va="center", linespacing=1.3,
                 fontweight="semibold")


def scene_fixed_moving(fig, t, d):
    header(fig, 4, "A fixed spike washes out, a moving one does not",
           alpha=prog(t, 0, 0.6))
    pin = prog(t, 0.2, 1.0)
    n = nice_n(t, 1.4, 9.0, 6, 400)
    for i, (pos, F, title, spike_x, lim_kind) in enumerate([
        ([0.07, 0.235, 0.40, 0.52], F_fixed(n, 3), "spike at fixed index  k = 3",
         3 / n, "zero"),
        ([0.56, 0.235, 0.40, 0.52], F_main(n), r"spike at  $j_n=\lfloor n/2\rfloor+1$",
         (n // 2 + 1) / n, "jump"),
    ]):
        ax = fig.add_axes(pos)
        style_ax(ax, alpha=pin)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.08, 1.2)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "¼", "½", "¾", "1"])
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(0.0, 1.035, title, transform=ax.transAxes, fontsize=BODY_PT + 2,
                alpha=pin, va="bottom", color=TEXT)
        if lim_kind == "zero":
            ax.plot([0, 1], [0, 0], color=TEAL, lw=LW, ls=(0, (5, 3)), alpha=pin, zorder=3)
        else:
            draw_limit(ax, alpha=pin, z=3)
        x, y = step_xy(F)
        ax.step(x, y, where="post", color=BLUE, lw=LW, alpha=pin, zorder=5)
        ax.axvline(spike_x, color=CORAL, lw=2.4, alpha=0.9 * pin, zorder=4)
        ax.scatter([spike_x], [1.12], marker="v", s=230, color=CORAL, alpha=pin,
                   zorder=7, clip_on=False)
        lab = rf"position $3/n={spike_x:.3f}$" if lim_kind == "zero" else \
            rf"position $j_n/n={spike_x:.3f}$"
        ax.text(0.40 if lim_kind == "zero" else 0.03, 0.80, lab, transform=ax.transAxes,
                fontsize=24, color=CORAL, alpha=pin)
        ax.text(0.97, 0.13, rf"$n={n}$", transform=ax.transAxes, ha="right",
                fontsize=26, color=BLUE, alpha=pin)
    pcap = prog(t, 9.3, 10.6)
    fig.text(0.07, 0.115,
             r"Fixed spike: its position $3/n\to 0$, so its influence disappears from every $x>0$.",
             fontsize=BODY_PT + 1, color=TEXT, alpha=pcap, va="center")
    fig.text(0.07, 0.055,
             r"Moving spike: $j_n/n\to 1/2$ and it keeps mass 1. That mass is what leaves the jump.",
             fontsize=BODY_PT + 1, color=CORAL, alpha=prog(t, 10.0, 11.1), va="center",
             fontweight="semibold")


def scene_measure(fig, t, d):
    header(fig, 5, "The fix: follow the mass, not the pointwise values",
           alpha=prog(t, 0, 0.6))
    pin = prog(t, 0.2, 1.0)
    wv = log_interp(0.5, 0.004, prog(t, 1.5, 9.5))
    a, b = 0.5 - wv / 2, 0.5 + wv / 2

    # forcing density
    ax1 = fig.add_axes([0.07, 0.575, 0.48, 0.19])
    style_ax(ax1, alpha=pin)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 24)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax1.set_xticklabels([])
    ax1.set_yticks([0, 10, 20])
    hgt = 1 / wv
    ax1.add_patch(Rectangle((a, 0), wv, min(hgt, 24), facecolor=CORAL, alpha=0.85 * pin,
                            edgecolor=CORAL, lw=2, zorder=3))
    if hgt > 24:
        ax1.scatter([0.5], [23.0], marker="^", s=220, color=CORAL, zorder=6, alpha=pin)
    ax1.text(0.0, 1.06, "forcing: mass 1 spread over width " + rf"$w={wv:.3f}$",
             transform=ax1.transAxes, fontsize=BODY_PT, alpha=pin, color=TEXT, va="bottom")

    # solution
    ax2 = fig.add_axes([0.07, 0.12, 0.48, 0.40])
    style_ax(ax2, alpha=pin)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.08, 1.2)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax2.set_xticklabels(["0", "¼", "½", "¾", "1"])
    ax2.set_yticks([0, 0.5, 1.0])
    draw_limit(ax2, alpha=pin, z=3)
    xs = np.linspace(0, 0.999, 1500)
    ax2.plot(xs, f_smoothed(xs, wv), color=BLUE, lw=LW, alpha=pin, zorder=5)
    ax2.text(0.62, 0.80, "ODE solution with\nsmoothed forcing", transform=ax2.transAxes,
             fontsize=BODY_PT, color=BLUE, alpha=pin, linespacing=1.25)

    # theorem block
    xr = 0.605
    ptb = prog(t, 0.9, 2.0)
    fig.add_artist(FancyBboxPatch((xr - 0.012, 0.15), 0.37, 0.62,
                                  boxstyle="round,pad=0.006,rounding_size=0.012",
                                  transform=fig.transFigure, facecolor=PANEL,
                                  edgecolor=VIOLET, lw=2, alpha=ptb))
    fig.text(xr + 0.01, 0.725, "Measure-valued limit", fontsize=26, color=VIOLET,
             fontweight="semibold", alpha=ptb, va="center")
    fig.text(xr + 0.01, 0.64, r"$\nu_n=\sum_m G_n(m)\,\delta_{m/n}\ \rightharpoonup\ \nu$",
             fontsize=29, alpha=prog(t, 1.6, 2.6), va="center")
    fig.text(xr + 0.01, 0.53,
             r"$f(x)=\int_{[x,1]}\Phi(x,s)\,d\nu(s)+\Theta\,\Phi(x,1)$",
             fontsize=28, alpha=prog(t, 2.6, 3.6), va="center")
    fig.text(xr + 0.01, 0.43, r"$\Phi(x,s)=\exp(-\int_x^s h)=(1-s)/(1-x)$",
             fontsize=28, alpha=prog(t, 3.6, 4.6), va="center")
    pb = prog(t, 6.0, 7.2)
    fig.text(xr + 0.01, 0.325, "smooth forcing", fontsize=BODY_PT, alpha=pb, va="center")
    fig.text(xr + 0.155, 0.325, r"$\Rightarrow$", fontsize=27, alpha=pb, va="center")
    fig.text(xr + 0.19, 0.325, "the ODE solution", fontsize=BODY_PT, color=TEAL,
             alpha=pb, va="center")
    pb2 = prog(t, 7.4, 8.6)
    fig.text(xr + 0.01, 0.245, "mass  a  at  c", fontsize=BODY_PT, alpha=pb2, va="center")
    fig.text(xr + 0.155, 0.245, r"$\Rightarrow$", fontsize=27, alpha=pb2, va="center")
    fig.text(xr + 0.19, 0.245, "a jump of size a at c", fontsize=BODY_PT, color=CORAL,
             alpha=pb2, va="center")
    pb3 = prog(t, 9.6, 10.8)
    fig.text(xr + 0.01, 0.185, r"here  $\nu=\delta_{1/2}$" + "  gives exactly the observed jump",
             fontsize=SMALL_PT, color=MUTED, alpha=pb3, va="center")


def scene_sciml(fig, t, d):
    header(fig, 6, "Why it matters for scientific machine learning",
           alpha=prog(t, 0, 0.6))
    # ---- network diagram
    pd = prog(t, 0.3, 1.6)
    labels = [r"$z_0$", r"$z_1$", r"$z_2$", r"$\cdots$", r"$z_{n/2}$", r"$\cdots$",
              r"$z_{n-1}$", r"$z_n$"]
    xs = np.linspace(0.10, 0.72, len(labels))
    yc = 0.655
    ry = 0.047
    rx = ry * H / W
    for i, (x, lab) in enumerate(zip(xs, labels)):
        if lab == r"$\cdots$":
            fig.text(x, yc, lab, ha="center", va="center", fontsize=34, color=MUTED, alpha=pd)
            continue
        col = CORAL if i == 4 else BLUE
        fig.add_artist(Ellipse((x, yc), 2 * rx, 2 * ry, transform=fig.transFigure,
                               facecolor=PANEL, edgecolor=col, lw=3, alpha=pd))
        fig.text(x, yc, lab, ha="center", va="center", fontsize=26, alpha=pd)
    gap = 0.006
    for i in range(len(xs) - 1):
        x0 = xs[i] + (rx + gap if labels[i] != r"$\cdots$" else 0.018)
        x1 = xs[i + 1] - (rx + gap if labels[i + 1] != r"$\cdots$" else 0.018)
        fig.add_artist(FancyArrowPatch((x0, yc), (x1, yc), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=22, color=MUTED,
                                       lw=2.2, alpha=pd, shrinkA=0, shrinkB=0))
    # final loss box
    bx = 0.79
    fig.add_artist(FancyBboxPatch((bx, yc - 0.035), 0.06, 0.07,
                                  boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, facecolor=PANEL,
                                  edgecolor=TEXT, lw=2.5, alpha=pd))
    fig.text(bx + 0.03, yc, r"$L$", ha="center", va="center", fontsize=30, alpha=pd)
    fig.add_artist(FancyArrowPatch((xs[-1] + rx + gap, yc), (bx - 0.006, yc),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=22, color=MUTED, lw=2.2, alpha=pd,
                                   shrinkA=0, shrinkB=0))
    fig.text(bx + 0.075, yc, "final loss", fontsize=SMALL_PT, color=MUTED, alpha=pd,
             va="center")
    # auxiliary loss above the middle node
    pa = prog(t, 1.2, 2.4)
    by = 0.755
    fig.add_artist(FancyBboxPatch((xs[4] - 0.042, by), 0.084, 0.055,
                                  boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, facecolor=PANEL,
                                  edgecolor=CORAL, lw=2.5, alpha=pa))
    fig.text(xs[4], by + 0.0275, r"$\ell_{\mathrm{aux}}$", ha="center", va="center",
             fontsize=28, color=CORAL, alpha=pa)
    fig.add_artist(Line2D([xs[4], xs[4]], [yc + ry, by - 0.004], transform=fig.transFigure,
                          color=CORAL, lw=2.5, alpha=pa))
    fig.text(xs[4] + 0.055, by + 0.0275, "auxiliary loss on one middle layer",
             fontsize=SMALL_PT, color=CORAL, alpha=pa, va="center")
    fig.text(0.075, 0.555,
             r"ResNet:  $z_{k+1}=z_k+f(z_k)/n$,   i.e. Euler steps of the neural ODE  $z'=f(z)$",
             fontsize=BODY_PT + 1, color=MUTED, alpha=pd, va="center")
    pbp = prog(t, 2.2, 3.4)
    fig.text(0.075, 0.497,
             r"backprop:  $\lambda_k=\lambda_{k+1}\,(1+f'(z_k)/n)+\partial_{z_k}\ell_{\mathrm{aux}}$"
             + "    is exactly the backward recurrence",
             fontsize=BODY_PT + 1, color=VIOLET, alpha=pbp, va="center")

    # ---- adjoint plot
    pin = prog(t, 2.8, 3.6)
    ax = fig.add_axes([0.075, 0.115, 0.52, 0.30])
    style_ax(ax, alpha=pin)
    ax.set_xlim(0, 1)
    ymax = max(LAM_TRUE.max(), LAM_NAIVE.max()) * 1.12
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "½", "1"])
    ax.set_yticks([0, 2, 4])
    ax.set_xlabel("")
    ax.text(1.0, -0.13, "depth  t = k/n", transform=ax.transAxes, ha="right", va="top",
            fontsize=SMALL_PT, color=MUTED, alpha=pin)
    ax.text(0.0, 1.04, r"adjoint $\lambda(t)=\partial L/\partial z(t)$", transform=ax.transAxes,
            fontsize=BODY_PT + 1, alpha=pin, va="bottom")
    # naive adjoint drawn backward from t=1
    pnv = prog(t, 3.6, 6.0)
    if pnv > 0:
        m = T_FINE >= 1 - pnv
        ax.plot(T_FINE[m], LAM_NAIVE[m], color=AMBER, lw=LW, alpha=pin, zorder=3)
    # discrete adjoint dots, backward
    pdd = prog(t, 6.0, 8.6)
    if pdd > 0:
        m = T_40 >= 1 - pdd - 1e-9
        ax.scatter(T_40[m], LAM_40[m], s=70, color=BLUE, zorder=6, alpha=pin, lw=0)
    ptr = prog(t, 8.6, 9.8)
    if ptr > 0:
        ax.plot(T_FINE, LAM_TRUE, color=TEAL, lw=LW, ls=(0, (5, 3)), alpha=ptr, zorder=5)
        lo = np.interp(0.5001, T_FINE, LAM_TRUE)
        hi = np.interp(0.4999, T_FINE, LAM_TRUE)
        ax.annotate("", xy=(0.5, lo), xytext=(0.5, hi),
                    arrowprops=dict(arrowstyle="<|-|>,head_length=0.6,head_width=0.3",
                                    color=CORAL, lw=3, alpha=ptr, shrinkA=4, shrinkB=4),
                    zorder=8)
        ax.text(0.515, (lo + hi) / 2, "jump", color=CORAL, fontsize=BODY_PT,
                fontweight="semibold", alpha=ptr, va="center")
    # legend (manual)
    xl = 0.635
    items = [(AMBER, "naive neural-ODE adjoint (no jump)", pnv),
             (BLUE, "backprop through the ResNet, n = 40", pdd),
             (TEAL, "measure-aware adjoint (with jump)", ptr)]
    for i, (c, s, a_) in enumerate(items):
        yy = 0.385 - i * 0.065
        fig.add_artist(Line2D([xl, xl + 0.03], [yy, yy], transform=fig.transFigure,
                              color=c, lw=LW, alpha=a_ * pin))
        fig.text(xl + 0.04, yy, s, fontsize=SMALL_PT, va="center", alpha=a_ * pin)
    pm = prog(t, 9.8, 10.8)
    fig.text(xl, 0.145, "Discretize-then-optimize and\noptimize-then-discretize disagree\nunless the jump is kept.",
             fontsize=BODY_PT, color=CORAL, alpha=pm, va="center", linespacing=1.3,
             fontweight="semibold")


def scene_end(fig, t, d):
    p1 = prog(t, 0.2, 1.2)
    fig.text(0.5, 0.62, "Adib Alqurashi", ha="center", va="center", fontsize=52,
             family=["Inter Display", "Inter"], fontweight="bold", alpha=p1)
    fig.text(0.5, 0.535, "King Fahd University of Petroleum and Minerals (KFUPM)",
             ha="center", va="center", fontsize=26, color=MUTED, alpha=p1)
    p2 = prog(t, 0.7, 1.7)
    fig.text(0.5, 0.465, "Principia Fall Competition 2026  ·  MATHDB 354674",
             ha="center", va="center", fontsize=26, color=TEAL, alpha=p2,
             fontweight="semibold")
    L = 0.22 * prog(t, 0.9, 1.9)
    fig.add_artist(Line2D([0.5 - L / 2, 0.5 + L / 2], [0.41, 0.41], transform=fig.transFigure,
                          color=CORAL, lw=4, solid_capstyle="round", alpha=p2))
    p3 = prog(t, 1.4, 2.4)
    credits = ["Conjecture: Bayón, Fortuny Ayuso, Grau, Oller-Marcén & Ruiz (2023)",
               "Counterexample: Principia Math in Progress (2026)",
               "Lean formalization: PrincipiaMathOrg/math-solutions"]
    for i, c in enumerate(credits):
        fig.text(0.5, 0.33 - i * 0.05, c, ha="center", va="center", fontsize=SMALL_PT,
                 color=MUTED, alpha=p3)


SCENE_FUNCS = {
    "title": scene_title,
    "recurrence": scene_recurrence,
    "coefficients": scene_coefficients,
    "convergence": scene_convergence,
    "fixed_moving": scene_fixed_moving,
    "measure": scene_measure,
    "sciml": scene_sciml,
    "end": scene_end,
}

# --------------------------------------------------------------------------- frame
_FIG = None


def get_fig():
    global _FIG
    if _FIG is None:
        _FIG = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    return _FIG


def draw_at(t, fades=True):
    fig = get_fig()
    fig.clf()
    fig.set_facecolor(BG)
    idx = int(np.searchsorted(STARTS, t, side="right") - 1)
    idx = max(0, min(idx, len(SCENES) - 1))
    name, d = SCENES[idx]
    lt = t - STARTS[idx]
    SCENE_FUNCS[name](fig, lt, d)
    if 0 < idx < len(SCENES) - 1:
        progress_bar(fig, t)
    if fades:
        a_in = 1 - clamp(lt / FADE)
        a_out = 1 - clamp((d - lt) / FADE)
        fade_overlay(fig, max(a_in, a_out))
    return fig


def render_rgb(i):
    t = i / FPS
    fig = draw_at(min(t, TOTAL - 1e-6))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].tobytes()


def make_video(out, workers=2):
    nframes = int(round(TOTAL * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(workers) as pool:
        for i, b in enumerate(pool.imap(render_rgb, range(nframes), chunksize=8)):
            proc.stdin.write(b)
            if i % 300 == 0:
                print(f"frame {i}/{nframes}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("done", out, flush=True)


STILLS = [
    ("01_title.png", 0 * 0 + 4.6, "title"),
    ("02_recurrence.png", STARTS[1] + 12.0, "recurrence"),
    ("03_coefficients.png", STARTS[2] + 13.5, "coefficients"),
    ("04_convergence_jump.png", STARTS[3] + 20.5, "convergence"),
    ("05_fixed_vs_moving.png", STARTS[4] + 11.5, "fixed vs moving"),
    ("06_measure_fix.png", STARTS[5] + 11.5, "measure view"),
    ("07_sciml_adjoint.png", STARTS[6] + 11.0, "sciml"),
    ("08_end.png", STARTS[7] + 4.0, "end"),
]


def make_stills(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, t, _ in STILLS:
        fig = draw_at(t, fades=False)
        fig.savefig(os.path.join(outdir, name), dpi=DPI, facecolor=BG)
        print("saved", name, flush=True)


if __name__ == "__main__":
    # sanity: closed form equals the recurrence
    for nn in (4, 5, 12, 101, 2000):
        assert np.allclose(F_main(nn), F_main_recurrence(nn)), nn
    mode = sys.argv[1]
    if mode == "video":
        w = 2
        if "--workers" in sys.argv:
            w = int(sys.argv[sys.argv.index("--workers") + 1])
        make_video(sys.argv[2], workers=w)
    elif mode == "stills":
        make_stills(sys.argv[2])
    elif mode == "frame":
        fig = draw_at(float(sys.argv[2]), fades=False)
        fig.savefig(sys.argv[3], dpi=DPI, facecolor=BG)
    print("TOTAL", TOTAL)

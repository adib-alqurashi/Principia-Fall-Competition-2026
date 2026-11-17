"""Write paper/numbers.tex: every number quoted in the paper, straight from results.json."""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(here, "..", "results.json")))
out = []


def m(name, val):
    out.append(f"\\newcommand{{\\{name}}}{{{val}}}")


m("NexactMax", R["e1"]["closed_form_verified_exactly_for_n"][1])
m("SlopeCE", f"{R['e2']['fitted_slope']:.2f}")
m("SlopeGen", f"{R['e3']['fitted_slope']:.2f}")
m("GapGen", f"{R['e3']['max_gap_limit_vs_naive_ode']:.2f}")
m("NmaxCE", R["e2"]["ns"][-1])
m("ErrCEmax", f"{R['e2']['sup_err_away_from_half'][-1]:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
m("FquarterLast", f"{R['e2']['f_n_quarter'][-1]:.6f}")
m("SmoothErr", f"{R['e4']['discrete_check_w0.1_n4096_sup_err']:.1e}".replace("e-0", "\\times10^{-") + "}")
e6 = R["e6"]
m("GradJump", f"{e6['grad_continuous_with_jump']:.6f}")
m("GradNaive", f"{e6['grad_continuous_naive']:.6f}")
m("GradRelErr", f"{100 * (e6['grad_continuous_with_jump'] - e6['grad_continuous_naive']) / e6['grad_continuous_with_jump']:.1f}")
m("FDdiff", f"{e6['fd_check_n64']['abs_diff']:.1e}".replace("e-", "\\times10^{-") + "}")
m("SlopeAdj", f"{e6['fitted_slope_jump']:.2f}")

rows = []
for n, g, ej, en in zip(e6["ns"], e6["grad_discrete"], e6["err_vs_jump"], e6["err_vs_naive"]):
    if n in (8, 32, 128, 512, 2048, 8192):
        mant, ex = f"{ej:.2e}".split("e")
        rows.append(f"{n} & {g:.6f} & ${mant}\\times10^{{{int(ex)}}}$ & {en:.4f} \\\\")
out.append("\\newcommand{\\AdjointRows}{" + "\n".join(rows) + "}")

e5 = R["e5"]
m("FixedLast", f"{e5['fixed_index_3'][-1]:.4f}")
m("HalfLast", f"{e5['moving_half'][-1]:.4f}")
m("ThirdLast", f"{e5['moving_third'][-1]:.4f}")
m("NmaxFive", e5["ns"][-1])

with open(os.path.join(here, "..", "paper", "numbers.tex"), "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))

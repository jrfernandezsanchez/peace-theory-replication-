# The Endogenous Peace Frontier.
# Theorem 1(a) eventual feasibility, (b) the explicit ripeness bound, (c) single
# infeasible interval under the concavity conditions, (e) comparative statics of the
# return crossing; Corollaries 1.1 and 1.2; Proposition 5 (grudge threshold) and
# Theorem A1(c) (silence threshold) via the belief operator of Appendix A.
import numpy as np, sys
from scipy.optimize import brentq

rng = np.random.default_rng(3); fails = []
T = 4000                                   # discrete horizon, periods


def path(par):
    """Discrete-time spite path for one player. Grievance stock depreciates,
    exhaustion does not. Returns lambda_i(t) truncated at zero, and the ceiling."""
    lam_th, rho, lam_H, alpha, delta, beta, ebar, gmax = par
    G = 0.0; lam = np.empty(T); mu = 0.0
    for t in range(T):
        mu = 1 - (1 - mu) * 0.97                       # concave belief path -> 1
        Lam_f = mu * lam_H + (1 - mu) * lam_th
        lam[t] = max(0.0, lam_th + rho * Lam_f + alpha * G - beta * ebar * t)
        G = np.exp(-delta) * G + gmax                  # flow at its ceiling
    Hbar = lam_th + rho * lam_H + alpha * gmax / (1 - np.exp(-delta))
    return lam, Hbar


# ------------------------------------------------- A. the grievance stock ceiling
# G_i(t) <= gmax/(1-exp(-delta)); the continuous-time bound gmax/delta understates it.
worst = -np.inf; understates = 0
for _ in range(500):
    delta = rng.uniform(0.02, 1.5); gmax = rng.uniform(0.05, 2.0)
    G = 0.0
    for t in range(3000):
        G = np.exp(-delta) * G + rng.uniform(0, gmax)  # any admissible flow
    ceil_d = gmax / (1 - np.exp(-delta))
    worst = max(worst, G - ceil_d)
    understates += int(gmax / delta < ceil_d)
print(f"A. grievance ceiling gmax/(1-e^-delta) never breached | worst excess "
      f"{worst:.2e}; continuous-time bound understates in {understates}/500 draws")
fails += [0 if (worst < 1e-9 and understates == 500) else 1]

# ------------------------------------------- B. Theorem 1(a) and (b): the bound
# The infeasible set is bounded, its maximum t_2 is finite, and t_2 <= tbar.
bad_bound = 0; bad_finite = 0; tested = 0
for _ in range(400):
    common = dict()
    pars = []
    for _i in range(2):
        pars.append((rng.uniform(0.05, 0.5),    # lam_theta
                     rng.uniform(0.1, 0.9),     # rho
                     rng.uniform(0.8, 2.5),     # lam_H
                     rng.uniform(0.005, 0.05),  # alpha
                     rng.uniform(0.05, 0.8),    # delta
                     rng.uniform(1e-4, 3e-3),   # beta
                     rng.uniform(0.5, 2.0),     # ebar
                     rng.uniform(0.05, 1.0))    # gmax
                    )
    (lamR, HR), (lamU, HU) = path(pars[0]), path(pars[1])
    Pi = lamR * lamU
    infeas = np.where(Pi > 1.0)[0]
    if len(infeas) == 0:
        continue
    tested += 1
    t2 = infeas.max()
    if t2 >= T - 1:                            # not bounded inside the horizon
        bad_finite += 1; continue
    tb = []
    for (i, j), (Hi, Hj) in (((0, 1), (HR, HU)), ((1, 0), (HU, HR))):
        num = Hi - 1.0 / Hj
        tb.append(max(0.0, num) / (pars[i][5] * pars[i][6]))
    tbar = min(tb)
    if t2 > tbar + 1:                          # +1 period of discretisation slack
        bad_bound += 1
print(f"B. Theorem 1(a)(b): t_2 finite and t_2 <= tbar | {tested} infeasible draws, "
      f"unbounded {bad_finite}, bound violations {bad_bound}")
fails += [bad_finite + bad_bound]

# --------------------------------- C. Theorem 1(c): a single infeasible interval
# Under concave belief paths, flows kept at the ceiling and non-decreasing losses,
# the infeasible set is contiguous.
gaps = 0; tested = 0
for _ in range(400):
    pars = [(rng.uniform(0.05, 0.5), rng.uniform(0.1, 0.9), rng.uniform(0.8, 2.5),
             rng.uniform(0.005, 0.05), rng.uniform(0.05, 0.8), rng.uniform(1e-4, 3e-3),
             rng.uniform(0.5, 2.0), rng.uniform(0.05, 1.0)) for _i in range(2)]
    Pi = path(pars[0])[0] * path(pars[1])[0]
    idx = np.where(Pi > 1.0)[0]
    if len(idx) == 0:
        continue
    tested += 1
    if idx.max() - idx.min() + 1 != len(idx):
        gaps += 1
print(f"C. Theorem 1(c): infeasible set is one interval | {tested} draws, "
      f"non-contiguous {gaps}")
fails += [gaps]

# ------------------------- D. Theorem 1(e): comparative statics of the crossing
# Continuous-time reading: Pi(t_2) = 1. Signs: d t_2/d beta < 0, d t_2/d delta < 0,
# d t_2/d rho > 0.
def t2_cont(beta, delta, rho, lam_th=0.30, lam_H=1.8, alpha=0.02, gmax=0.5,
            ebar=1.0, kappa=0.97):
    def lam(t):
        mu = 1 - (1 - 0.0) * kappa ** (t + 1)
        Lam_f = mu * lam_H + (1 - mu) * lam_th
        G = gmax / (1 - np.exp(-delta)) * (1 - np.exp(-delta * (t + 1)))
        return max(0.0, lam_th + rho * Lam_f + alpha * G - beta * ebar * t)
    f = lambda t: lam(t) * lam(t) - 1.0
    grid = np.arange(0.0, 6000.0, 1.0)         # locate the last crossing
    pos = np.where(np.array([f(t) for t in grid]) > 0)[0]
    if len(pos) == 0 or pos.max() == len(grid) - 1:
        return None
    i = pos.max()
    return brentq(f, grid[i], grid[i + 1], xtol=1e-8)

base = dict(beta=1.2e-3, delta=0.3, rho=0.6)
t0 = t2_cont(**base)
signs = {}
for k, step in (("beta", 1e-4), ("delta", 0.02), ("rho", 0.02)):
    up = dict(base); up[k] = base[k] + step
    t1 = t2_cont(**up)
    signs[k] = np.sign(t1 - t0)
ok_e = (signs["beta"] < 0) and (signs["delta"] < 0) and (signs["rho"] > 0)
print(f"D. Theorem 1(e): t_2={t0:.1f}; d/dbeta {signs['beta']:+.0f}, "
      f"d/ddelta {signs['delta']:+.0f}, d/drho {signs['rho']:+.0f} | as stated {ok_e}")
fails += [0 if ok_e else 1]

# ------------------------------- E. Corollaries 1.1 and 1.2: cheap wars, armistice
tb = lambda beta, ebar, Hi=2.0, Hj=2.0: (Hi - 1 / Hj) / (beta * ebar)
mono = all(tb(1e-3, e) > tb(1e-3, e + 0.1) for e in np.arange(0.1, 3.0, 0.1))
armistice = np.isinf(tb(1e-3, 0.0)) if False else (tb(1e-3, 1e-12) > 1e12)
print(f"E. Corollary 1.1: tbar strictly decreasing in beta*ebar {mono}; "
      f"Corollary 1.2: ebar -> 0 sends the bound to infinity {armistice}")
fails += [0 if (mono and armistice) else 1]

# --------------------------------------------- F. Proposition 5: grudge threshold
# lam_i = lam_D + rho*(mu_j*lam_H + (1-mu_j)*lam_D). The trap region {Pi^DD > 1}
# is non-empty iff lam_D + rho*lam_H > 1; it is upward-closed and contains a
# neighbourhood of (1,1).
mu = np.linspace(0, 1, 401)
MR, MU = np.meshgrid(mu, mu, indexing="ij")
bad_exist = 0; bad_mono = 0; bad_nbhd = 0
for _ in range(600):
    lam_D = rng.uniform(0.05, 0.9)
    lam_H = lam_D + rng.uniform(0.2, 2.2)      # the hawk is the more spiteful type
    rho = rng.uniform(0, 1)
    thr = lam_D + rho * lam_H
    if abs(thr - 1.0) < 1e-3:
        continue
    lamR = lam_D + rho * (MU * lam_H + (1 - MU) * lam_D)   # R's spite depends on mu_U
    lamU = lam_D + rho * (MR * lam_H + (1 - MR) * lam_D)
    trap = (lamR * lamU) > 1.0
    if trap.any() != (thr > 1.0):
        bad_exist += 1
    if trap.any():
        # upward-closed: monotone in each belief coordinate
        if not (np.all(trap[:-1, :] <= trap[1:, :]) and np.all(trap[:, :-1] <= trap[:, 1:])):
            bad_mono += 1
        if not trap[-3:, -3:].all():                        # neighbourhood of (1,1)
            bad_nbhd += 1
print(f"F. Proposition 5: non-empty iff lam_D+rho*lam_H>1 | mismatches {bad_exist}; "
      f"not upward-closed {bad_mono}; missing corner {bad_nbhd}")
fails += [bad_exist + bad_mono + bad_nbhd]

# ------------------------ G. Theorem A1(c): the trap needs grudge AND silence
# Belief operator Phi_j(mu) = mu_j if lam_D + rho*Lam_f(mu_i) >= lam_1, else 0.
# A trap is a fixed point with both coordinates positive and Pi^DD > 1.
def trap_exists(lam_D, lam_H, rho, lam_1, n=201):
    m = np.linspace(0, 1, n)
    A, B = np.meshgrid(m, m, indexing="ij")            # A = mu_R, B = mu_U
    Lam_R = B * lam_H + (1 - B) * lam_D                # what R infers about U
    Lam_U = A * lam_H + (1 - A) * lam_D
    freeze_R = (lam_D + rho * Lam_U) >= lam_1          # R's conduct stops separating
    freeze_U = (lam_D + rho * Lam_R) >= lam_1
    fixed = ((A == 0) | freeze_R) & ((B == 0) | freeze_U)
    lamR = lam_D + rho * Lam_R; lamU = lam_D + rho * Lam_U
    return bool((fixed & (A > 0) & (B > 0) & (lamR * lamU > 1)).any())

bad = 0; tested = 0
for _ in range(800):
    lam_D = rng.uniform(0.05, 0.9)
    lam_H = lam_D + rng.uniform(0.2, 2.2)
    rho = rng.uniform(0.05, 1.0); lam_1 = rng.uniform(0.3, 3.0)
    thr = lam_D + rho * lam_H
    if abs(thr - 1.0) < 1e-2 or abs(thr - lam_1) < 1e-2:
        continue
    tested += 1
    if trap_exists(lam_D, lam_H, rho, lam_1) != (thr >= max(lam_1, 1.0)):
        bad += 1
print(f"G. Theorem A1(c): trap exists iff lam_D+rho*lam_H >= max(lam_1,1) | "
      f"{tested} draws, mismatches {bad}")
fails += [bad]

# --------------- H. comparative statics of the freeze region (Appendix A remark)
# The trap is non-decreasing in rho and lam_H, non-increasing in lam_1.
viol = 0
for _ in range(300):
    lam_D = rng.uniform(0.05, 0.6)
    lam_H = lam_D + rng.uniform(0.3, 2.0)
    rho = rng.uniform(0.1, 0.8); lam_1 = rng.uniform(0.5, 2.0)
    base_t = trap_exists(lam_D, lam_H, rho, lam_1)
    if base_t:
        if not trap_exists(lam_D, lam_H, min(1.0, rho + 0.05), lam_1): viol += 1
        if not trap_exists(lam_D, lam_H + 0.2, rho, lam_1): viol += 1
        if not trap_exists(lam_D, lam_H, rho, max(0.01, lam_1 - 0.2)): viol += 1
print(f"H. trap monotone: up in rho and lam_H, down in lam_1 | violations {viol}")
fails += [viol]

print("RESULT:", "all claims verified" if not any(fails) else "FAILURES PRESENT")
sys.exit(1 if any(fails) else 0)

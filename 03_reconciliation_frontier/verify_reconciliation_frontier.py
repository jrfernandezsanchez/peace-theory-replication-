# The Reconciliation Frontier.
# Lemma 1 against a numerical integration oracle; Theorem 1 (critical forgetting
# rate, regimes, the date T, comparative statics, and the ratio result); Propositions
# 2, 3, 4, 6 (the closing window) and Proposition 1 (symmetric vs targeted
# enforcement, which behave oppositely on the two sides of the frontier).
import numpy as np, sys
from scipy.optimize import brentq

rng = np.random.default_rng(4); fails = []


# ------------------------------------------------------------------- A. Lemma 1
# G(t) = R + (G0 - R)e^{-delta t} against Euler integration of Gdot = gbar - delta*G.
worst = 0.0
for _ in range(400):
    gbar = rng.uniform(0.0, 3.0); delta = rng.uniform(0.02, 1.2); G0 = rng.uniform(0, 60)
    R = gbar / delta
    Tend = rng.uniform(1.0, 40.0); n = 400000; h = Tend / n
    G = G0
    for _k in range(n):
        G += h * (gbar - delta * G)
    worst = max(worst, abs(G - (R + (G0 - R) * np.exp(-delta * Tend))))
print(f"A. Lemma 1: closed form vs Euler integration | max error {worst:.2e}")
fails += [0 if worst < 1e-4 else 1]

# --------------------------------------------------- B. Theorem 1(a): delta* root
# delta* = alpha*gbar/(1-kappa) is the unique root of Pi_inf(delta) = 1, found here
# by bisection on the raw definition rather than by the formula.
worst = 0.0; nonmono = 0
for _ in range(600):
    kappa = rng.uniform(0.05, 0.95); alpha = rng.uniform(0.002, 0.06)
    gbar = rng.uniform(0.05, 3.0)
    Pi = lambda d: (kappa + alpha * gbar / d) ** 2
    f = lambda d: Pi(d) - 1.0
    lo, hi = 1e-9, 1.0
    while f(hi) > 0:
        hi *= 2
    root = brentq(f, lo, hi, xtol=1e-14, rtol=1e-15)
    worst = max(worst, abs(root - alpha * gbar / (1 - kappa)))
    ds = np.linspace(root / 4, root * 4, 500)
    if not np.all(np.diff([Pi(d) for d in ds]) < 0):
        nonmono += 1
print(f"B. Theorem 1(a): delta*=alpha*gbar/(1-kappa) vs bisection | max error "
      f"{worst:.2e}; Pi_inf strictly decreasing in delta | violations {nonmono}")
fails += [0 if (worst < 1e-9 and nonmono == 0) else 1]

# ------------------------------- C. Theorem 1(b) and (c): regimes and the date T
# Simulate lambda(t) = kappa + alpha*G(t) and read off the first date at which the
# product falls below one; compare with T = (1/delta) ln[(G0-R)/(G*-R)].
worst = 0.0; wrong_regime = 0
for _ in range(600):
    kappa = rng.uniform(0.05, 0.9); alpha = rng.uniform(0.005, 0.05)
    gbar = rng.uniform(0.05, 2.0)
    dstar = alpha * gbar / (1 - kappa)
    delta = dstar * rng.choice([rng.uniform(0.2, 0.9), rng.uniform(1.1, 6.0)])
    R = gbar / delta; Gstar = (1 - kappa) / alpha
    G0 = Gstar * rng.uniform(1.05, 4.0)                     # infeasible at armistice
    reconciles = delta > dstar
    if reconciles != (R < Gstar):                            # Theorem 1(e), part 1
        wrong_regime += 1
    ts = np.linspace(0, 4000, 400001)
    G = R + (G0 - R) * np.exp(-delta * ts)
    lam = kappa + alpha * G
    below = np.where(lam * lam < 1.0)[0]
    if reconciles:
        if len(below) == 0:
            wrong_regime += 1; continue
        T_sim = ts[below.min()]
        T = (1 / delta) * np.log((G0 - R) / (Gstar - R))
        if T < 4000:
            worst = max(worst, abs(T_sim - T))
    else:
        if len(below) > 0:
            wrong_regime += 1
print(f"C. Theorem 1(b)(c): regimes correct and T vs simulated first passage | "
      f"max error {worst:.2e}, regime errors {wrong_regime}")
fails += [0 if (worst < 2e-2 and wrong_regime == 0) else 1]

# ---------------------------------------------------- D. Theorem 1(d): signs of T
def Tdate(kappa, alpha, gbar, delta, G0):
    R = gbar / delta; Gstar = (1 - kappa) / alpha
    return np.inf if Gstar <= R else (1 / delta) * np.log((G0 - R) / (Gstar - R))

viol = 0
for _ in range(600):
    kappa = rng.uniform(0.05, 0.8); alpha = rng.uniform(0.005, 0.05)
    gbar = rng.uniform(0.05, 1.5); dstar = alpha * gbar / (1 - kappa)
    delta = dstar * rng.uniform(1.3, 6.0)
    G0 = (1 - kappa) / alpha * rng.uniform(1.2, 4.0)
    T0 = Tdate(kappa, alpha, gbar, delta, G0)
    if not np.isfinite(T0):
        continue
    if not Tdate(kappa, alpha, gbar, delta * 1.02, G0) < T0: viol += 1   # dT/ddelta<0
    if not Tdate(kappa, alpha, gbar * 1.02, delta, G0) > T0: viol += 1   # dT/dgbar>0
    if not Tdate(kappa, alpha, gbar, delta, G0 * 1.02) > T0: viol += 1   # dT/dG0>0
    if not Tdate(kappa * 0.98, alpha, gbar, delta, G0) < T0: viol += 1   # beta*E lowers kappa
print(f"D. Theorem 1(d): T down in delta, up in gbar and G0, down in beta*Ebar | "
      f"violations {viol}")
fails += [viol]

# -------------------------- E. Theorem 1(e): it is the ratio R = gbar/delta, not the rate
worst = 0.0
for _ in range(600):
    kappa = rng.uniform(0.05, 0.9); alpha = rng.uniform(0.005, 0.05)
    R = rng.uniform(0.5, 60.0)
    vals = []
    for _k in range(5):
        delta = rng.uniform(0.05, 2.0); gbar = R * delta
        vals.append((kappa + alpha * gbar / delta) ** 2)
    worst = max(worst, max(vals) - min(vals))
print(f"E. Theorem 1(e): Pi_inf invariant across (gbar,delta) with equal ratio | "
      f"max spread {worst:.2e}")
fails += [0 if worst < 1e-12 else 1]

# ------------------------------------------------------------- F. Proposition 2
# Pi_0 = kappa_R*kappa_U >= 1 implies Pi_inf >= 1 for every memory regime.
viol = 0
for _ in range(2000):
    kR = rng.uniform(0.5, 3.0); kU = rng.uniform(1.0 / kR, 3.0)   # Pi_0 >= 1
    alpha = rng.uniform(0.001, 0.1); RR = rng.uniform(0, 80); RU = rng.uniform(0, 80)
    if (kR + alpha * RR) * (kU + alpha * RU) < 1.0:
        viol += 1
print(f"F. Proposition 2: Pi_0>=1 implies Pi_inf>=1 for every (gbar,delta) | "
      f"violations {viol}")
fails += [viol]

# ------------------------------------------------- G. Propositions 3 and 4
# Along the frontier dR_U/dR_R = -lam_U/lam_R; and dPi_inf/dR_i = alpha*lam_j.
worst3 = 0.0; worst4 = 0.0; h = 1e-6
for _ in range(1000):
    kR = rng.uniform(0.05, 0.9); kU = rng.uniform(0.05, 0.9); alpha = rng.uniform(0.005, 0.05)
    RR = rng.uniform(1.0, 40.0)
    RU = (1.0 / (kR + alpha * RR) - kU) / alpha                  # on the frontier
    if RU <= 0:
        continue
    lamR = kR + alpha * RR; lamU = kU + alpha * RU
    RU_p = (1.0 / (kR + alpha * (RR + h)) - kU) / alpha
    worst3 = max(worst3, abs((RU_p - RU) / h - (-lamU / lamR)))
    Pi = lambda a, b: (kR + alpha * a) * (kU + alpha * b)
    worst4 = max(worst4, abs((Pi(RR + h, RU) - Pi(RR, RU)) / h - alpha * lamU),
                 abs((Pi(RR, RU + h) - Pi(RR, RU)) / h - alpha * lamR))
print(f"G. Proposition 3: dR_U/dR_R = -lam_U/lam_R max error {worst3:.2e}; "
      f"Proposition 4: dPi/dR_i = alpha*lam_j max error {worst4:.2e}")
fails += [0 if (worst3 < 1e-4 and worst4 < 1e-6) else 1]

# ------------------------------------------------- H. Proposition 6: closing window
# kappa(t) = kappa_inf - beta*Ebar*e^{-eps t} rises; delta*(t) rises; if kappa_inf>1
# there is a finite tbar with kappa(tbar)=1; and for delta between delta*(0) and
# delta*_inf feasibility is attained and then lost at a finite date.
bad_mono = 0; bad_tbar = 0; bad_trans = 0; n_trans = 0
for _ in range(600):
    alpha = rng.uniform(0.005, 0.05); gbar = rng.uniform(0.05, 1.5)
    eps = rng.uniform(0.02, 0.4); bE = rng.uniform(0.1, 0.8)
    k_inf = rng.uniform(0.2, 1.6)
    kap = lambda t: k_inf - bE * np.exp(-eps * t)
    if kap(0.0) <= 0 or kap(0.0) >= 1:
        continue
    ts = np.linspace(0, 400, 40001)
    kv = kap(ts)
    live = kv < 1 - 1e-12
    dstar_t = np.where(live, alpha * gbar / np.maximum(1 - kv, 1e-15), np.inf)
    if not np.all(np.diff(dstar_t[live]) >= -1e-12):
        bad_mono += 1
    if k_inf > 1:
        tb = brentq(lambda t: kap(t) - 1.0, 0.0, 1e4)
        if not (np.isfinite(tb) and abs(kap(tb) - 1) < 1e-9):
            bad_tbar += 1
        if (kap(tb + 1.0) <= 1.0):
            bad_tbar += 1
    else:
        d0 = alpha * gbar / (1 - kap(0.0)); dinf = alpha * gbar / (1 - k_inf)
        if dinf <= d0:
            continue
        delta = rng.uniform(d0 * 1.01, dinf * 0.99)             # transient regime
        n_trans += 1
        R = gbar / delta; G0 = R                                # stock at its limit
        lam = kap(ts) + alpha * (R + (G0 - R) * np.exp(-delta * ts))
        feas = lam * lam < 1.0
        # feasible early, infeasible from some finite date onward
        if not (feas[0] and (not feas[-1])):
            bad_trans += 1
print(f"H. Proposition 6: delta*(t) increasing | violations {bad_mono}; finite tbar "
      f"when kappa_inf>1 | failures {bad_tbar}; transient feasibility lost at a finite "
      f"date | {n_trans} draws, failures {bad_trans}")
fails += [bad_mono + bad_tbar + bad_trans]

# ------------------------------------------------------------- I. Proposition 1
# (a) symmetric enforcement: dW/ds = 2(1-lR lU)/((1+lR)(1+lU)), sign flips at the
# frontier. (b) targeted enforcement: a finite s sustains peace at any product.
worst = 0.0; bad_sign = 0; h = 1e-6
for _ in range(1500):
    p = rng.uniform(0.2, 0.8); cR = rng.uniform(0.05, 0.25); cU = rng.uniform(0.05, 0.25)
    lR = rng.uniform(0, 3.0); lU = rng.uniform(0, 3.0)
    if abs(lR * lU - 1) < 1e-3:
        continue
    def W(s):
        aR = lR / (1 + lR); aU = lU / (1 + lU)
        return (cR + s + cU + s) * (1 - aR - aU)
    d = (W(h) - W(0)) / h
    pred = 2 * (1 - lR * lU) / ((1 + lR) * (1 + lU))
    worst = max(worst, abs(d - pred))
    if np.sign(d) != np.sign(1 - lR * lU):
        bad_sign += 1
# (b) targeted: R accepts x rather than revise iff x - lR(1-x) >= (uR - s) - lR*uU
grid = np.linspace(0, 1, 100001); bad_targeted = 0
for _ in range(400):
    p = rng.uniform(0.2, 0.8); cR = rng.uniform(0.05, 0.25); cU = rng.uniform(0.05, 0.25)
    uR = p - cR; uU = (1 - p) - cU
    lR = rng.uniform(1.2, 4.0); lU = rng.uniform(1.2, 4.0)     # product well above 1
    s = 50.0
    okR = (grid - lR * (1 - grid)) >= (uR - s) - lR * uU
    okU = ((1 - grid) - lU * grid) >= (uU - s) - lU * uR
    if not (okR & okU).any():
        bad_targeted += 1
print(f"I. Proposition 1(a): dW/ds formula max error {worst:.2e}, sign errors "
      f"{bad_sign}; (b) targeted enforcement sustains peace at Pi>1 | failures "
      f"{bad_targeted}")
fails += [0 if (worst < 1e-4 and bad_sign == 0 and bad_targeted == 0) else 1]

print("RESULT:", "all claims verified" if not any(fails) else "FAILURES PRESENT")
sys.exit(1 if any(fails) else 0)

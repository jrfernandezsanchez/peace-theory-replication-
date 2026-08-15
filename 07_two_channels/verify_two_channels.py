# The Two Channels: Transitional Justice, Memory Policy and the Reconciliation
# Frontier.
# A. Proposition 1: the total effect of a policy on the long-run product runs
#    through the belief channel and the flow channel and through no other
#    (analytic decomposition vs numerical differentiation).
# B. Proposition 2: the memory-free floor bounds the product from below; a
#    floor at or above one defeats every flow instrument.
# C. The symmetric critical forgetting rate delta* = alpha*gbar/(1 - kappa_inf).
# D. Proposition 7 of Paper III as used here: the ceiling G*(t) descends from
#    (1 - kappa_inf + beta*E)/alpha to (1 - kappa_inf)/alpha.
# E. The law of motion: closed form, convergence, steady state gbar/delta.
import numpy as np, sys

rng = np.random.default_rng(7)
fails = []

def lam_inf(lam_th, rho, Lbar, alpha, gbar, delta):
    return lam_th + rho * Lbar + alpha * gbar / delta

# ------------------------------------------------ A. the two-channel formula
badA = 0
for _ in range(500):
    th = rng.uniform(0.05, 0.5, 2); rho = rng.uniform(0.1, 0.8)
    Lb = rng.uniform(0.1, 0.9, 2); al = rng.uniform(0.1, 0.9)
    gb = rng.uniform(0.05, 0.5, 2); de = rng.uniform(0.1, 0.9, 2)
    mL = rng.uniform(-1.0, 1.0, 2); mg = rng.uniform(-1.0, 1.0, 2)  # loadings

    def Pi(p):
        lR = lam_inf(th[0], rho, Lb[1] + p * mL[1], al, gb[0] + p * mg[0], de[0])
        lU = lam_inf(th[1], rho, Lb[0] + p * mL[0], al, gb[1] + p * mg[1], de[1])
        return lR * lU

    lR0 = lam_inf(th[0], rho, Lb[1], al, gb[0], de[0])
    lU0 = lam_inf(th[1], rho, Lb[0], al, gb[1], de[1])
    analytic = (lU0 * (rho * mL[1] + (al / de[0]) * mg[0])
                + lR0 * (rho * mL[0] + (al / de[1]) * mg[1]))
    eps = 1e-6
    numeric = (Pi(eps) - Pi(-eps)) / (2 * eps)
    if abs(numeric - analytic) > 1e-6 * max(1.0, abs(analytic)):
        badA += 1
print("A. Proposition 1: analytic dPi/dp vs numerical derivative, 500 draws | mismatches %d" % badA)
fails.append(badA)

# ---------------------------------------------------------- B. the floor
badB = 0
for _ in range(500):
    kR = rng.uniform(0.05, 1.4); kU = rng.uniform(0.05, 1.4)
    al = rng.uniform(0.1, 0.9)
    GR = rng.uniform(0.0, 3.0); GU = rng.uniform(0.0, 3.0)
    Pi = (kR + al * GR) * (kU + al * GU)
    if Pi < kR * kU - 1e-12:
        badB += 1
# floor >= 1 defeats every flow setting
kR, kU = 1.25, 0.85            # product 1.0625 >= 1
al = 0.5
worst = min((kR + al * g1) * (kU + al * g2)
            for g1 in np.linspace(0, 2, 41) for g2 in np.linspace(0, 2, 41))
if worst < kR * kU - 1e-12 or worst < 1.0:
    badB += 1
print("B. Proposition 2: Pi >= floor always; floor >= 1 kills every flow instrument | violations %d" % badB)
fails.append(badB)

# ----------------------------------------------- C. critical forgetting rate
badC = 0
for _ in range(200):
    k = rng.uniform(0.1, 0.9); al = rng.uniform(0.1, 0.9); gb = rng.uniform(0.05, 0.5)
    ds = al * gb / (1.0 - k)
    hi = (k + al * gb / (ds * 1.10)) ** 2
    lo = (k + al * gb / (ds * 0.90)) ** 2
    if not (hi < 1.0 < lo):
        badC += 1
print("C. delta* = alpha*gbar/(1-kappa): reconciliation iff delta above it, 200 draws | violations %d" % badC)
fails.append(badC)

# ------------------------------------------------------- D. the ceiling
k_inf = 0.6; beta_E = 0.25; al = 0.5; epsr = 0.15
t = np.linspace(0.0, 60.0, 601)
kap = k_inf - beta_E * np.exp(-epsr * t)
G = (1.0 - kap) / al
okD = np.all(np.diff(G) < 0) and abs(G[0] - (1 - k_inf + beta_E) / al) < 1e-9 \
      and abs(G[-1] - (1 - k_inf) / al) < 1e-3
print("D. ceiling G*(t) strictly descends between its stated endpoints | %s" % ("ok" if okD else "FAIL"))
fails.append(0 if okD else 1)

# ------------------------------------------------------ E. law of motion
gb, de, G0 = 0.3, 0.2, 2.0
dt = 1e-3
G = G0
for _ in range(int(50 / dt)):
    G += dt * (gb - de * G)
closed = gb / de + (G0 - gb / de) * np.exp(-de * 50)
okE = abs(G - closed) < 1e-3 and abs(G - gb / de) < 1e-3
print("E. Gdot = gbar - delta*G: simulation matches closed form; steady state gbar/delta | %s"
      % ("ok" if okE else "FAIL"))
fails.append(0 if okE else 1)

if sum(fails) == 0:
    print("ALL CHECKS PASS")
    sys.exit(0)
print("FAILURES:", fails)
sys.exit(1)

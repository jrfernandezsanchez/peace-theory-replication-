# The Manufacture of Grievance: Domestic Politics and the Supply of Spite.
# A. Proposition 1 (the cliff): the analytic best response - frontier-defending
#    g_c while the frontier is reachable and the dividend worth defending, the
#    rally corner g = 1 otherwise - attains the brute-force maximum on 2,000
#    randomized draws.
# B. Theorem 2 benchmark (kappa=0.8, a=0.6, r=d=1, w=0.5): iterated best
#    response from joint restraint reaches the frontier-riding regime
#    g = (0.75, 0), lambda = (1.25, 0.80), product exactly 1; from joint
#    mobilization it stays at the hawk corner, product 1.96, locked by
#    kappa*(kappa+a) = 1.12 > 1.
# C. No unilateral deviation from either regime is profitable.
# D. At w = 0.95 joint restraint itself slides to the hawk corner.
# E. The hawk basin is non-decreasing in w.
import numpy as np, sys

rng = np.random.default_rng(11)
fails = []
GG = np.linspace(0.0, 1.0, 2001)

def V(gi, gj, kap, a, r, d, w):
    li = kap + a * gi
    lj = kap + a * gj
    div = 1.0 if li * lj <= 1.0 + 1e-12 else 0.0
    return w * r * li + (1.0 - w) * d * div

def br_brute(gj, kap, a, r, d, w):
    vals = np.array([V(g, gj, kap, a, r, d, w) for g in GG])
    return GG[int(np.argmax(vals))], vals.max()

def br_analytic(gj, kap, a, r, d, w):
    lj = kap + a * gj
    if kap * lj > 1.0 + 1e-12:
        return 1.0                                # collapse: rally corner
    gc = np.clip((1.0 / lj - kap) / a, 0.0, 1.0)  # frontier-defending level
    return gc if V(gc, gj, kap, a, r, d, w) >= V(1.0, gj, kap, a, r, d, w) else 1.0

# --------------------------------------------------------- A. the cliff
badA = 0
for _ in range(2000):
    kap = rng.uniform(0.3, 0.95); a = rng.uniform(0.2, 1.2)
    r = rng.uniform(0.3, 2.0); d = rng.uniform(0.3, 2.0)
    w = rng.uniform(0.1, 0.9); gj = rng.uniform(0.0, 1.0)
    ga = br_analytic(gj, kap, a, r, d, w)
    _, vb = br_brute(gj, kap, a, r, d, w)
    if V(ga, gj, kap, a, r, d, w) < vb - 1e-9:
        badA += 1
print("A. Proposition 1: analytic best response attains the brute-force optimum, 2000 draws | failures %d" % badA)
fails.append(badA)

# ---------------------------------------------------- B. the benchmark
kap, a, r, d, w = 0.8, 0.6, 1.0, 1.0, 0.5

def iterate(g0, w_):
    g = list(g0)
    for _ in range(200):
        g[0] = br_analytic(g[1], kap, a, r, d, w_)
        g[1] = br_analytic(g[0], kap, a, r, d, w_)
    return g

g_dove = iterate([0.0, 0.0], w)
lam = (kap + a * g_dove[0], kap + a * g_dove[1])
prod = lam[0] * lam[1]
ok_dove = (abs(g_dove[0] - 0.75) < 1e-9 and abs(g_dove[1]) < 1e-9
           and abs(lam[0] - 1.25) < 1e-9 and abs(lam[1] - 0.80) < 1e-9
           and abs(prod - 1.0) < 1e-9)
g_hawk = iterate([1.0, 1.0], w)
lamh = (kap + a * g_hawk[0], kap + a * g_hawk[1])
ok_hawk = (g_hawk == [1.0, 1.0] and abs(lamh[0] * lamh[1] - 1.96) < 1e-12
           and kap * (kap + a) > 1.0)
print("B. benchmark: restraint -> g=(%.2f,%.2f), product %.3f; mobilization -> hawk, product %.2f, lock %.2f > 1 | %s"
      % (g_dove[0], g_dove[1], prod, lamh[0] * lamh[1], kap * (kap + a),
         "ok" if ok_dove and ok_hawk else "FAIL"))
fails.append(0 if ok_dove and ok_hawk else 1)

# ---------------------------------------------------- C. no deviation
badC = 0
for (g1, g2) in (tuple(g_dove), tuple(g_hawk)):
    for i, (gi, gj) in enumerate(((g1, g2), (g2, g1))):
        _, best = br_brute(gj, kap, a, r, d, w)
        if V(gi, gj, kap, a, r, d, w) < best - 1e-9:
            badC += 1
print("C. both regimes are best-response fixed points (no profitable unilateral deviation) | violations %d" % badC)
fails.append(badC)

# -------------------------------------------------------- D. high w
g_hi = iterate([0.0, 0.0], 0.95)
okD = g_hi == [1.0, 1.0]
print("D. w = 0.95: joint restraint slides to the hawk corner | %s" % ("ok" if okD else "FAIL"))
fails.append(0 if okD else 1)

# ---------------------------------------------------- E. basin in w
fracs = []
inits = [(x, y) for x in np.linspace(0, 1, 11) for y in np.linspace(0, 1, 11)]
for w_ in (0.3, 0.5, 0.7, 0.9):
    hawk = sum(1 for g0 in inits
               if (lambda g: (kap + a * g[0]) * (kap + a * g[1]) > 1.0 + 1e-9)(iterate(list(g0), w_)))
    fracs.append(hawk / len(inits))
okE = all(fracs[i] <= fracs[i + 1] + 1e-12 for i in range(3))
print("E. hawk basin share by w in (0.3,0.5,0.7,0.9): %s, non-decreasing | %s"
      % (["%.2f" % f for f in fracs], "ok" if okE else "FAIL"))
fails.append(0 if okE else 1)

if sum(fails) == 0:
    print("ALL CHECKS PASS")
    sys.exit(0)
print("FAILURES:", fails)
sys.exit(1)

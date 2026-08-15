# The Two Yardsticks: Heterogeneous Valuation and the Generalized Product Frontier.
# A. Theorem 3 (symmetric pivot): f(1/2) = g(1/2) = 1 for every index pair.
# B. Theorem 1 (frontier) against a grid-search oracle on randomized
#    heterogeneous instances (2,973 in the paper's setup).
# C. Corollary 1.1: feasibility iff lam_R*lam_U <= Delta(x*).
# D. Proposition 2: affine agreement of the yardsticks gives Delta identically 1.
# E. Asymmetric conflict positions move the threshold in both directions;
#    common-index pairs stay exactly at 1.
# F. Proposition 4: the threshold moves linearly in the divergence.
# G. Section 5: the elicited product at the conflict allocation predicts
#    feasibility better than the true product.
import numpy as np, sys

rng = np.random.default_rng(9)
fails = []

FAMILY = {
    "linear": ((lambda x: x), (lambda x: np.ones_like(np.asarray(x, dtype=float)))),
    "pow0.5": ((lambda x: np.power(x, 0.5)), (lambda x: 0.5 * np.power(x, -0.5))),
    "pow0.2": ((lambda x: np.power(x, 0.2)), (lambda x: 0.2 * np.power(x, -0.8))),
    "pow2":   ((lambda x: np.power(x, 2.0)), (lambda x: 2.0 * x)),
    "pow5":   ((lambda x: np.power(x, 5.0)), (lambda x: 5.0 * np.power(x, 4.0))),
    "log":    ((lambda x: np.log(x + 0.05)), (lambda x: 1.0 / (x + 0.05))),
}
NAMES = list(FAMILY)

def gains(vR, vU, qR, qU):
    def f(x):  # R's exchange rate, in R's currency
        return (vR(x) - vR(qR)) / (vR(1.0 - x) - vR(qU))
    def g(x):  # U's exchange rate, in U's currency
        return (vU(1.0 - x) - vU(qU)) / (vU(x) - vU(qR))
    return f, g

def x_star(f, lam, qR, qU):
    lo, hi = qR + 1e-12, 1.0 - qU - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < lam: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def oracle(vR, vU, lR, lU, qR, qU, n=20001):
    x = np.linspace(qR + 1e-9, 1.0 - qU - 1e-9, n)
    AR = vR(x) - vR(qR); BR = vR(1.0 - x) - vR(qU)
    AU = vU(1.0 - x) - vU(qU); BU = vU(x) - vU(qR)
    ok = (AR >= lR * BR - 1e-12) & (AU >= lU * BU - 1e-12)
    return bool(ok.any())

# ------------------------------------------------------------- A. the pivot
badA = 0
for q in (0.10, 0.25, 0.40):
    for a in NAMES:
        for b in NAMES:
            f, g = gains(FAMILY[a][0], FAMILY[b][0], q, q)
            if abs(f(0.5) - 1.0) > 1e-12 or abs(g(0.5) - 1.0) > 1e-12:
                badA += 1
print("A. pivot: f(1/2)=g(1/2)=1 over 3 symmetric q x 36 ordered pairs | violations %d" % badA)
fails.append(badA)

# ----------------------------------------- B/C. frontier and product form
N = 2973
badB = badC = 0; testedB = 0
for _ in range(N):
    a, b = rng.choice(NAMES, 2)
    vR, vU = FAMILY[a][0], FAMILY[b][0]
    qR = rng.uniform(0.05, 0.45); qU = rng.uniform(0.05, min(0.9 - qR, 0.45))
    lR = rng.uniform(0.05, 4.0)
    f, g = gains(vR, vU, qR, qU)
    xs = x_star(f, lR, qR, qU)
    Phi = g(xs)
    lU = rng.uniform(0.05, 4.0)
    if abs(lU - Phi) < 0.02 * (1.0 + Phi):
        continue                       # skip the frontier's own neighbourhood
    testedB += 1
    pred = lU <= Phi
    if oracle(vR, vU, lR, lU, qR, qU) != pred:
        badB += 1
    if ((lR * lU <= f(xs) * g(xs)) != pred):
        badC += 1
print("B. Theorem 1: frontier vs oracle | %d instances, disagreements %d" % (testedB, badB))
print("C. Corollary 1.1: product <= Delta(x*) equivalent to the frontier test | violations %d" % badC)
fails += [badB, badC]

# -------------------------------------------------- D. affine agreement
vR = FAMILY["pow0.2"][0]
vU = lambda x: 2.5 * vR(x) + 0.7
badD = 0
for _ in range(20):
    qR = rng.uniform(0.05, 0.45); qU = rng.uniform(0.05, min(0.9 - qR, 0.45))
    f, g = gains(vR, vU, qR, qU)
    wdt = 1.0 - qU - qR
    x = np.linspace(qR + 0.02 * wdt, 1.0 - qU - 0.02 * wdt, 501)
    if np.max(np.abs(f(x) * g(x) - 1.0)) > 1e-9:
        badD += 1
print("D. Proposition 2: v_U affine in v_R gives Delta = 1 identically | violations %d" % badD)
fails.append(badD)

# ------------------------------------- E. asymmetric conflict positions
qR, qU = 0.20, 0.45
above = below = 0; badE = 0
for a in NAMES:
    for b in NAMES:
        f, g = gains(FAMILY[a][0], FAMILY[b][0], qR, qU)
        Phi1 = g(x_star(f, 1.0, qR, qU))
        if a == b:
            if abs(Phi1 - 1.0) > 1e-9: badE += 1
        else:
            if Phi1 > 1.0 + 1e-9: above += 1
            elif Phi1 < 1.0 - 1e-9: below += 1
okE = (badE == 0) and (above >= 3) and (below >= 3)
print("E. asymmetric q=(0.20,0.45): Phi(1)>1 in %d pairs, <1 in %d, common pairs exactly 1 | %s"
      % (above, below, "ok" if okE else "FAIL"))
fails.append(0 if okE else 1)

# ----------------------------------------------- F. smooth degradation
vRb, dvRb = FAMILY["pow0.5"]
h = lambda x: np.sin(3.0 * x) + 0.5 * x * x
ratios = []
for eps in (0.08, 0.04, 0.02, 0.01):
    vU = (lambda e: (lambda x: vRb(x) + e * h(x)))(eps)
    f, g = gains(vRb, vU, 0.20, 0.45)
    Phi1 = g(x_star(f, 1.0, 0.20, 0.45))
    ratios.append(abs(Phi1 - 1.0) / eps)
tail = ratios[1:]
okF = max(tail) / min(tail) < 1.25
print("F. Proposition 4: |Phi(1)-1|/eps stabilises: %s | %s"
      % (["%.3f" % r for r in ratios], "ok" if okF else "FAIL"))
fails.append(0 if okF else 1)

# ------------------------------------ G. contamination as information
N = 2932
acc_true = acc_elic = tot = 0
for _ in range(N):
    a, b = rng.choice(NAMES, 2)
    if a == b:
        continue                     # heterogeneous instances
    vR, dvR = FAMILY[a]; vU, dvU = FAMILY[b]
    qR = rng.uniform(0.05, 0.45); qU = rng.uniform(0.05, min(0.9 - qR, 0.45))
    lR = rng.uniform(0.05, 3.0); lU = rng.uniform(0.05, 3.0)
    feas = oracle(vR, vU, lR, lU, qR, qU)
    tot += 1
    if (lR * lU <= 1.0) == feas:
        acc_true += 1
    elic = (lR * lU) * (dvR(qU) / dvR(qR)) * (dvU(qR) / dvU(qU))
    if (elic <= 1.0) == feas:
        acc_elic += 1
pt, pe = 100.0 * acc_true / tot, 100.0 * acc_elic / tot
okG = pe > pt and pe > 80.0
print("G. Section 5: elicited product at the conflict allocation %.1f%% vs true product %.1f%% | %s"
      % (pe, pt, "ok" if okG else "FAIL"))
fails.append(0 if okG else 1)

if sum(fails) == 0:
    print("ALL CHECKS PASS")
    sys.exit(0)
print("FAILURES:", fails)
sys.exit(1)

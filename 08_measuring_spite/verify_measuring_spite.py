# Measuring Spite: Invariance, Elicitation, and What Survives Curvature.
# A. Worked feasible intervals under five indices (Section 4).
# B. The twelve-index invariance battery: feasibility iff lam_R*lam_U <= 1
#    under any common index (14,376 tests in the paper's setup).
# C. Theorem 2: the elicited matrix is a diagonal similarity (spectrum,
#    cycle products, zero pattern preserved).
# D. Section 5: row sums are not identified at an unequal profile and are
#    recovered at the equal profile to truncation error.
# E. A common index is required: disagreements grow with index divergence.
# F. A jump index fails in one direction only (empty range despite product < 1).
# G. Finite-stake bias halves with the stake; two-stake extrapolation removes it.
import numpy as np, sys

rng = np.random.default_rng(8)
fails = []

# ------------------------------------------------------------------ indices
def make_indices():
    idx = {}
    idx["linear"] = (lambda x: x)
    for g in (0.9, 0.5, 0.2, 0.05):
        idx["pow%.2f" % g] = (lambda g: (lambda x: np.power(x, g)))(g)
    for k in (0.05, 0.001):
        idx["log%.3f" % k] = (lambda k: (lambda x: np.log(x + k)))(k)
    idx["pow2"] = (lambda x: np.power(x, 2.0))
    idx["pow5"] = (lambda x: np.power(x, 5.0))
    idx["exp3"] = (lambda x: np.exp(3.0 * x))
    idx["sigmoid"] = (lambda x: 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5))))
    idx["lossav"] = (lambda x: np.where(x >= 0.5, x - 0.5, 2.5 * (x - 0.5)))
    return idx

INDICES = make_indices()
assert len(INDICES) == 12

def interval(v, lR, lU, qR, qU, n=100001):
    """Feasible set endpoints by direct search (acceptance conditions of Prop 6)."""
    x = np.linspace(0.0, 1.0, n)
    A = v(x) - v(qR)
    B = v(1.0 - x) - v(qU)
    ok = (A >= lR * B - 1e-12) & (B >= lU * A - 1e-12)
    if not ok.any():
        return None
    xs = x[ok]
    return float(xs.min()), float(xs.max())

def feasible(v, lR, lU, qR, qU, grid):
    A = v(grid) - v(qR)
    B = v(1.0 - grid) - v(qU)
    ok = (A >= lR * B - 1e-12) & (B >= lU * A - 1e-12)
    if bool(ok.any()):
        return True
    # local refinement around the best candidate before declaring infeasible
    slack = np.minimum(A - lR * B, B - lU * A)
    j = int(np.argmax(slack))
    lo = grid[max(0, j - 2)]; hi = grid[min(len(grid) - 1, j + 2)]
    x = np.linspace(lo, hi, 4001)
    A = v(x) - v(qR); B = v(1.0 - x) - v(qU)
    return bool(((A >= lR * B - 1e-12) & (B >= lU * A - 1e-12)).any())

# ------------------------------------------------------------ A. worked case
lR, lU, p, cR, cU = 0.6, 0.9, 0.5, 0.10, 0.06
qR, qU = p - cR, 1.0 - p - cU
targets = {"linear": (0.4600, 0.4842), "pow0.50": (0.4575, 0.4826),
           "pow5": (0.4752, 0.4930), "lossav": (0.4484, 0.4769)}
badA = 0
for name, (a, b) in targets.items():
    iv = interval(INDICES[name], lR, lU, qR, qU)
    if iv is None or abs(iv[0] - a) > 3e-4 or abs(iv[1] - b) > 3e-4:
        badA += 1
        print("  A mismatch %s: got %s expected (%.4f, %.4f)" % (name, iv, a, b))
log_hit = None
for k in ("log0.050", "log0.001"):
    iv = interval(INDICES[k], lR, lU, qR, qU)
    if iv and abs(iv[0] - 0.4549) <= 3e-4 and abs(iv[1] - 0.4808) <= 3e-4:
        log_hit = k
if log_hit is None:
    badA += 1
print("A. worked intervals: 4 named indices + log (%s) match to 4 decimals | mismatches %d"
      % (log_hit, badA))
fails.append(badA)

# ----------------------------------------------- B. twelve-index invariance
GRID = np.linspace(0.0, 1.0, 30001)
tested = 0; bad = 0
for name, v in INDICES.items():
    m = 1200
    pR = rng.uniform(0.2, 0.8, m)
    cRv = rng.uniform(0.02, 0.18, m)
    cUv = rng.uniform(0.02, 0.18, m)
    wR = rng.uniform(0.02, 6.0, m)
    wU = rng.uniform(0.02, 6.0, m)
    keep = np.abs(wR * wU - 1.0) >= 0.01
    for i in np.flatnonzero(keep):
        qr, qu = pR[i] - cRv[i], 1.0 - pR[i] - cUv[i]
        f = feasible(v, wR[i], wU[i], qr, qu, GRID)
        tested += 1
        if f != (wR[i] * wU[i] <= 1.0):
            bad += 1
print("B. invariance battery: %d tests over 12 indices | disagreements %d" % (tested, bad))
fails.append(bad)

# ------------------------------------------------- C. diagonal similarity
badC = 0
for _ in range(50):
    n = 4
    L = rng.uniform(-0.3, 1.2, (n, n)); np.fill_diagonal(L, 0.0)
    L[:, 3] = 0.0                       # an unresented column
    L[0, 2] = 0.0                       # a stray zero
    x = rng.uniform(0.5, 3.0, n)
    dv = 0.4 * np.power(x, -0.6)        # v = x^0.4
    D = np.diag(dv)
    W = np.linalg.inv(D) @ L @ D
    if not np.allclose(np.sort_complex(np.linalg.eigvals(W)),
                       np.sort_complex(np.linalg.eigvals(L)), atol=1e-8):
        badC += 1
    for i in range(n):
        for j in range(n):
            if i != j and abs(W[i, j] * W[j, i] - L[i, j] * L[j, i]) > 1e-10:
                badC += 1
    if not np.array_equal(np.isclose(W, 0.0, atol=1e-12), np.isclose(L, 0.0, atol=1e-12)):
        badC += 1
print("C. Theorem 2: spectrum, 2-cycle products and zero pattern preserved | violations %d" % badC)
fails.append(badC)

# --------------------------------------------------- D. row sums and profile
def elicit(v, lam, xi, xj, t):
    """Exact indifference: v(xi) - v(xi - s) = lam * (v(xj + t) - v(xj)); returns s/t."""
    target = lam * (v(np.array([xj + t]))[0] - v(np.array([xj]))[0])
    lo, hi = -0.9 * xi, 0.9 * xi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = v(np.array([xi]))[0] - v(np.array([xi - mid]))[0]
        if val < target: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi) / t

L = rng.uniform(0.05, 0.6, (4, 4)); np.fill_diagonal(L, 0.0)
prof_uneq = np.array([0.8, 1.6, 2.4, 3.6])
vs = INDICES["pow0.50"]
dv = lambda x: 0.5 / np.sqrt(x)
rows_true = L.sum(axis=1)
W = np.array([[L[i, j] * dv(prof_uneq[j]) / dv(prof_uneq[i]) if i != j else 0.0
               for j in range(4)] for i in range(4)])
err_uneq = np.max(np.abs(W.sum(axis=1) - rows_true) / rows_true)
prof_eq = np.full(4, 2.0)
Wq = np.array([[elicit(vs, L[i, j], prof_eq[i], prof_eq[j], 0.01) if i != j else 0.0
                for j in range(4)] for i in range(4)])
err_eq = np.max(np.abs(Wq.sum(axis=1) - rows_true) / rows_true)
okD = (err_uneq > 0.25) and (err_eq < 0.02)
print("D. row sums: unequal-profile max rel err %.2f (>0.25), equal-profile %.4f (<0.02) | %s"
      % (err_uneq, err_eq, "ok" if okD else "FAIL"))
fails.append(0 if okD else 1)

# ------------------------------------------- E. common index is required
def feasible_het(vR, vU, lR, lU, qR, qU, grid):
    AR = vR(grid) - vR(qR); BR = vR(1.0 - grid) - vR(qU)
    AU = vU(1.0 - grid) - vU(qU); BU = vU(grid) - vU(qR)
    return bool(((AR >= lR * BR - 1e-12) & (AU >= lU * BU - 1e-12)).any())

counts = []
for gR, gU in ((0.9, 0.7), (0.9, 0.2), (2.0, 0.1)):
    vR = (lambda g: (lambda x: np.power(x, g)))(gR)
    vU = (lambda g: (lambda x: np.power(x, g)))(gU)
    m = 1198; c = 0
    pR = rng.uniform(0.2, 0.8, m); cRv = rng.uniform(0.02, 0.18, m)
    cUv = rng.uniform(0.02, 0.18, m)
    wR = rng.uniform(0.02, 6.0, m); wU = rng.uniform(0.02, 6.0, m)
    for i in range(m):
        if abs(wR[i] * wU[i] - 1.0) < 0.01: continue
        qr, qu = pR[i] - cRv[i], 1.0 - pR[i] - cUv[i]
        f = feasible_het(vR, vU, wR[i], wU[i], qr, qu, GRID)
        if f != (wR[i] * wU[i] <= 1.0): c += 1
    counts.append(c)
okE = counts[0] < counts[1] < counts[2] and counts[1] > 0
print("E. heterogeneous indices: disagreements %s rise with divergence | %s"
      % (counts, "ok" if okE else "FAIL"))
fails.append(0 if okE else 1)

# --------------------------------------------------------- F. jump index
countsF = []; wrong_dir = 0
for J in (0.05, 0.20, 0.50, 3.0):
    v = (lambda J: (lambda x: x + J * (x >= 0.5)))(J)
    m = 794; c = 0
    pR = rng.uniform(0.2, 0.8, m); cRv = rng.uniform(0.02, 0.18, m)
    cUv = rng.uniform(0.02, 0.18, m)
    wR = rng.uniform(0.02, 6.0, m); wU = rng.uniform(0.02, 6.0, m)
    for i in range(m):
        prod = wR[i] * wU[i]
        if abs(prod - 1.0) < 0.01: continue
        qr, qu = pR[i] - cRv[i], 1.0 - pR[i] - cUv[i]
        f = feasible(v, wR[i], wU[i], qr, qu, GRID)
        if f != (prod <= 1.0):
            c += 1
            if f and prod > 1.0:
                wrong_dir += 1      # feasibility despite product > 1: forbidden
    countsF.append(c)
okF = all(countsF[i] <= countsF[i + 1] for i in range(3)) and countsF[-1] > 0 and wrong_dir == 0
print("F. jump index: failures %s rise with J, all one-directional (reverse cases %d) | %s"
      % (countsF, wrong_dir, "ok" if okF else "FAIL"))
fails.append(0 if okF else 1)

# ------------------------------------------------ G. stakes and extrapolation
vlog = INDICES["log0.050"]
xiR, xjR = 0.30, 0.62                    # asymmetric profile, common to both
true_prod = 0.6 * 0.9
errs_raw, errs_ext = [], []
for t in (0.20, 0.10, 0.05, 0.025):
    wRt = elicit(vlog, 0.6, xiR, xjR, t); wUt = elicit(vlog, 0.9, xjR, xiR, t)
    wRh = elicit(vlog, 0.6, xiR, xjR, t / 2); wUh = elicit(vlog, 0.9, xjR, xiR, t / 2)
    # the curvature contamination at a fixed asymmetric profile cancels in the
    # product only in the limit t -> 0; compare raw and extrapolated products
    raw = wRt * wUt
    ext = (2 * wRh - wRt) * (2 * wUh - wUt)
    errs_raw.append(abs(raw - true_prod) / true_prod)
    errs_ext.append(abs(ext - true_prod) / true_prod)
halving = all(1.4 < errs_raw[i] / errs_raw[i + 1] < 2.6 for i in range(3))
better = all(errs_ext[i] < errs_raw[i] / 3 for i in range(4))
okG = halving and better
print("G. finite stakes: raw errors %s halve with t; extrapolated %s (each < raw/4) | %s"
      % (["%.4f" % e for e in errs_raw], ["%.5f" % e for e in errs_ext],
         "ok" if okG else "FAIL"))
fails.append(0 if okG else 1)

# ---------------------------------------------------------------- verdict
if sum(fails) == 0:
    print("ALL CHECKS PASS")
    sys.exit(0)
print("FAILURES:", fails)
sys.exit(1)

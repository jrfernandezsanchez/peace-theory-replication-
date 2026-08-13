# Spite, Divisibility and the Existence of Peace: A General Product Theorem.
# Theorem 1(a)/(b) against a grid oracle, Corollary 1.1 (strict competitiveness),
# Corollary 1.3 (canonical benchmark closed form), Theorem 2(b) (randomization
# width), Theorem 3 remark (mutual impoverishment counterexample), Theorem 4
# (inversion of deterrence), Proposition 3 (risk aversion beyond the frontier).
import numpy as np, sys

rng = np.random.default_rng(1); fails = []
GRID = np.linspace(0.0, 1.0, 200001)          # oracle mesh over pure divisions
TOL = 1e-12
BAND = 0.05                                    # exclude the knife edge |prod-1|<BAND


def acceptable_pure(v, VR, VU, lR, lU, grid=GRID):
    """Oracle: does some pure division satisfy both acceptance conditions?
    Searches the whole unit interval; uses no property of the theorem."""
    A = v(grid) - VR
    B = v(1.0 - grid) - VU
    ok = (A >= lR * B - 1e-10) & (B >= lU * A - 1e-10)
    return bool(ok.any()), (grid[ok] if ok.any() else np.array([]))


# ---------------------------------------------------------------- A. Theorem 1(a)
# Conflict ex post inefficient (chat_R + chat_U < 1): range non-empty iff prod <= 1.
bad = 0; tested = 0
for _ in range(3000):
    g = rng.uniform(0.3, 3.0)                 # v(x) = x^g, continuous, strictly incr.
    v = (lambda g: (lambda x: np.power(x, g)))(g)
    cR = rng.uniform(0.02, 0.85); cU = rng.uniform(0.02, 0.95 - cR)  # sum < 1
    lR = rng.uniform(0.0, 3.0); lU = rng.uniform(0.0, 3.0)
    prod = lR * lU
    if abs(prod - 1.0) < BAND:                # knife edge excluded, see remark below
        continue
    tested += 1
    found, _ = acceptable_pure(v, v(cR), v(cU), lR, lU)
    if found != (prod <= 1.0):
        bad += 1
print(f"A. Theorem 1(a): non-empty range iff lR*lU<=1 | {tested} draws, mismatches {bad}")
fails += [bad]

# ---------------------------------------------------------------- B. Theorem 1(b)
# Conflict strictly dominates every division (chat_R + chat_U > 1): iff prod >= 1.
bad = 0; tested = 0
for _ in range(3000):
    g = rng.uniform(0.3, 3.0)
    v = (lambda g: (lambda x: np.power(x, g)))(g)
    cR = rng.uniform(0.15, 0.95); cU = rng.uniform(1.02 - cR, 0.98)  # sum > 1
    lR = rng.uniform(0.0, 4.0); lU = rng.uniform(0.0, 4.0)
    prod = lR * lU
    if abs(prod - 1.0) < BAND:
        continue
    tested += 1
    found, _ = acceptable_pure(v, v(cR), v(cU), lR, lU)
    if found != (prod >= 1.0):
        bad += 1
print(f"B. Theorem 1(b): non-empty range iff lR*lU>=1 | {tested} draws, mismatches {bad}")
fails += [bad]

# ------------------------------------------------- B'. the knife edge, checked apart
# At chat_R + chat_U = 1 the division x = chat_R gives A = B = 0 and is acceptable
# for every spite pair, whatever the index.
bad = 0
for _ in range(500):
    g = rng.uniform(0.3, 3.0); cR = rng.uniform(0.05, 0.95); cU = 1.0 - cR
    v = lambda x: np.power(x, g)
    A = v(cR) - v(cR); B = v(1 - cR) - v(cU)
    lR = rng.uniform(0, 5); lU = rng.uniform(0, 5)
    if not (A >= lR * B - 1e-12 and B >= lU * A - 1e-12):
        bad += 1
print(f"B'. knife edge chat_R+chat_U=1: x=chat_R acceptable for every (lR,lU) | violations {bad}")
fails += [bad]

# ------------------------------------------------------------- C. Corollary 1.1
# U_R + lR*U_U vanishes identically in (Z_R, Z_U) iff lR*lU = 1.
worst_on, worst_off = 0.0, np.inf
for _ in range(2000):
    lR = rng.uniform(0.05, 5.0); lU_on = 1.0 / lR
    lU_off = lU_on * rng.choice([rng.uniform(0.2, 0.9), rng.uniform(1.1, 3.0)])
    ZR, ZU = rng.uniform(-3, 3, 2)
    on = (ZR - lR * ZU) + lR * (ZU - lU_on * ZR)
    off = (ZR - lR * ZU) + lR * (ZU - lU_off * ZR)
    worst_on = max(worst_on, abs(on))
    if abs(ZR) > 0.2:
        worst_off = min(worst_off, abs(off))
print(f"C. Corollary 1.1: |U_R+lR*U_U| at the frontier max {worst_on:.2e}; "
      f"off it min {worst_off:.2e}")
fails += [0 if (worst_on < 1e-9 and worst_off > 1e-3) else 1]

# ------------------------------------------------------------- D. Corollary 1.3
# Canonical benchmark closed form against the grid oracle.
worst = 0.0; bad = 0
for _ in range(3000):
    p = rng.uniform(0.05, 0.95)
    cR_ = rng.uniform(0.01, 0.4); cU_ = rng.uniform(0.01, min(0.4, 1 - p - 1e-3))
    uR = p - cR_; uU = (1 - p) - cU_
    if uR <= 0 or uU <= 0:
        continue
    lR = rng.uniform(0, 2.0); lU = rng.uniform(0, 2.0)
    aR = lR / (1 + lR); aU = lU / (1 + lU); c = cR_ + cU_
    if aR + aU >= 1 - 1e-3:                   # positive width only
        continue
    s_min = uR + aR * c; s_max = uR + (1 - aU) * c; W = c * (1 - aR - aU)
    v = lambda x: x
    found, xs = acceptable_pure(v, uR, uU, lR, lU)
    if not found:
        bad += 1; continue
    worst = max(worst, abs(xs.min() - s_min), abs(xs.max() - s_max),
                abs((xs.max() - xs.min()) - W))
print(f"D. Corollary 1.3: closed form vs grid oracle | max error {worst:.2e}, "
      f"empty-range failures {bad}")
fails += [0 if (worst < 1e-4 and bad == 0) else 1]

# ------------------------------------------------------------- E. Theorem 2(b)
# Lotteries over the two extreme divisions: width S_v(1-lR lU)/((1+lR)(1+lU)).
worst = 0.0
wgrid = np.linspace(0, 1, 200001)
for _ in range(1500):
    VR = rng.uniform(0.02, 0.6); VU = rng.uniform(0.02, 0.95 - VR)
    Sv = 1 - VR - VU
    lR = rng.uniform(0, 1.5); lU = rng.uniform(0, 1.5)
    if lR * lU > 1 - 1e-3:
        continue
    A = wgrid - VR; B = (1 - wgrid) - VU
    ok = (A >= lR * B - 1e-10) & (B >= lU * A - 1e-10)
    if not ok.any():
        worst = np.inf; break
    W_hat = wgrid[ok].max() - wgrid[ok].min()
    W = Sv * (1 - lR * lU) / ((1 + lR) * (1 + lU))
    worst = max(worst, abs(W_hat - W))
print(f"E. Theorem 2(b): lottery width formula vs grid | max error {worst:.2e}")
fails += [0 if worst < 1e-4 else 1]

# ------------------------------------- F. Theorem 3 remark: mutual impoverishment
# v = sqrt, conflict the decisive split (0.45, 0.45), lR = lU = 2 (product 4).
v = np.sqrt
VR = VU = v(0.45)
lR = lU = 2.0
found, _ = acceptable_pure(v, VR, VU, lR, lU)
xa, xb = 0.949, 0.051
EvR = 0.5 * (v(xa) + v(xb)); EvU = 0.5 * (v(1 - xa) + v(1 - xb))
A = EvR - VR; B = EvU - VU
lottery_ok = (A >= lR * B - 1e-12) and (B >= lU * A - 1e-12)
strict_R = (EvR - lR * EvU) > (VR - lR * VU) + 1e-9
strict_U = (EvU - lU * EvR) > (VU - lU * VR) + 1e-9
print(f"F. Theorem 3 remark: no pure division acceptable {not found}; "
      f"lottery accepted {lottery_ok}; strictly preferred by both "
      f"{strict_R and strict_U}; A=B={A:.4f}<0 {A < 0 and B < 0}")
fails += [0 if (not found and lottery_ok and strict_R and strict_U and A < 0) else 1]

# ------------------------------------------------------------- G. Theorem 4
# (A) the four partials; (B) sign(dW/dc) = sign(1 - lR lU). Numerical derivatives.
h = 1e-6; worstA = 0.0; badB = 0
def bounds(p, cR_, cU_, lR, lU):
    uR = p - cR_; c = cR_ + cU_
    aR = lR / (1 + lR); aU = lU / (1 + lU)
    return uR + aR * c, uR + (1 - aU) * c
for _ in range(2000):
    p = rng.uniform(0.2, 0.8); cR_ = rng.uniform(0.05, 0.3); cU_ = rng.uniform(0.05, 0.3)
    lR = rng.uniform(0, 3.0); lU = rng.uniform(0, 3.0)
    s0, S0 = bounds(p, cR_, cU_, lR, lU)
    s1, S1 = bounds(p, cR_ + h, cU_, lR, lU)
    s2, S2 = bounds(p, cR_, cU_ + h, lR, lU)
    worstA = max(worstA,
                 abs((s1 - s0) / h - (-1 / (1 + lR))),
                 abs((s2 - s0) / h - (lR / (1 + lR))),
                 abs((S2 - S0) / h - (1 / (1 + lU))),
                 abs((S1 - S0) / h - (-lU / (1 + lU))))
    dW_R = ((S1 - s1) - (S0 - s0)) / h
    dW_U = ((S2 - s2) - (S0 - s0)) / h
    pred = (1 - lR * lU) / ((1 + lR) * (1 + lU))
    if abs(dW_R - pred) > 1e-5 or abs(dW_U - pred) > 1e-5:
        badB += 1
print(f"G. Theorem 4: four partials max error {worstA:.2e}; "
      f"dW/dc_R = dW/dc_U = (1-lR lU)/((1+lR)(1+lU)) | mismatches {badB}")
fails += [0 if (worstA < 1e-4 and badB == 0) else 1]

# ------------------------------------------------------------- H. Proposition 3
# Symmetric spite, p = 1/2, symmetric destruction: M(1/2) = (1-l)*pi exactly, and
# the range is non-empty iff (1-l)*pi >= 0.
worst = 0.0; bad = 0
for _ in range(2000):
    g = rng.uniform(0.3, 3.0); v = lambda x: np.power(x, g)
    kap = rng.uniform(0.01, 0.6); lam = rng.uniform(0.0, 2.5)
    if abs(lam - 1.0) < 1e-2:
        continue
    V = 0.5 * v(1 - kap)
    pi = v(0.5) - 0.5 * v(1 - kap)
    M_half = (v(0.5) - lam * v(0.5)) - (V - lam * V)
    worst = max(worst, abs(M_half - (1 - lam) * pi))
    found, _ = acceptable_pure(v, V, V, lam, lam)
    if found != ((1 - lam) * pi >= 0):
        bad += 1
print(f"H. Proposition 3: M(1/2)=(1-l)*pi max error {worst:.2e}; "
      f"non-empty iff (1-l)*pi>=0 | mismatches {bad}")
fails += [0 if (worst < 1e-10 and bad == 0) else 1]

print("RESULT:", "all claims verified" if not any(fails) else "FAILURES PRESENT")
sys.exit(1 if any(fails) else 0)

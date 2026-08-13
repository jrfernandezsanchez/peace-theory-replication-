# Rows and Columns: Relative Gains, Group Size and the Feasibility of Multilateral
# Division: Snidal (1991) in a Bargaining Setting (v1.0).
# Lemma A1 (the reduction) and Lemma A2 (spectral sufficiency); Proposition 2
# (nesting and the degeneracy of the dyad); Proposition 3 (rho = r at every n);
# Proposition 4 (feasibility iff r <= 1 at every n) with its Farkas certificate;
# Section 5 (the shrinking budget and the saturation of Snidal's own recursion);
# Theorem 5, Corollary 6 and the three-state example of Section 7.
# Feasibility is decided throughout by a linear program on the cone, not by the
# spectral or row-sum properties being tested.
import numpy as np, sys
from scipy.optimize import linprog

rng = np.random.default_rng(2); fails = []


def feasible(Lam, tol=1e-7):
    """Oracle for the cone condition of Lemma A1: does there exist z, free in sign,
    with (I - Lam)z >= 0 and 1'z > 0? The cone is homogeneous, so z is boxed and
    the question is whether the maximum of 1'z is strictly positive."""
    n = len(Lam); M = np.eye(n) - Lam
    res = linprog(c=-np.ones(n), A_ub=-M, b_ub=np.zeros(n),
                  bounds=[(-1, 1)] * n, method="highs")
    return bool(res.status == 0 and -res.fun > tol)


def division_exists(Lam, p, c, tol=1e-9):
    """Independent oracle in the original variables: is there a division x with
    1'x = 1 and V_i(x) >= V_i(q) for every i, where q = p - c?"""
    n = len(Lam); M = np.eye(n) - Lam; q = p - c
    res = linprog(c=np.zeros(n), A_ub=-M, b_ub=-(M @ q),
                  A_eq=np.ones((1, n)), b_eq=[1.0],
                  bounds=[(None, None)] * n, method="highs")
    return res.status == 0


# --------------------------------------------------------------- A. Lemma A1
# The two formulations agree on random instances: a division beating conflict for
# everyone exists iff the cone contains a vector of positive coordinate sum.
mis = 0
for _ in range(1500):
    n = int(rng.integers(2, 8))
    Lam = rng.uniform(0, 1.6, (n, n)); np.fill_diagonal(Lam, 0)
    p = rng.dirichlet(np.ones(n))
    c = rng.uniform(0.01, 0.15, n)
    if division_exists(Lam, p, c) != feasible(Lam):
        mis += 1
print(f"A. Lemma A1: division formulation and cone formulation agree | 1500 random "
      f"instances, mismatches {mis}")
fails += [mis]

# --------------------------------------------------------------- B. Lemma A2
# rho(Lam) < 1 gives z = M^{-1}s with every constraint strict, for any s > 0.
bad = 0
for _ in range(1500):
    n = int(rng.integers(2, 9))
    Lam = rng.uniform(0, 1, (n, n)); np.fill_diagonal(Lam, 0)
    Lam *= rng.uniform(0.05, 0.95) / max(abs(np.linalg.eigvals(Lam)))
    M = np.eye(n) - Lam
    s = rng.uniform(0.1, 1.0, n)
    z = np.linalg.solve(M, s)
    if np.min(M @ z) <= 0 or z.sum() <= s.sum() - 1e-9 or not feasible(Lam):
        bad += 1
print(f"B. Lemma A2: rho(Lam)<1 gives a strictly preferred division | failures {bad}")
fails += [bad]

# ------------------------------------------------------------- C. Proposition 3
# Any non-negative matrix with constant row sums r has spectral radius exactly r.
worst = 0.0
for _ in range(3000):
    n = int(rng.integers(2, 12)); r = rng.uniform(0.1, 3.0)
    W = rng.uniform(0, 1, (n, n)); np.fill_diagonal(W, 0)
    W = W / W.sum(1, keepdims=True)                  # Snidal's normalization
    worst = max(worst, abs(max(abs(np.linalg.eigvals(r * W))) - r))
print(f"C. Proposition 3: rho(Lam) = r | max error {worst:.2e} over 3000 matrices, "
      f"n = 2..11, non-uniform weights")
fails += [0 if worst < 1e-8 else 1]

# ------------------------------------------------------------- D. Proposition 4
# Uniform weights w_ij = 1/(n-1): feasible iff r <= 1, at every n from 2 to 40.
mis = 0; tested = 0
for n in range(2, 41):
    for r in [0.05, 0.3, 0.7, 0.95, 1.0, 1.05, 1.5, 2.5, 9.0]:
        Lam = np.full((n, n), r / (n - 1)); np.fill_diagonal(Lam, 0)
        tested += 1
        if feasible(Lam) != (r <= 1.0 + 1e-12):
            mis += 1
print(f"D. Proposition 4: feasible iff r <= 1 | n = 2..40, {tested} cases, "
      f"mismatches {mis} (the case r = 1 included)")
fails += [mis]

# ------------------------------------------- E. the Farkas certificate of Prop. 4
# For r > 1 the certificate t = a*1 with a = 1/(r-1) >= 0 satisfies Lam' t = t + 1.
worst = 0.0; neg = 0
for n in range(2, 41):
    for r in [1.05, 1.3, 2.0, 5.0]:
        Lam = np.full((n, n), r / (n - 1)); np.fill_diagonal(Lam, 0)
        a = 1.0 / (r - 1.0); t = a * np.ones(n)
        worst = max(worst, np.max(np.abs(Lam.T @ t - (t + 1))))
        neg += int(a < 0)
print(f"E. Farkas certificate: a = 1/(r-1) | max residual {worst:.2e}, sign "
      f"violations {neg}")
fails += [0 if (worst < 1e-9 and neg == 0) else 1]

# ------------------------------------------ F. the halving that changes nothing
# n = 2 -> n = 3 halves lambda_12 from r to r/2 and moves the threshold not at all.
r = 1.4
same = all(feasible(np.where(np.eye(n) == 1, 0.0, np.full((n, n), rr / (n - 1))))
           == (rr <= 1.0) for n in (2, 3, 10, 20) for rr in (0.8, 1.4))
print(f"F. the halving: lambda_12 falls {r:.2f} -> {r / 2:.2f} from n=2 to n=3; "
      f"threshold unchanged at n = 2, 3, 10, 20 {same}")
fails += [0 if same else 1]

# --------------------------------------- G. Section 5: the budget that shrinks
# With sigma(n) = sum_j w_ij <= 1 the threshold becomes r*sigma(n) <= 1; and
# Snidal's own recursion w_n <= ((n-1)/n) w_{n-1} is saturated by w_n = w_1/n,
# a path along which the total carried n*w_n is constant.
worst = 0.0; mis = 0
for _ in range(400):
    n = int(rng.integers(3, 15)); r = rng.uniform(0.3, 3.0); sig = rng.uniform(0.2, 1.0)
    if abs(r * sig - 1.0) < 1e-3:
        continue
    Lam = np.full((n, n), r * sig / (n - 1)); np.fill_diagonal(Lam, 0)
    worst = max(worst, abs(max(abs(np.linalg.eigvals(Lam))) - r * sig))
    if feasible(Lam) != (r * sig <= 1.0):
        mis += 1
w1 = 0.7
w = [w1 / k for k in range(1, 21)]                   # the saturating path
recursion_ok = all(w[k] <= (k / (k + 1)) * w[k - 1] + 1e-15 for k in range(1, 20))
totals = [(k + 1) * w[k] for k in range(20)]
constant_total = max(totals) - min(totals) < 1e-12
print(f"G. Section 5: rho = r*sigma(n) max error {worst:.2e}, threshold r*sigma<=1 "
      f"mismatches {mis}; recursion saturated by w_n = w_1/n {recursion_ok}, total "
      f"carried constant {constant_total}")
fails += [0 if (worst < 1e-9 and mis == 0 and recursion_ok and constant_total) else 1]

# ------------------------------------------------------------- H. Proposition 2
# In the dyad the normalization forces lambda_12 = lambda_21 = r, so the product
# condition degenerates to r^2 < 1 and there is nothing left to substitute.
bad = 0
for _ in range(500):
    r = rng.uniform(0.1, 2.5)
    if abs(r - 1) < 1e-3:
        continue
    Lam = np.array([[0.0, r], [r, 0.0]])
    if feasible(Lam) != (r * r <= 1.0) or abs(Lam[0, 1] - Lam[1, 0]) > 0:
        bad += 1
print(f"H. Proposition 2: the normalized dyad is symmetric and degenerates to "
      f"r^2 <= 1 | violations {bad}")
fails += [bad]

# ---------------------------------------------------------------- I. Theorem 5
# An identically zero column makes the system feasible whatever the rest is,
# including configurations whose spectral radius is far above one.
bad = 0; above_one = 0
for _ in range(2000):
    n = int(rng.integers(3, 10))
    Lam = rng.uniform(0, 2.5, (n, n)); np.fill_diagonal(Lam, 0)
    i = int(rng.integers(0, n)); Lam[:, i] = 0.0        # nobody weights state i
    z = np.zeros(n); z[i] = 1.0
    if np.min((np.eye(n) - Lam) @ z) < -1e-12 or not feasible(Lam):
        bad += 1
    above_one += int(max(abs(np.linalg.eigvals(Lam))) > 1.0)
print(f"I. Theorem 5: an unweighted column suffices | violations {bad}; "
      f"{above_one}/2000 of these draws had rho(Lam) > 1")
fails += [0 if (bad == 0 and above_one > 0) else 1]

# --------------------------------------------------------------- J. Corollary 6
n = 32
Lam = np.full((n, n), 1.2); np.fill_diagonal(Lam, 0); Lam[:, n - 1] = 0.0
big_ok = feasible(Lam)
Lam3 = np.full((3, 3), 1.2); np.fill_diagonal(Lam3, 0)
small_ok = feasible(Lam3)
print(f"J. Corollary 6: 32 states, 31 hostile, one unweighted -> feasible {big_ok}; "
      f"3 states all weighted -> feasible {small_ok}")
fails += [0 if (big_ok and not small_ok) else 1]

# ------------------------------------- K. the three-state example of Section 7
# r = 1.5; states 1 and 2 spend their whole budget on each other; state 3, whom
# neither weights, splits its budget between them. q = (0.4, 0.4, 0.1), C = 0.1.
Lam = np.zeros((3, 3))
Lam[0, 1] = 1.5; Lam[1, 0] = 1.5; Lam[2, 0] = 0.75; Lam[2, 1] = 0.75
M = np.eye(3) - Lam
q = np.array([0.4, 0.4, 0.1]); x = np.array([0.4, 0.4, 0.2])
rows_ok = np.allclose(Lam.sum(1), 1.5)
col3_zero = np.allclose(Lam[:, 2], 0.0)
accepts = np.all(M @ x >= M @ q - 1e-12)
tight_12 = abs((M @ x)[0] - (M @ q)[0]) < 1e-12 and abs((M @ x)[1] - (M @ q)[1]) < 1e-12
gains_3 = (M @ x)[2] > (M @ q)[2] + 1e-12
rho = max(abs(np.linalg.eigvals(Lam)))
pair = Lam[0, 1] * Lam[1, 0]
dest = [i for i in range(3) if np.min(M @ np.eye(3)[i]) >= -1e-12]
sums_ok = abs(x.sum() - 1.0) < 1e-12 and abs(q.sum() - 0.9) < 1e-12
print(f"K. Section 7 example: rows sum to r {rows_ok}, column 3 zero {col3_zero}, "
      f"x=(0.4,0.4,0.2) accepted by all {accepts} (1 and 2 exactly indifferent "
      f"{tight_12}, 3 strictly gains {gains_3}); rho = {rho:.3f} > 1, pair product "
      f"= {pair:.2f} > 1; admissible unit destinations {[d + 1 for d in dest]}")
fails += [0 if (rows_ok and col3_zero and accepts and tight_12 and gains_3
                and rho > 1 and pair > 1 and dest == [2] and sums_ok) else 1]

# -------------------------- L. Section 8: enlargement with a weighted entrant
# Adding a state with any positive column weakly raises rho(Lam) and never lowers it.
drops = 0
for _ in range(1500):
    n = int(rng.integers(2, 8))
    Lam = rng.uniform(0, 1.2, (n, n)); np.fill_diagonal(Lam, 0)
    r0 = max(abs(np.linalg.eigvals(Lam)))
    big = np.zeros((n + 1, n + 1)); big[:n, :n] = Lam
    big[:n, n] = rng.uniform(0.01, 1.0, n)            # incumbents weight the entrant
    big[n, :n] = rng.uniform(0.0, 1.0, n)
    if max(abs(np.linalg.eigvals(big))) < r0 - 1e-9:
        drops += 1
print(f"L. Section 8: a salient entrant never lowers rho(Lam) | violations {drops}")
fails += [drops]

print("RESULT:", "all claims verified" if not any(fails) else "FAILURES PRESENT")
sys.exit(1 if any(fails) else 0)

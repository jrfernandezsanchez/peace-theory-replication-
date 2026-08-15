# The Network in Motion: Multilateral Ripeness, Early Settlement, and the
# Mildness Gap.
# A. Proposition 2 algebra: the positive root of h^2 - 2*eps*h - (1+2*eps)
#    is exactly h* = 1 + 2*eps (so h* = 1.0600 at eps = 0.03), and the rescue
#    condition holds with equality there.
# B. The feasible set in h for the leading topology is (0, 1] union (h*, inf),
#    confirmed against a linear-programming oracle.
# C. The rescue condition 2(1+h)*eps < h^2 - 1 coincides with the oracle for
#    h > 1 off the boundary.
# D. Early settlement: at h = 2.2, eps = 0.03 the pair is bilaterally
#    infeasible (cycle product 4.84) while the network is feasible.
# E. Perron bound: the spectral radius is at most the maximum row sum.
# F. Theorem 1 (ripeness): under the edgewise law of motion with ceilings,
#    every row sum crosses one by the stated date and the network is feasible.
import numpy as np, sys
from scipy.optimize import linprog

rng = np.random.default_rng(10)
fails = []

def lp_feasible(L, tol=1e-7):
    """Exists z (free in sign) with (I - L) z >= 0 and 1'z > 0, checked by
    maximizing 1'z over the box [-1,1]^n subject to (L - I) z <= 0. Signs are
    free because under spite a party accepts a material loss when the rival's
    larger loss more than pays for it."""
    n = L.shape[0]
    res = linprog(c=-np.ones(n), A_ub=(L - np.eye(n)), b_ub=np.zeros(n),
                  bounds=[(-1.0, 1.0)] * n, method="highs")
    return bool(res.status == 0 and -res.fun > tol)

def leading(h, eps):
    return np.array([[0.0, h, eps], [h, 0.0, eps], [0.0, 0.0, 0.0]])

# ------------------------------------------------------------ A. the root
badA = 0
for eps in np.linspace(0.001, 0.4, 200):
    root = eps + np.sqrt(eps * eps + 1.0 + 2.0 * eps)
    if abs(root - (1.0 + 2.0 * eps)) > 1e-13:
        badA += 1
    h = 1.0 + 2.0 * eps
    if abs((h * h - 1.0) - 2.0 * (1.0 + h) * eps) > 1e-12:
        badA += 1
okA = badA == 0 and abs((0.03 + np.sqrt(0.03 ** 2 + 1.06)) - 1.06) < 1e-13
print("A. h* = 1 + 2*eps exactly; boundary equality of the rescue condition; h*(0.03) = 1.0600 | %s"
      % ("ok" if okA else "FAIL"))
fails.append(0 if okA else 1)

# --------------------------------------------------------- B. the window
eps = 0.03; hstar = 1.06
badB = 0
for h in np.arange(0.50, 1.50001, 0.002):
    inside = (1.0 + 1e-9 < h <= hstar + 1e-9)
    if abs(h - 1.0) < 2e-3 or abs(h - hstar) < 2e-3:
        continue                        # skip the two boundaries themselves
    if lp_feasible(leading(h, eps)) == inside:
        badB += 1
spot = (lp_feasible(leading(1.000, eps)) and not lp_feasible(leading(1.030, eps))
        and not lp_feasible(leading(1.058, eps)) and lp_feasible(leading(1.062, eps)))
okB = badB == 0 and spot
print("B. feasible set in h is (0,1] union (1.06, inf) against the LP oracle | %s"
      % ("ok" if okB else "FAIL"))
fails.append(0 if okB else 1)

# ------------------------------------------------- C. rescue equivalence
badC = 0
for _ in range(400):
    h = rng.uniform(1.001, 3.0); e = rng.uniform(0.005, 0.4)
    lhs, rhs = 2.0 * (1.0 + h) * e, h * h - 1.0
    if abs(lhs - rhs) < 1e-3:
        continue
    if (lhs < rhs) != lp_feasible(leading(h, e)):
        badC += 1
print("C. rescue condition 2(1+h)eps < h^2 - 1 coincides with the oracle, 400 draws | mismatches %d" % badC)
fails.append(badC)

# ------------------------------------------------- D. early settlement
h, e = 2.2, 0.03
bilateral = lp_feasible(np.array([[0.0, h], [h, 0.0]]))
network = lp_feasible(leading(h, e))
budget = (h * h - 1.0) - 2.0 * (1.0 + h) * e
okD = (not bilateral) and network and budget > 0 and abs(h * h - 4.84) < 1e-12
print("D. h=2.2: cycle product 4.84, bilaterally infeasible, network feasible, budget %.3f > 0 | %s"
      % (budget, "ok" if okD else "FAIL"))
fails.append(0 if okD else 1)

# ---------------------------------------------------- E. Perron bound
badE = 0
for _ in range(200):
    n = rng.integers(2, 7)
    M = rng.uniform(0.0, 1.5, (n, n))
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    if rho > M.sum(axis=1).max() + 1e-9:
        badE += 1
print("E. Perron root <= maximum row sum, 200 random non-negative matrices | violations %d" % badE)
fails.append(badE)

# -------------------------------------------------------- F. ripeness
n = 3
b = rng.uniform(0.2, 0.5, (n, n)); np.fill_diagonal(b, 0.0)
rho_r, lamH = 0.3, 1.0
gbar, delta = 0.10, 0.20
beta = np.full(n, 0.05); ebar = np.full(n, 0.05)
ceil_g = gbar / (1.0 - np.exp(-delta))          # per-edge grievance ceiling
T = np.zeros(n)
for i in range(n):
    hard = sum(b[i, j] + rho_r * lamH + 0.3 * ceil_g for j in range(n) if j != i)
    T[i] = (hard - 1.0) / ((n - 1) * beta[i] * ebar[i])
Tstar = int(np.ceil(T.max()))
def L_at(t):
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i, j] = max(0.0, b[i, j] + rho_r * lamH + 0.3 * ceil_g
                              - beta[i] * ebar[i] * t)
    return L
L = L_at(Tstar)
okF = (L.sum(axis=1) < 1.0 + 1e-9).all() \
      and np.max(np.abs(np.linalg.eigvals(L))) < 1.0 and lp_feasible(L)
print("F. Theorem 1: worst-case rows cross 1 by T* = %d, spectral radius < 1, LP feasible | %s"
      % (Tstar, "ok" if okF else "FAIL"))
fails.append(0 if okF else 1)

if sum(fails) == 0:
    print("ALL CHECKS PASS")
    sys.exit(0)
print("FAILURES:", fails)
sys.exit(1)

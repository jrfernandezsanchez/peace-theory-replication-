Verification for "Rows and Columns: Relative Gains, Group Size and the Feasibility
of Multilateral Division: Snidal (1991) in a Bargaining Setting" (v1.0, SSRN
7273658).

verify_snidal.py checks the two Appendix lemmas (A1, the reduction of the division
problem to the cone condition, verified by running both formulations as separate
linear programs on random instances; A2, spectral sufficiency via the Neumann
series), Proposition 2 (the normalization forces symmetry in the dyad and
degenerates the product condition to r^2 <= 1), Proposition 3 (constant row sums r
give spectral radius exactly r, at every n and under non-uniform weights),
Proposition 4 (under uniform weights a mutually acceptable division exists iff
r <= 1, for n = 2 to 40, the case r = 1 included) together with its Farkas
certificate a = 1/(r-1), and the halving of the pairwise weight that leaves the
threshold untouched at every system size.

Section 5 is verified too: with a possibly n-dependent total the threshold becomes
r*sigma(n) <= 1, and Snidal's own recursion w_n <= ((n-1)/n)*w_{n-1} is shown to be
saturated by w_n = w_1/n, a path along which the total carried is constant at every
size, which is the point of that section.

Section 7: Theorem 5 (an unweighted column makes the system feasible whatever the
remaining weights, including configurations with spectral radius far above one),
Corollary 6, and the three-state example in full, with r = 1.5, lambda_12 =
lambda_21 = 1.5, lambda_31 = lambda_32 = 0.75, conflict allocation q = (0.4, 0.4,
0.1) and surplus C = 0.1, confirming that x = (0.4, 0.4, 0.2) is accepted by all
three, that states 1 and 2 are exactly indifferent while state 3 strictly gains, and
that the set of admissible unit destinations is exactly {3}. Section 8's enlargement
claim is checked as well: a salient entrant never lowers rho(Lambda).

Feasibility is decided throughout by a linear program on the cone, z free in sign as
in Lemma A1, not by the spectral or row-sum properties being tested.

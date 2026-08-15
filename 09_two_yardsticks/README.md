Verification for "The Two Yardsticks: Heterogeneous Valuation and the
Generalized Product Frontier" (Working Paper v1.0; add the SSRN number after
posting).

verify_two_yardsticks.py checks Theorem 3 (the symmetric pivot: both exchange
rates equal one at the even split for every ordered pair of six test indices
and three symmetric conflict points, to machine precision), Theorem 1 (the
generalized frontier lambda_U <= Phi(lambda_R) against a grid-search oracle on
about 2,900 randomized heterogeneous instances, zero disagreements off the
frontier band), Corollary 1.1 (feasibility if and only if the product is at
most Delta(x*)), Proposition 2 (affine agreement of the yardsticks forces
Delta identically one), the asymmetric benchmark q = (0.20, 0.45) (the
threshold moves strictly above one for some ordered pairs and strictly below
for others while every common-index pair stays exactly at one), Proposition 4
(the threshold moves linearly in a smooth divergence: |Phi(1) - 1| / eps
stabilizes as eps shrinks), and the Section 5 finding that the product
elicited at the conflict allocation predicts feasibility better than the true
product on heterogeneous instances.

Run: python3 verify_two_yardsticks.py
Prints one line per check and ALL CHECKS PASS, exit code 0. Requires numpy.


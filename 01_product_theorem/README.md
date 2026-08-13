Verification for "Spite, Divisibility and the Existence of Peace: A General Product
Theorem" (SSRN 7248818).

verify_product_theorem.py checks both branches of Theorem 1 against a grid oracle
over pure divisions (non-empty range iff lambda_R*lambda_U <= 1 when conflict is ex
post inefficient, iff >= 1 when conflict dominates every division), the knife-edge
case, Corollary 1.1 (the frontier is strictly competitive), the closed form of
Corollary 1.3, the lottery width of Theorem 2(b), the mutual-impoverishment
counterexample of the remark to Theorem 3, the four partials and the sign law of
Theorem 4, and the factorization of Proposition 3.

Knife edge: draws with |lambda_R*lambda_U - 1| < 0.05 are excluded from the two
grid tests, since at the frontier the acceptable set is a single point and no finite
mesh can be expected to land on it. That case is checked separately in part B'.


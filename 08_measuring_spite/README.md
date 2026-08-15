Verification for "Measuring Spite: Invariance, Elicitation, and What Survives
Curvature" (Working Paper v1.3; add the SSRN number after posting).

verify_measuring_spite.py checks the worked feasible intervals of Section 4
under five indices to four decimals (linear, square root, x^5, the loss-averse
kink and the logarithmic index), the twelve-index invariance battery (about
14,400 randomized configurations; a non-empty acceptable range if and only if
lambda_R * lambda_U <= 1 under any common index, knife-edge band excluded),
Theorem 2 (the elicited matrix is a diagonal similarity of the true matrix:
spectrum, two-cycle products and zero pattern preserved), the Section 5
non-identification of row sums at an unequal profile together with their
recovery at the equal profile to truncation error, the necessity of a common
index (disagreement counts strictly rising with index divergence), the
one-directional failure mode of a jump index (empty ranges despite a product
below one, never the reverse), and the finite-stake bias (raw product errors
roughly halving with the stake, the two-stake extrapolation removing most of
the remainder).

The grid oracle refines locally around the best candidate before declaring a
configuration infeasible, so narrow acceptable intervals are not lost to the
mesh.

Run: python3 verify_measuring_spite.py
Prints one line per check and ALL CHECKS PASS, exit code 0. Requires numpy.


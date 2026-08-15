Verification for "The Manufacture of Grievance: Domestic Politics and the
Supply of Spite" (Working Paper v1.0; add the SSRN number after posting).

verify_manufacture.py checks Proposition 1 (the cliff: the analytic best
response, frontier-defending while the frontier is reachable and the peace
dividend worth defending, the rally corner otherwise, attains the brute-force
grid optimum on 2,000 randomized draws), the Theorem 2 benchmark at kappa =
0.8, a = 0.6, r = d = 1, w = 0.5 (iterated best response from joint restraint
reaches g = (0.75, 0) with lambda = (1.25, 0.80) and product exactly one;
from joint mobilization it stays at the hawk corner with product 1.96, locked
because kappa*(kappa + a) = 1.12 > 1), the fixed-point property of both
regimes (no profitable unilateral deviation on the grid), the slide of joint
restraint itself to the hawk corner at w = 0.95, and the monotone growth of
the hawk basin in the rally weight w.

Run: python3 verify_manufacture.py
Prints one line per check and ALL CHECKS PASS, exit code 0. Requires numpy.


Verification for "The Network in Motion: Multilateral Ripeness, Early
Settlement, and the Mildness Gap" (Working Paper v1.0; add the SSRN number
after posting).

verify_network_in_motion.py checks the Proposition 2 algebra (the positive
root of h^2 - 2*eps*h - (1 + 2*eps) is exactly h* = 1 + 2*eps, since the
discriminant is a perfect square; hence h* = 1.0600 at eps = 0.03, and the
rescue condition holds with equality at h*), the feasible set of the leading
topology in h against a linear-programming oracle (feasible on (0, 1], an
infeasibility window on (1, h*], feasible again above h*), the equivalence of
the rescue condition 2(1+h)*eps < h^2 - 1 with the oracle for h > 1 (400
randomized draws off the boundary), the early-settlement example (h = 2.2:
cycle product 4.84, bilaterally infeasible, network feasible with slack), the
Perron bound (spectral radius at most the maximum row sum), and Theorem 1
(under the edgewise law of motion with worst-case ceilings all row sums cross
one by the stated date and the settlement program is feasible there).

The oracle maximizes total surplus subject to every transformed participation
constraint, with allocations free in sign: under spite a party accepts a
material loss when the rival's larger loss more than pays for it, which is
exactly the mechanism of the infeasibility window.

Run: python3 verify_network_in_motion.py
Prints one line per check and ALL CHECKS PASS, exit code 0. Requires numpy and
scipy.


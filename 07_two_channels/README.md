Verification for "The Two Channels: Transitional Justice, Memory Policy and the
Reconciliation Frontier" (Working Paper v1.0; add the SSRN number after posting).

verify_two_channels.py checks Proposition 1 (the total effect of any memory
policy on the long-run spite product decomposes exactly into the belief channel
and the flow channel: the analytic derivative matches numerical differentiation
on 500 randomized parameter draws), Proposition 2 (the memory-free floor bounds
the product from below, and a floor at or above one defeats every flow
instrument over a full grid of grievance-flow settings), the symmetric critical
forgetting rate delta* = alpha*gbar/(1 - kappa_inf) (reconciliation on one side
of it, not on the other, 200 draws), the descending acknowledgment ceiling
G*(t) between its stated endpoints, and the grievance law of motion (Euler
simulation against the closed form; steady state gbar/delta).

Run: python3 verify_two_channels.py
Prints one line per check and ALL CHECKS PASS, exit code 0. Requires numpy.


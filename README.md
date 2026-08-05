# Replication package — A General Theory of Peace, Reconciliation and Memory

Author: Jaime Ramón Fernández Sánchez (jaimewale@hotmail.com)
Working-paper series, 2026. One folder per paper with computational claims.

## What this is

Every mathematical claim in the series was verified computationally before being
asserted. This package contains self-contained verification scripts, one folder
per paper, each with a fixed random seed and explicit pass/fail assertions
(exit code 0 = all claims verified). The scripts implement the models exactly as
stated in the papers and test the theorems against independent oracles
(grid/LP/backward-induction), not against the formulas being tested.

## Contents

- 01_product_theorem       — Paper I: two-branch product theorem, constant-sum frontier, deterrence inversion, mutual impoverishment beyond the frontier
- 02_endogenous_frontier   — Paper II: ripeness bound; belief-trap thresholds (silence, grudge)
- 03_reconciliation_frontier — Paper III: critical claim/forgetting ratio, closing window, symmetric vs targeted enforcement signs
- 04_unresented_party      — spectral sufficiency, unresented-column theorem, Farkas certificate, exact absorption condition
- 05_rows_and_columns      — Snidal: row sums = spectral radius, n-invariance of feasibility
- 06_fehr_schmidt          — piecewise-spite equivalence; group-size-free marginal incentive
- 07_two_channels          — opposite-sign channels illustration
- 08_measuring_spite       — diagonal-similarity identification: what is and is not invariant
- 09_two_yardsticks        — heterogeneous-index frontier vs oracle, symmetric pivot, elicited vs true product
- 10_network_in_motion     — multilateral ripeness bound, settlement before ripeness, mildness boundary
- 11_manufacture_of_grievance — cliff structure, two stable regimes, hawk basin in w (implements the v1.1 correction)
- 12_whose_utilities_count — the two Paretos part company; scorched-earth refusal
- 13_thucydides_coordinate — original scripts of the final paper: exact one-step condition (0/17,577-class batteries), general path characterization, patient limit, width identity (symbolic), gradualism, endogenous fear

## How to run

    pip install numpy scipy sympy
    bash run_all.sh          # runs every script; prints PASS/FAIL per paper

Python 3.9+. Total runtime a few minutes on a laptop.

## Notes on reproducibility and honesty

Scripts are seeded, so each run reproduces its own numbers exactly. Where a paper
cites a specific battery count (e.g. "0 mismatches in 17,577 economies"), folder 13
contains the original scripts; the other folders contain reconstructions of the
original verification designs — the zero-mismatch claims are exact theorems and
replicate at any seed, while percentage-type figures (e.g. classifier accuracies
in folder 09) are seed-dependent point estimates that reproduce approximately.
The series' practice of reporting its own falsifications is preserved here: the
comments in folders 11 and 13 document the two claims that verification killed
(v1.0's supermodularity framing of the grievance game; the per-period-speed form
of the Thucydides conjecture) and what replaced them.

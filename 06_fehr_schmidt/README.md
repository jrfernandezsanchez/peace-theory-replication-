# 06_fehr_schmidt

Mathematical checks for *Fehr–Schmidt as a Piecewise Spite Model: Local Incentives, Reference Weights, and the Limits of Group-Size Invariance* (v1.3, October 2026). SSRN: https://ssrn.com/abstract=7273438

```
pip install numpy
python verify_math.py
```

The script writes `math_checks.json` and exits with code 0 only if every check passes. Seed fixed at 20261003; runtime about 15 seconds.

| Check | Paper statement |
|---|---|
| `lemma1_piecewise_reduction` | Lemma 1: exact linear relative-gains representation on each strict payoff-ordering region |
| `lemma1_row_sum_and_positive_bound` | Signed row sum k_i; positive entries sum to αa/c < 1 |
| `lemma1_no_positive_cycle` | Positive edges point upward in payoffs; reciprocal products are nonpositive |
| `prop2_derivative_vs_numerical` | Proposition 2 against central differences, 500 profiles at n = 4, 10, 40, 200, 1,000 (bound in the text: 1.3 × 10⁻⁸) |
| `prop2_one_sided_derivatives_at_ties` | Left and right derivatives at ties |
| `cor3_replication_invariance` | Corollary 3: k-fold replication preserves every own-contribution comparison |
| `symmetric_equilibrium_condition` | Common positive contribution is a best response iff β ≥ 1 − m; zero profile always an equilibrium |
| `symmetric_equilibrium_probabilities` | 0.9⁴ = 0.6561 and 0.9⁴⁰ ≈ 0.0148 |
| `population_sampling_variance` | Fraction below g_i has variance F(1 − F)/(n − 1) under independent draws |
| `prop4_weighted_derivative` | Proposition 4 with fixed nonuniform weights |
| `prop4_entrant_change` | ΔD_i formula for an entrant with weight η |
| `sec8_general_common_benefit` | x_j − x_i = g_i − g_j under any common benefit B |

These are implementation checks of the stated identities, not evidence about behavior.

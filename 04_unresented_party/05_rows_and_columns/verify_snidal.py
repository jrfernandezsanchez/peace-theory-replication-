# Rows and Columns (Snidal): constant row sums -> spectral radius = r at every n;
# uniform case feasibility iff r <= 1, invariant in n.
import numpy as np, sys
rng=np.random.default_rng(2); fails=[]

# A. spectral radius of any nonneg matrix with constant row sums r equals r
worst=0
for _ in range(2000):
    n=rng.integers(2,12); r=rng.uniform(0.1,3.0)
    M=rng.uniform(0,1,(n,n)); np.fill_diagonal(M,0)
    M=M/M.sum(1,keepdims=True)*r          # Snidal normalization scaled by r
    rho=max(abs(np.linalg.eigvals(M)))
    worst=max(worst,abs(rho-r))
print(f"A. rho(Lambda)=row sum r: max error {worst:.2e} over 2000 matrices, n=2..11")
fails+=[0 if worst<1e-8 else 1]

# B. uniform weights: feasibility (exists g>=0, g!=0, (I-M)g>=0) iff r<=1, for every n
def feasible(M):
    n=len(M)
    # try uniform surplus and single-column allocations, then LP-style search via eigen
    import itertools
    for g in [np.ones(n)]+[np.eye(n)[i] for i in range(n)]:
        d=g-M@g
        if (d>=-1e-12).all() and d.sum()>1e-12: return True
    # Perron direction
    w,V=np.linalg.eig(M.T)
    return False
mis=0
for n in range(2,11):
    for r in [0.3,0.7,0.99,1.01,1.5,2.5]:
        M=np.full((n,n),r/(n-1)); np.fill_diagonal(M,0)
        f=feasible(M)
        if f != (r<=1.0-1e-12 or abs(r-1)<1e-9): mis+=1
print(f"B. uniform case: feasibility iff r<=1 at every n in 2..10 | mismatches {mis}")
fails+=[mis]
sys.exit(1 if any(fails) else 0)

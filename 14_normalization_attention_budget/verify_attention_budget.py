# Normalization as an Attention Budget (v1.1): Prop 2 (FS marginal condition,
# group size absent), Prop 4 (Snidal inertness), Prop 5 (weighted marginal
# condition under general W; zero-column individual is behaviourally inert).
import random, sys
import numpy as np
random.seed(14); fails=[]
def FS_w(x,i,W,al,be):  # weighted FS-type objective with general attention row W[i]
    n=len(x)
    return x[i] - al*sum(W[i][j]*max(x[j]-x[i],0) for j in range(n) if j!=i) \
                - be*sum(W[i][j]*max(x[i]-x[j],0) for j in range(n) if j!=i)
# A. Prop 2 (uniform W): dU/dg = (m-1) - a*phi + b*psi, no n  (cross-check of folder 06)
worst=0; eps=1e-7; tested=0
while tested<1500:
    n=random.randint(4,60); m=random.uniform(0.2,0.9); al=random.uniform(0,1.5); be=random.uniform(0,min(al,0.99))
    c=[random.random() for _ in range(n)]; i=random.randrange(n)
    if min(abs(c[j]-c[i]) for j in range(n) if j!=i)<10*eps: continue
    W=[[1.0/(n-1)]*n for _ in range(n)]
    def payoff(ci):
        cc=c[:]; cc[i]=ci; tot=sum(cc); x=[1-cc[k]+m*tot for k in range(n)]
        return FS_w(x,i,W,al,be)
    num=(payoff(c[i]+eps)-payoff(c[i]-eps))/(2*eps)
    phi=sum(1 for j in range(n) if j!=i and c[j]<c[i])/(n-1)
    psi=sum(1 for j in range(n) if j!=i and c[j]>c[i])/(n-1)
    worst=max(worst,abs(num-((m-1)-al*phi+be*psi))); tested+=1
print(f"A. Prop 2 uniform: max |numeric-formula| = {worst:.2e} (n=4..60)"); fails+=[0 if worst<1e-4 else 1]
# B. Prop 4: constant row sums r -> rho = r at every n (cross-check of folder 05)
worst=0
rng=np.random.default_rng(14)
for _ in range(1000):
    n=int(rng.integers(2,12)); r=rng.uniform(0.1,3.0)
    M=rng.uniform(0,1,(n,n)); np.fill_diagonal(M,0); M=M/M.sum(1,keepdims=True)*r
    worst=max(worst,abs(max(abs(np.linalg.eigvals(M)))-r))
print(f"B. Prop 4 inertness: max |rho - r| = {worst:.2e}"); fails+=[0 if worst<1e-8 else 1]
# C. Prop 5: weighted marginal condition; and zero-column individual is inert
worst=0; inert_viol=0; tested=0
while tested<1000:
    n=random.randint(4,20); m=random.uniform(0.2,0.9); al=random.uniform(0,1.5); be=random.uniform(0,min(al,0.99))
    Wm=[[random.random() if j!=i2 else 0.0 for j in range(n)] for i2 in range(n)]
    z=random.randrange(n)                          # nobody attends to z
    for i2 in range(n): Wm[i2][z]=0.0
    c=[random.random() for _ in range(n)]; i=random.randrange(n)
    if i==z or min(abs(c[j]-c[i]) for j in range(n) if j!=i)<1e-6: continue
    def payoff(ci,cvec):
        cc=cvec[:]; cc[i]=ci; tot=sum(cc); x=[1-cc[k]+m*tot for k in range(n)]
        return FS_w(x,i,Wm,al,be)
    num=(payoff(c[i]+1e-7,c)-payoff(c[i]-1e-7,c))/(2e-7)
    pred=(m-1)-al*sum(Wm[i][j] for j in range(n) if j!=i and c[j]<c[i]) \
              +be*sum(Wm[i][j] for j in range(n) if j!=i and c[j]>c[i])
    worst=max(worst,abs(num-pred))
    # inertness: perturb z's contribution; i's marginal incentive must not move
    c2=c[:]; c2[z]=min(1.0,c2[z]+0.2)
    num2=(payoff(c[i]+1e-7,c2)-payoff(c[i]-1e-7,c2))/(2e-7)
    # note: changing c_z moves the public good equally for all, cancelling from
    # comparisons; with W[:,z]=0 nothing else moves either
    if abs(num2-num)>1e-5: inert_viol+=1
    tested+=1
print(f"C. Prop 5 weighted condition: max err {worst:.2e} | zero-column inertness violations {inert_viol}/1000")
fails+=[0 if worst<1e-4 else 1, inert_viol]
sys.exit(1 if any(fails) else 0)

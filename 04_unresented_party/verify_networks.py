# The Unresented Party: cone condition, spectral sufficiency (not necessity),
# unresented-column theorem, Farkas certificate, spite-financed absorption condition.
import numpy as np, sys
from scipy.optimize import linprog
rng=np.random.default_rng(5); fails=[]
def feasible(L):
    # exists g>=0, sum g = 1, (I-L)g >= 0 ?
    n=len(L)
    res=linprog(c=np.zeros(n),A_ub=-(np.eye(n)-L),b_ub=np.zeros(n),
                A_eq=np.ones((1,n)),b_eq=[1.0],bounds=[(0,None)]*n,method="highs")
    return res.status==0
# A. spectral sufficiency: rho(L)<1 -> feasible (2000 random networks)
bad=0
for _ in range(2000):
    n=int(rng.integers(2,8)); L=rng.uniform(0,1,(n,n)); np.fill_diagonal(L,0)
    L*=rng.uniform(0.1,0.95)/max(1e-12,max(abs(np.linalg.eigvals(L))))
    if not feasible(L): bad+=1
print(f"A. rho<1 implies feasible: violations {bad}/2000"); fails+=[bad]
# B. spectral NOT necessary: unresented column rescues arbitrarily hostile networks
bad=0
for _ in range(1000):
    n=int(rng.integers(3,8)); L=rng.uniform(1.0,4.0,(n,n)); np.fill_diagonal(L,0)
    j=int(rng.integers(0,n)); L[:,j]=0.0   # nobody resents j
    rho=max(abs(np.linalg.eigvals(L)))
    if rho<=1 or not feasible(L): bad+=1
print(f"B. unresented column -> feasible despite rho>1: violations {bad}/1000"); fails+=[bad]
# C. Farkas certificate for infeasibility: t>=0 with (L^T - I)t >= 1 exists iff infeasible
mism=0
for _ in range(1500):
    n=int(rng.integers(2,7)); L=rng.uniform(0,2.2,(n,n)); np.fill_diagonal(L,0)
    f=feasible(L)
    res=linprog(c=np.zeros(n),A_ub=-(L.T-np.eye(n)),b_ub=-np.ones(n),bounds=[(0,None)]*n,method="highs")
    cert=(res.status==0)
    if f==cert: mism+=1   # exactly one of the two must hold
print(f"C. Farkas duality (feasible XOR certificate): violations {mism}/1500"); fails+=[mism]
# D. spite-financed absorption: principals 1,2 (product>1) + absorber 3 resented
#    (e1,e2). Settlement x>=0, sum=1, U_i >= war_i with war shares w,1-w (c->0).
#    Exact condition: (1+l21)e1 + (1+l12)e2 < l12*l21 - 1.
mism=0; n_t=0
for _ in range(4000):
    l12=rng.uniform(0.2,2.5); l21=rng.uniform(0.2,2.5)
    if l12*l21<=1.02: continue
    e1=rng.uniform(0,1.2); e2=rng.uniform(0,1.2)
    lhs=(1+l21)*e1+(1+l12)*e2; rhs=l12*l21-1
    if abs(lhs-rhs)<0.06: continue
    w=rng.uniform(0.2,0.8); c=0.01
    A1=w*(1-c)-l12*(1-w)*(1-c); A2=(1-w)*(1-c)-l21*w*(1-c)
    # exists x: x1-l12*x2-e1*x3>=A1 ; x2-l21*x1-e2*x3>=A2 ; x>=0 ; sum=1
    res=linprog(np.zeros(3),
                A_ub=np.array([[-1,l12,e1],[l21,-1,e2]]),b_ub=np.array([-A1,-A2]),
                A_eq=np.ones((1,3)),b_eq=[1.0],bounds=[(0,1)]*3,method="highs")
    n_t+=1
    if (res.status==0)!=(lhs<rhs): mism+=1
print(f"D. absorption condition (1+l21)e1+(1+l12)e2 < l12*l21-1: {n_t} triads | mismatches {mism}")
fails+=[mism]
sys.exit(1 if any(fails) else 0)

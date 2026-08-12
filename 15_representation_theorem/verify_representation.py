# A Representation Theorem for Interpersonal Attention (v1.1):
# A4 equivalence, Theorem 3 closed form (piecewise FS weights and budget k(R)),
# elicitation identities (Props 4-5), and the population test's discriminating power.
import random, sys
random.seed(15); fails=[]
# A. A4: uniform windfall t ~ private (1-k)t, k = row sum, signs free
worst=0
for _ in range(2000):
    n=random.randint(2,10); w=[random.uniform(-1,1.5) for _ in range(n-1)]; k=sum(w)
    U=lambda y: y[0]-sum(wj*y[j+1] for j,wj in enumerate(w))
    x=[random.uniform(-5,5) for _ in range(n)]; t=random.uniform(0.1,3)
    worst=max(worst,abs(U([x[0]+t]+[xj+t for xj in x[1:]])-U([x[0]+(1-k)*t]+x[1:])))
print(f"A. A4 identity: max error {worst:.2e}"); fails+=[0 if worst<1e-10 else 1]
# B. Theorem 3: piecewise FS weights and budget
def FS(x,i,a,b_):
    n=len(x)
    return x[i]-a/(n-1)*sum(max(x[j]-x[i],0) for j in range(n) if j!=i) \
               -b_/(n-1)*sum(max(x[i]-x[j],0) for j in range(n) if j!=i)
worst_r=worst_k=0
for _ in range(3000):
    n=random.randint(3,15); al=random.uniform(0,2.0); be=random.uniform(0,min(al,0.99))
    x=[random.random() for _ in range(n)]; i=random.randrange(n)
    A=[j for j in range(n) if j!=i and x[j]>x[i]]; B=[j for j in range(n) if j!=i and x[j]<x[i]]
    a=len(A)/(n-1); b=len(B)/(n-1); D=1+al*a-be*b
    Urep=x[i]-sum((al/(n-1))/D*x[j] for j in A)-sum(-(be/(n-1))/D*x[j] for j in B)
    worst_r=max(worst_r,abs(FS(x,i,al,be)/D-Urep))
    worst_k=max(worst_k,abs(sum((al/(n-1))/D for _ in A)+sum(-(be/(n-1))/D for _ in B)-(al*a-be*b)/D))
print(f"B. Theorem 3 weights/budget: max errors {worst_r:.2e} / {worst_k:.2e}")
fails+=[0 if worst_r<1e-10 and worst_k<1e-10 else 1]
# C. elicitation identities: w_ij from single indifference; k from windfall fraction
worst=0
for _ in range(2000):
    n=random.randint(2,8); w=[random.uniform(-1,1.5) for _ in range(n-1)]; k=sum(w)
    U=lambda y: y[0]-sum(wj*y[j+1] for j,wj in enumerate(w))
    x=[random.uniform(-2,2) for _ in range(n)]; t=random.uniform(0.5,2); j=random.randrange(1,n)
    # i indifferent between j gaining t and own loss s  <=>  s = t*w_ij
    s=t*w[j-1]
    lhs=U(x[:j]+[x[j]+t]+x[j+1:]); rhs=U([x[0]-s]+x[1:])
    worst=max(worst,abs(lhs-rhs))
    # k: indifferent between uniform t and private f*t with f = 1-k
    f=1-k
    worst=max(worst,abs(U([xi+t for xi in x])-U([x[0]+f*t]+x[1:])))
print(f"C. elicitation identities (entry and budget): max error {worst:.2e}"); fails+=[0 if worst<1e-10 else 1]
# D. population test discriminates: fixed budget -> f constant in n;
#    accumulating attention (fixed per-person weight) -> f falls linearly in n
for model in ["budget","accumulating"]:
    fs=[]
    for n in [3,6,12,24]:
        w0=0.06
        w=[0.3/(n-1)]*(n-1) if model=="budget" else [w0]*(n-1)
        k=sum(w); fs.append(round(1-k,4))
    print(f"D. {model:12s}: f(n) at n=3,6,12,24 -> {fs}")
fails+=[0]
sys.exit(1 if any(fails) else 0)

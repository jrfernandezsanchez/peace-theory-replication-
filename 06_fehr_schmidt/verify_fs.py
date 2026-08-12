# Fehr–Schmidt as piecewise spite: exact linear relative-gains form within a fixed
# payoff ordering; public-goods marginal incentive (m-1) - a*phi + b*psi, n absent.
import random, sys
random.seed(3); fails=[]
def FS(x,i,a,b):
    n=len(x)
    return x[i] - a/(n-1)*sum(max(x[j]-x[i],0) for j in range(n) if j!=i) \
                - b/(n-1)*sum(max(x[i]-x[j],0) for j in range(n) if j!=i)
# A. piecewise linear relative-gains form
worst=0
for _ in range(3000):
    n=random.randint(3,12); a=random.uniform(0,1.5); b=random.uniform(0,min(a,0.99))
    x=[random.random() for _ in range(n)]
    i=random.randrange(n)
    lam=[0.0]*n; own=1.0
    for j in range(n):
        if j==i: continue
        if x[j]>x[i]: lam[j]= a/(n-1); own+= a/(n-1)
        else:         lam[j]=-b/(n-1); own-= b/(n-1)
    lin=own*x[i]-sum(lam[j]*x[j] for j in range(n) if j!=i)
    worst=max(worst,abs(lin-FS(x,i,a,b)))
print(f"A. FS = piecewise linear relative-gains form: max error {worst:.2e} (3000 profiles, n=3..12)")
fails+=[0 if worst<1e-10 else 1]
# B. public goods: dU_i/dc_i = (m-1) - a*phi_i + b*psi_i  — group size absent
worst=0; eps=1e-7; tested=0
while tested<2000:
    n=random.randint(4,80); m=random.uniform(0.2,0.9); a=random.uniform(0,1.5); b=random.uniform(0,min(a,0.99))
    c=[random.random() for _ in range(n)]; i=random.randrange(n)
    if min(abs(c[j]-c[i]) for j in range(n) if j!=i)<10*eps: continue  # keep ordering fixed
    def payoff(ci):
        cc=c[:]; cc[i]=ci
        tot=sum(cc); x=[1-cc[k]+m*tot for k in range(n)]
        return FS(x,i,a,b)
    num=(payoff(c[i]+eps)-payoff(c[i]-eps))/(2*eps)
    phi=sum(1 for j in range(n) if j!=i and c[j]<c[i])/(n-1)
    psi=sum(1 for j in range(n) if j!=i and c[j]>c[i])/(n-1)
    pred=(m-1)-a*phi+b*psi
    worst=max(worst,abs(num-pred)); tested+=1
print(f"B. marginal incentive (m-1)-a*phi+b*psi, n=4..80: max |numeric-formula| = {worst:.2e}")
fails+=[0 if worst<1e-4 else 1]
sys.exit(1 if any(fails) else 0)

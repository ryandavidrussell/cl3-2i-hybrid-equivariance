"""
hybrid_trainable.py  --  E2/E3 negative controls (leakage & eigenvector collapse)
plus intermediate honest trainable version of hybrid experiment.
NumPy only. Run: python hybrid_trainable.py
"""
import numpy as np
from cl3_2i_gate import ICOSIANS, N, GRADE, GP
from vec_cl3_2i import gp_batch, reverse_batch, scalar_part

V=np.array(ICOSIANS); NV=len(V)
Gm=V@V.T; np.fill_diagonal(Gm,-2); MX=Gm.max()
ADJ=(np.abs(Gm-MX)<1e-6).astype(float)
DEG=ADJ.sum(1); Dinv=np.diag(1/np.sqrt(DEG)); Snorm=Dinv@ADJ@Dinv

def quat_to_mv(q):
    w,x,y,z=q; m=np.zeros(N); m[0]=w; m[6]=x; m[5]=-y; m[4]=z; return m
VERT_MV=np.array([quat_to_mv(q) for q in V])
def vfeat(x): return VERT_MV*(1.0+x[:,None])

def target(x):
    M=vfeat(x); nbr=ADJ@M
    p1=scalar_part(gp_batch(reverse_batch(M),nbr))
    nbr2=ADJ@gp_batch(M,M)
    p2=scalar_part(gp_batch(reverse_batch(M),nbr2))
    return np.tanh(p1)+0.3*p2

class MLP:
    def __init__(self,din,H,seed):
        r=np.random.default_rng(seed)
        self.W1=r.normal(size=(din,H))*0.5; self.b1=np.zeros(H)
        self.W2=r.normal(size=(H,1))*0.5;  self.b2=0.0
    def __call__(self,F):
        self.z=F@self.W1+self.b1; self.a=np.tanh(self.z)
        return (self.a@self.W2+self.b2).ravel()
    def grad(self,F,y):
        n=len(y); pred=self(F); r=(pred-y)[:,None]/n
        gW2=self.a.T@r; gb2=r.sum(); da=r@self.W2.T*(1-self.a**2)
        return pred,F.T@da,da.sum(0),gW2,gb2
    def step(self,gW1,gb1,gW2,gb2,lr):
        self.W1-=lr*gW1; self.b1-=lr*gb1; self.W2-=lr*gW2; self.b2-=lr*gb2

def feats_C(x):
    M=vfeat(x); sp=scalar_part(gp_batch(reverse_batch(M),M))
    return np.stack([x,x**2,sp,scalar_part(M)],axis=1)
def feats_G(x):
    Sx=Snorm@x;S2=Snorm@Sx;S3=Snorm@S2; return np.stack([x,Sx,S2,S3],axis=1)
def feats_H(x):
    M=vfeat(x); sp_self=scalar_part(gp_batch(reverse_batch(M),M))
    Sx=Snorm@x;S2=Snorm@Sx; return np.stack([x,Sx,S2,sp_self],axis=1)

def train_eval(feat_fn,H=16,epochs=4000,lr=0.05,seed=0):
    xs=[np.random.default_rng(3000+s).normal(size=NV) for s in range(50)]
    ys=[target(x) for x in xs]
    Ftr=np.vstack([feat_fn(x) for x in xs[:35]]); Ytr=np.concatenate(ys[:35])
    Fte=np.vstack([feat_fn(x) for x in xs[35:]]); Yte=np.concatenate(ys[35:])
    mu=Ftr.mean(0); sd=Ftr.std(0)+1e-9; Ftr=(Ftr-mu)/sd; Fte=(Fte-mu)/sd
    net=MLP(Ftr.shape[1],H,seed)
    for _ in range(epochs):
        _,gW1,gb1,gW2,gb2=net.grad(Ftr,Ytr); net.step(gW1,gb1,gW2,gb2,lr)
    return np.mean((net(Ftr)-Ytr)**2), np.mean((net(Fte)-Yte)**2), np.mean(Yte**2)

def run():
    print("="*70); print("E2/E3 + TRAINABLE HYBRID EXPERIMENT"); print("="*70)
    res={}
    for nm,fn in [("Clifford-only",feats_C),("Graph-only",feats_G),("HYBRID",feats_H)]:
        tr,te,var=train_eval(fn); res[nm.strip()]=te
        print(f" [{nm:14s}] train {tr:.4e} | test {te:.4e} | rel {te/var:.3f}")
    best=min(res["Clifford-only"],res["Graph-only"]); h=res["HYBRID"]
    print(f" improvement: {best/max(h,1e-300):.2f}x")
    print("="*70)

if __name__ == "__main__":
    run()

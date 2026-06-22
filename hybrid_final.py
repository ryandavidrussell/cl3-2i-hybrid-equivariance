"""
hybrid_final.py  --  E4: honest hybrid on synthetic 600-cell.
Collapse-free target: routes independent per-vertex signal through
geometric product AND graph neighbor sum. Three matched-capacity
trained MLPs (Clifford-only / Graph-only / Hybrid).
NumPy only. Run: python hybrid_final.py
"""
import numpy as np
from cl3_2i_gate import ICOSIANS, N
from vec_cl3_2i import gp_batch, reverse_batch, scalar_part

V=np.array(ICOSIANS); NV=len(V)
Gm=V@V.T; np.fill_diagonal(Gm,-2); MX=Gm.max()
ADJ=(np.abs(Gm-MX)<1e-6).astype(float)
DEG=ADJ.sum(1); Dinv=np.diag(1/np.sqrt(DEG)); Snorm=Dinv@ADJ@Dinv

def quat_to_mv(q):
    w,x,y,z=q; m=np.zeros(N); m[0]=w; m[6]=x; m[5]=-y; m[4]=z; return m
VERT_MV=np.array([quat_to_mv(q) for q in V])

def vfeat(x):
    out=VERT_MV.copy(); out[:,0]+=x; return out

def target(x):
    M=vfeat(x)
    return np.tanh(scalar_part(gp_batch(reverse_batch(M), ADJ@M)))

class MLP:
    def __init__(self,din,H,seed):
        r=np.random.default_rng(seed)
        self.W1=r.normal(size=(din,H))*0.5; self.b1=np.zeros(H)
        self.W2=r.normal(size=(H,1))*0.5;  self.b2=0.0
    def __call__(self,F):
        self.a=np.tanh(F@self.W1+self.b1); return (self.a@self.W2+self.b2).ravel()
    def grad(self,F,y):
        n=len(y); pred=self(F); r=(pred-y)[:,None]/n
        gW2=self.a.T@r; gb2=r.sum(); da=r@self.W2.T*(1-self.a**2)
        return pred, F.T@da, da.sum(0), gW2, gb2
    def step(self,g,lr):
        self.W1-=lr*g[0]; self.b1-=lr*g[1]; self.W2-=lr*g[2]; self.b2-=lr*g[3]

def feats_C(x):
    M=vfeat(x); return np.stack([x,x**2,scalar_part(gp_batch(reverse_batch(M),M)),M[:,0]],axis=1)
def feats_G(x):
    Sx=Snorm@x; S2=Snorm@Sx; S3=Snorm@S2; return np.stack([x,Sx,S2,S3],axis=1)
def feats_H(x):
    M=vfeat(x); nbr=ADJ@M
    cross=scalar_part(gp_batch(reverse_batch(M),nbr))
    return np.stack([x,Snorm@x,scalar_part(gp_batch(reverse_batch(M),M)),cross],axis=1)

def train_eval(fn,H=16,epochs=6000,lr=0.05,seed=0):
    xs=[np.random.default_rng(4000+s).normal(size=NV) for s in range(50)]
    ys=[target(x) for x in xs]
    Ftr=np.vstack([fn(x) for x in xs[:35]]); Ytr=np.concatenate(ys[:35])
    Fte=np.vstack([fn(x) for x in xs[35:]]); Yte=np.concatenate(ys[35:])
    mu=Ftr.mean(0); sd=Ftr.std(0)+1e-9; Ftr=(Ftr-mu)/sd; Fte=(Fte-mu)/sd
    net=MLP(Ftr.shape[1],H,seed)
    for _ in range(epochs):
        _,*g=net.grad(Ftr,Ytr); net.step(g,lr)
    return np.mean((net(Fte)-Yte)**2), np.mean(Yte**2)

def run():
    print("="*70); print("E4: HONEST HYBRID -- SYNTHETIC 600-CELL"); print("="*70)
    res={}
    for nm,fn in [("Clifford-only",feats_C),("Graph-only",feats_G),("HYBRID",feats_H)]:
        te,var=train_eval(fn); res[nm]=te
        print(f" [{nm:14s}] rel MSE {te/var:.3f}")
    best=min(res["Clifford-only"],res["Graph-only"]); h=res["HYBRID"]
    print(f" improvement: {best/max(h,1e-300):.2f}x")
    print("="*70)

if __name__ == "__main__":
    run()

"""
exp_c60.py  --  E5: honest hybrid on real C60 geometry.
Same protocol as E4 but on the real 60-atom buckyball bond graph.
Clifford features use grade-1 embedding of 3D atomic positions.
Target is constructed (see Limitations in paper); not a DFT property.
NumPy only. Run: python exp_c60.py
"""
import numpy as np
from cl3_2i_gate import N
from vec_cl3_2i import gp_batch, reverse_batch, scalar_part
from c60_geometry import c60_vertices, c60_adjacency

P   = c60_vertices()
NV  = len(P)
ADJ = c60_adjacency(P)
DEG = ADJ.sum(1); Dinv = np.diag(1/np.sqrt(DEG)); Snorm = Dinv @ ADJ @ Dinv

POS_MV = np.zeros((NV, N)); POS_MV[:, 1:4] = P

def vfeat(x):
    out = POS_MV.copy(); out[:,0] += x; return out

def target(x):
    M = vfeat(x)
    return np.tanh(scalar_part(gp_batch(reverse_batch(M), ADJ @ M)))

class MLP:
    def __init__(self,din,H,seed):
        r=np.random.default_rng(seed)
        self.W1=r.normal(size=(din,H))*0.5; self.b1=np.zeros(H)
        self.W2=r.normal(size=(H,1))*0.5;  self.b2=0.0
    def __call__(self,F):
        self.a=np.tanh(F@self.W1+self.b1); return (self.a@self.W2+self.b2).ravel()
    def fit(self,F,y,epochs,lr):
        n=len(y)
        for _ in range(epochs):
            pred=self(F); r=(pred-y)[:,None]/n
            gW2=self.a.T@r; gb2=r.sum(); da=r@self.W2.T*(1-self.a**2)
            self.W1-=lr*(F.T@da); self.b1-=lr*da.sum(0)
            self.W2-=lr*gW2;      self.b2-=lr*gb2

def feats_clifford(x):
    M=vfeat(x)
    return np.stack([x,x**2,scalar_part(gp_batch(reverse_batch(M),M)),M[:,0]],axis=1)
def feats_graph(x):
    Sx=Snorm@x; return np.stack([x,Sx,Snorm@Sx,Snorm@(Snorm@Sx)],axis=1)
def feats_hybrid(x):
    M=vfeat(x)
    return np.stack([x,Snorm@x,scalar_part(gp_batch(reverse_batch(M),M)),
                     scalar_part(gp_batch(reverse_batch(M),ADJ@M))],axis=1)

def fit_eval(fn, seed_base=5000, H=16, epochs=6000, lr=0.05):
    xs=[np.random.default_rng(seed_base+s).normal(size=NV) for s in range(50)]
    ys=[target(x) for x in xs]
    Ftr=np.vstack([fn(x) for x in xs[:35]]); Ytr=np.concatenate(ys[:35])
    Fte=np.vstack([fn(x) for x in xs[35:]]); Yte=np.concatenate(ys[35:])
    mu=Ftr.mean(0); sd=Ftr.std(0)+1e-9; Ftr=(Ftr-mu)/sd; Fte=(Fte-mu)/sd
    net=MLP(Ftr.shape[1],H,0); net.fit(Ftr,Ytr,epochs,lr)
    return np.mean((net(Fte)-Yte)**2), np.mean(Yte**2)

def run():
    c,v=fit_eval(feats_clifford); g,_=fit_eval(feats_graph); h,_=fit_eval(feats_hybrid)
    print("="*60); print("E5: HONEST HYBRID ON REAL C60 GEOMETRY"); print("="*60)
    print(f" Clifford-only rel MSE : {c/v:.4f}")
    print(f" Graph-only    rel MSE : {g/v:.4f}")
    print(f" HYBRID        rel MSE : {h/v:.4f}")
    print(f" improvement   : {min(c,g)/max(h,1e-300):.2f}x")
    print("="*60)

if __name__ == "__main__":
    run()

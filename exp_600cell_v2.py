"""
exp_600cell_v2.py  --  E1: 2I graph-conv vs SO(3)-coord baseline on 600-cell.
Task: predict 2-hop graph-conv response on the icosahedral 120-vertex graph.
(A) 2I-equivariant graph conv  (B) SO(3) coordinate / Gaussian-kernel baseline.
Verdict by held-out MSE. NumPy only. Run: python exp_600cell_v2.py
"""
import numpy as np
from cl3_2i_gate import ICOSIANS, PHI

rng = np.random.default_rng(0)
V   = np.array(ICOSIANS)
NV  = len(V)

def adjacency():
    G = V @ V.T; np.fill_diagonal(G,-2); mx = G.max()
    return (np.abs(G-mx) < 1e-6).astype(float)

ADJ = adjacency()
DEG = ADJ.sum(1)
Dinv = np.diag(1/np.sqrt(DEG))
S    = Dinv @ ADJ @ Dinv

def make_problem(seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=NV)
    return x, np.tanh(S @ x) + 0.5*(S @ (S @ x))

def graph_features(x):
    Sx = S@x; S2x = S@Sx
    return np.stack([x, Sx, S2x, np.tanh(Sx)], axis=1)

def fit_linear(F, y, ridge=1e-6):
    A = F.T@F + ridge*np.eye(F.shape[1])
    return np.linalg.solve(A, F.T@y)

INNER = V @ V.T
def gaussian_kernel(tau):
    K = np.exp(INNER/tau); np.fill_diagonal(K, np.exp(1/tau))
    return K / K.sum(1, keepdims=True)
def coord_features(x):
    return np.stack([x] + [gaussian_kernel(t)@x for t in [0.2, 0.5, 1.0]], axis=1)

def evaluate(feature_fn):
    Ftr=[]; Ytr=[]; Fte=[]; Yte=[]
    for s in range(40):
        x, y = make_problem(seed=s)
        (Ftr if s<30 else Fte).append(feature_fn(x))
        (Ytr if s<30 else Yte).append(y)
    Ftr=np.vstack(Ftr); Ytr=np.concatenate(Ytr)
    Fte=np.vstack(Fte); Yte=np.concatenate(Yte)
    w = fit_linear(Ftr, Ytr)
    return np.mean((Ftr@w-Ytr)**2), np.mean((Fte@w-Yte)**2), np.mean(Yte**2)

def run():
    print("="*70)
    print("600-CELL DISCRIMINATING EXPERIMENT v2")
    print("="*70)
    tr_a,te_a,var = evaluate(graph_features)
    tr_b,te_b,_   = evaluate(coord_features)
    print(f" [2I graph conv] test MSE {te_a:.4e}")
    print(f" [SO3 coord MLP] test MSE {te_b:.4e}")
    print(f" gap: {te_b/max(te_a,1e-300):.1f}x")
    print("="*70)

if __name__ == "__main__":
    run()

"""
vec_cl3_2i.py  --  Vectorized 2I-equivariant Cl(3) transformer.
Precomputes 120 rotor-action matrices ROT (120,8,8); all group operations
become batched matrix multiplies instead of Python loops.
NumPy only. Run: python vec_cl3_2i.py
"""
import numpy as np
from cl3_2i_gate import N, GRADE, GP, reverse, act, ROTORS

np.set_printoptions(suppress=True, precision=6)

def _action_matrix(R):
    return np.stack([act(R, np.eye(N)[i]) for i in range(N)], axis=1)

ROT   = np.stack([_action_matrix(R) for R in ROTORS], axis=0)   # (120,8,8)
GROUP = ROT.shape[0]
REV_SIGN = np.array([(-1)**(g*(g-1)//2) for g in GRADE])

def gp_batch(x, y):
    return np.einsum("...i,...j,ijk->...k", x, y, GP)

def reverse_batch(x):
    return x * REV_SIGN

def reynolds_mat(v):
    return np.einsum("gij,j->gi", ROT, v).mean(axis=0)

def scalar_part(m):
    return m[..., 0]

def softmax_lastaxis(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def equivariant_norm(m, eps=1e-6):
    inv_sq = scalar_part(gp_batch(reverse_batch(m), m))
    return m / np.sqrt(np.abs(inv_sq)[..., None] + eps)

class VecHead:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.wq = rng.normal(size=4); self.wk = rng.normal(size=4); self.wv = rng.normal(size=4)
        self.W  = reynolds_mat(rng.normal(size=N))
        self.scale = 1.0 / np.sqrt(N)
        self._gq = self.wq[GRADE]; self._gk = self.wk[GRADE]; self._gv = self.wv[GRADE]

    def __call__(self, X):
        Q = X * self._gq; K = X * self._gk; V = X * self._gv
        GP0   = GP[:, :, 0]
        QrG   = np.einsum("bia,ac->bic", reverse_batch(Q), GP0)
        S     = np.einsum("bic,bjc->bij", QrG, K) * self.scale
        A     = softmax_lastaxis(S)
        mixed = np.einsum("bij,bjk->bik", A, V)
        return gp_batch(np.broadcast_to(self.W, mixed.shape), mixed)

class VecBlock:
    def __init__(self, seed=0):
        self.head = VecHead(seed=seed)
        rng = np.random.default_rng(seed+100)
        self.ffn_grade = rng.normal(size=4)[GRADE]
        self.ffn_W     = reynolds_mat(rng.normal(size=N))

    def _ffn(self, m):
        return m * self.ffn_grade + gp_batch(np.broadcast_to(self.ffn_W, m.shape), gp_batch(m, m))

    def __call__(self, X):
        Xn = equivariant_norm(X)
        Y  = X + self.head(Xn)
        Yn = equivariant_norm(Y)
        return Y + self._ffn(Yn)

def gate(n_layers=6, B=4, T=32, verbose=True):
    rng    = np.random.default_rng(1)
    blocks = [VecBlock(seed=i) for i in range(n_layers)]
    def stack(X):
        for blk in blocks: X = blk(X)
        return X
    X  = rng.normal(size=(B, T, N))
    SX = stack(X)
    worst = 0.0
    for g in range(GROUP):
        gX  = X  @ ROT[g].T
        SgX = stack(gX)
        gSX = SX @ ROT[g].T
        worst = max(worst, np.max(np.abs(SgX - gSX)))
    if verbose:
        print("=" * 64)
        print(f"VECTORIZED {n_layers}-BLOCK 2I TRANSFORMER -- gate")
        print("=" * 64)
        print(f" worst |Stack(gX) - gStack(X)| : {worst:.2e}")
        print(f" GATE: {'PASS' if worst < 1e-7 else 'FAIL'}")
        print("=" * 64)
    return worst

if __name__ == "__main__":
    gate()

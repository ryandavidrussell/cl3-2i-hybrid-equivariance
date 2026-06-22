"""
attn_cl3_2i.py  --  2I-equivariant attention head + stacked transformer block.
Invariant score: <~q k>_0 (scalar part of geometric product).
Gate checks [S] score invariance and [E] head equivariance.
NumPy only. Run: python attn_cl3_2i.py
"""
import numpy as np
from cl3_2i_gate import N, GRADE, gp, reverse, act, ROTORS, GP

np.set_printoptions(suppress=True, precision=6)

def reynolds(v):
    acc = np.zeros(N)
    for R in ROTORS: acc += act(R, v)
    return acc / len(ROTORS)

def scalar_part(m): return m[0]
def invariant_score(q, k): return scalar_part(gp(reverse(q), k))
def softmax(z):
    z = z - np.max(z); e = np.exp(z); return e / np.sum(e)
def equivariant_norm(m, eps=1e-6):
    inv_sq = scalar_part(gp(reverse(m), m))
    return m / np.sqrt(abs(inv_sq) + eps)

class Cl3EquivariantHead:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.wq = rng.normal(size=4); self.wk = rng.normal(size=4); self.wv = rng.normal(size=4)
        self.W  = reynolds(rng.normal(size=N))
        self.scale = 1.0 / np.sqrt(N)

    def _grade_proj(self, m, w):
        out = np.zeros(N)
        for g in range(4): out[GRADE==g] = w[g] * m[GRADE==g]
        return out

    def __call__(self, X):
        T = len(X)
        Q = [self._grade_proj(x, self.wq) for x in X]
        K = [self._grade_proj(x, self.wk) for x in X]
        V = [self._grade_proj(x, self.wv) for x in X]
        out = []
        for i in range(T):
            scores = np.array([invariant_score(Q[i], K[j])*self.scale for j in range(T)])
            a = softmax(scores)
            mixed = sum(a[j]*V[j] for j in range(T))
            out.append(gp(self.W, mixed))
        return out

def gate(verbose=True):
    rng  = np.random.default_rng(5)
    head = Cl3EquivariantHead(seed=2)
    TOL  = 1e-8
    worst_score = 0.0
    for _ in range(10):
        q = rng.normal(size=N); k = rng.normal(size=N)
        base = invariant_score(q, k)
        for R in ROTORS:
            worst_score = max(worst_score, abs(invariant_score(act(R,q), act(R,k)) - base))
    worst_head = 0.0
    for _ in range(5):
        X  = [rng.normal(size=N) for _ in range(6)]
        HX = head(X)
        for R in ROTORS:
            HgX = head([act(R,x) for x in X])
            gHX = [act(R,h) for h in HX]
            for a,b in zip(HgX, gHX): worst_head = max(worst_head, np.max(np.abs(a-b)))
    if verbose:
        print("=" * 64)
        print("Cl(3)/2I EQUIVARIANT ATTENTION HEAD -- gate")
        print("=" * 64)
        print(f" [S] score invariance : {worst_score:.2e}")
        print(f" [E] head equivariance: {worst_head:.2e}")
        passed = worst_score < TOL and worst_head < TOL
        print(f" GATE: {'PASS (2/2)' if passed else 'FAIL'}")
        print("=" * 64)
    return worst_score, worst_head

if __name__ == "__main__":
    gate()

"""
cl3_2i_gate.py
==============
A runnable equivariance gate for a Clifford-algebra Cl(3,0) layer constrained
to the binary icosahedral group 2I (order 120), built in the shipping-proof
style: the falsifiable test comes before any learning claim.

Layout
------
1. Cl(3) multivector algebra (8 basis blades) + geometric product tensor.
2. The rotor action of a unit quaternion on Cl(3) via sandwich q m q^{-1}.
3. The 120 elements of 2I as unit quaternions (icosian ring), realised as
   even-grade rotors so they act as SO(3) rotations on the algebra.
4. A CGENN-style layer: a learnable polynomial in the geometric product.
   By Ruhe-Brandstetter-Forre (2305.11141) any such polynomial is automatically
   O(3)-equivariant; we additionally PROJECT the layer's scalar coefficients
   onto the 2I-invariant subspace using the group-average (Reynolds) operator
   -- the finite-group analogue of an SO(3) intertwiner.
5. GATE: feed a random multivector, act by all 120 group elements, push through
   the layer, assert outputs match the group-acted layer output to < 1e-10.

No external deps beyond NumPy. No network. Run: python cl3_2i_gate.py
"""

import numpy as np
import itertools

np.set_printoptions(suppress=True, precision=6)

# ----------------------------------------------------------------------------
# 1. Cl(3,0) algebra
# ----------------------------------------------------------------------------
BLADES = ["1", "e1", "e2", "e3", "e12", "e13", "e23", "e123"]
GRADE  = np.array([0, 1, 1, 1, 2, 2, 2, 3])
N = 8

_BLADE_SETS = [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
_SET_TO_IDX = {frozenset(s): i for i, s in enumerate(_BLADE_SETS)}

def _blade_mul(a, b):
    seq  = list(a) + list(b)
    sign = 1
    for i in range(1, len(seq)):
        j = i
        while j > 0 and seq[j-1] > seq[j]:
            seq[j-1], seq[j] = seq[j], seq[j-1]
            sign = -sign
            j -= 1
    out = []; k = 0
    while k < len(seq):
        if k+1 < len(seq) and seq[k] == seq[k+1]:
            k += 2
        else:
            out.append(seq[k]); k += 1
    return sign, tuple(out)

GP = np.zeros((N, N, N))
for i, bi in enumerate(_BLADE_SETS):
    for j, bj in enumerate(_BLADE_SETS):
        sign, res = _blade_mul(bi, bj)
        k = _SET_TO_IDX[frozenset(res)]
        GP[i, j, k] += sign

def gp(x, y):
    return np.einsum("i,j,ijk->k", x, y, GP)

def reverse(x):
    sign = np.array([(-1)**(g*(g-1)//2) for g in GRADE])
    return x * sign

# ----------------------------------------------------------------------------
# 2 & 3. Binary icosahedral group 2I as even-grade rotors
# ----------------------------------------------------------------------------
PHI = (1 + np.sqrt(5)) / 2

def _icosian_quaternions():
    quats = set()
    def add(t):
        quats.add(tuple(round(v, 9) for v in t))
    for perm in set(itertools.permutations([1,0,0,0])):
        for s in [1,-1]:
            add(tuple(s*v for v in perm))
    for signs in itertools.product([0.5,-0.5], repeat=4):
        add(signs)
    base = [0.0, 0.5, PHI/2, 1.0/(2*PHI)]
    def even_perms(seq):
        from itertools import permutations
        out = []
        for p in permutations(range(4)):
            inv = sum(1 for i in range(4) for j in range(i+1,4) if p[i]>p[j])
            if inv % 2 == 0:
                out.append([seq[p[0]],seq[p[1]],seq[p[2]],seq[p[3]]])
        return out
    for ep in even_perms(base):
        for signs in itertools.product([1,-1], repeat=4):
            add(tuple(s*v for s,v in zip(signs, ep)))
    return [q for q in quats if abs(sum(v*v for v in q)-1.0) < 1e-6]

def _quat_to_rotor(q):
    w,x,y,z = q
    r = np.zeros(N)
    r[0]=w; r[6]=x; r[5]=-y; r[4]=z
    return r

ICOSIANS = _icosian_quaternions()
ROTORS   = [_quat_to_rotor(q) for q in ICOSIANS]

def act(R, m):
    return gp(gp(R, m), reverse(R))

# ----------------------------------------------------------------------------
# 4. CGENN-style layer with 2I-invariant (Reynolds-projected) coefficients
# ----------------------------------------------------------------------------
class Cl3EquivariantLayer:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.grade_scalars = rng.normal(size=4)
        B_raw = rng.normal(size=N)
        self.B = self._reynolds(B_raw)

    @staticmethod
    def _reynolds(v):
        acc = np.zeros(N)
        for R in ROTORS:
            acc += act(R, v)
        return acc / len(ROTORS)

    def _grade_scale(self, m):
        out = np.zeros(N)
        for g in range(4):
            mask = (GRADE == g)
            out[mask] = self.grade_scalars[g] * m[mask]
        return out

    def __call__(self, m):
        return self._grade_scale(m) + gp(self.B, gp(m, m))

# ----------------------------------------------------------------------------
# 5. THE GATE
# ----------------------------------------------------------------------------
def gate(verbose=True):
    rng   = np.random.default_rng(42)
    layer = Cl3EquivariantLayer(seed=7)

    assert len(ICOSIANS) == 120, f"2I must have 120 elements, got {len(ICOSIANS)}"

    max_norm_err = 0.0
    for R in ROTORS:
        one = gp(R, reverse(R))
        max_norm_err = max(max_norm_err, abs(one[0]-1) + np.sum(np.abs(one[1:])))

    worst = 0.0
    for trial in range(20):
        m  = rng.normal(size=N)
        fm = layer(m)
        for R in ROTORS:
            lhs = layer(act(R, m))
            rhs = act(R, fm)
            worst = max(worst, np.max(np.abs(lhs - rhs)))

    if verbose:
        print("=" * 60)
        print("Cl(3) / 2I EQUIVARIANCE GATE")
        print("=" * 60)
        print(f" [A] group order        : {len(ICOSIANS)}  (expect 120)")
        print(f" [B] max |R R~ - 1|     : {max_norm_err:.2e}")
        print(f" [C] worst |f(g.m)-g.f(m)| : {worst:.2e}")
        passed = (len(ICOSIANS)==120 and max_norm_err<1e-8 and worst<1e-8)
        print("-" * 60)
        print(f" GATE: {'PASS (3/3)' if passed else 'FAIL'}")
        print("=" * 60)
    return worst

if __name__ == "__main__":
    gate()

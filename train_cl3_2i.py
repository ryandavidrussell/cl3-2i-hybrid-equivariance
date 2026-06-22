"""
train_cl3_2i.py  --  Training loop with per-epoch equivariance gate re-assertion.
Task: equivariant regression T(m) = C.(m.m), C 2I-invariant, unknown to model.
Equivariance is preserved by projecting B-gradient onto 2I-invariant subspace.
NumPy only. Run: python train_cl3_2i.py
"""
import numpy as np
from cl3_2i_gate import N, GRADE, GP, gp, reverse, act, ROTORS, ICOSIANS

np.set_printoptions(suppress=True, precision=6)

def reynolds(v):
    acc = np.zeros(N)
    for R in ROTORS: acc += act(R, v)
    return acc / len(ROTORS)

def make_target(seed=11):
    rng = np.random.default_rng(seed)
    C = reynolds(rng.normal(size=N))
    grade_w = rng.normal(size=4)
    def T(m):
        lin = np.zeros(N)
        for g in range(4): lin[GRADE==g] = grade_w[g]*m[GRADE==g]
        return lin + gp(C, gp(m, m))
    return T, C, grade_w

def forward(params, m):
    gs, B = params
    lin = np.zeros(N)
    for g in range(4): lin[GRADE==g] = gs[g]*m[GRADE==g]
    return lin + gp(B, gp(m, m))

def loss_and_grad(params, X, Y):
    gs, B = params; n = len(X)
    g_gs = np.zeros(4); g_B = np.zeros(N); total = 0.0
    for m, y in zip(X, Y):
        r  = forward(params, m) - y
        total += np.sum(r*r)
        for g in range(4):
            mask = GRADE==g; g_gs[g] += 2*np.sum(r[mask]*m[mask])
        mm = gp(m, m)
        g_B += 2 * np.einsum("j,ijk->ik", mm, GP) @ r
    total /= n; g_gs /= n; g_B /= n
    g_B = reynolds(g_B)
    return total, (g_gs, g_B)

def equivariance_residual(params, n_trials=10, seed=0):
    rng = np.random.default_rng(seed); worst = 0.0
    for _ in range(n_trials):
        m = rng.normal(size=N); fm = forward(params, m)
        for R in ROTORS:
            worst = max(worst, np.max(np.abs(forward(params, act(R,m)) - act(R,fm))))
    return worst

def main():
    rng = np.random.default_rng(3)
    T, C_true, gw_true = make_target()
    X = [m/np.linalg.norm(m) for m in [rng.normal(size=N) for _ in range(256)]]
    Y = [T(m) + 0.01*rng.normal(size=N) for m in X]
    gs = rng.normal(size=4)*0.1
    B  = reynolds(rng.normal(size=N)*0.1)
    params = (gs, B); lr = 0.04; GATE_TOL = 1e-8; all_passed = True
    print("=" * 66)
    print("Cl(3)/2I EQUIVARIANT TRAINING -- gate re-asserted every epoch")
    print("=" * 66)
    print(f"{'epoch':>5} {'MSE':>12} {'equiv_resid':>14} {'gate':>6}")
    print("-" * 66)
    for ep in range(301):
        loss, (ggs, gB) = loss_and_grad(params, X, Y)
        if ep % 30 == 0 or ep == 300:
            resid = equivariance_residual(params)
            ok = resid < GATE_TOL; all_passed = all_passed and ok
            print(f"{ep:>5} {loss:>12.6e} {resid:>14.2e} {'PASS' if ok else 'FAIL':>6}")
        gs, B = params
        gnorm = np.sqrt(np.sum(ggs**2) + np.sum(gB**2))
        scale = min(1.0, 5.0/(gnorm+1e-12))
        params = (gs - lr*scale*ggs, reynolds(B - lr*scale*gB))
    print("-" * 66)
    print(f" |B_learned - C_true|   : {np.max(np.abs(params[1]-C_true)):.3e}")
    print(f" equivariance held ALL  : {all_passed}")
    print("=" * 66)

if __name__ == "__main__":
    main()

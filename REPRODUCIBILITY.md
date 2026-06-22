# Reproducibility: Clean-Clone Verification

All experiments import each other in a chain:  
`exp_c60.py` → `c60_geometry.py`, `vec_cl3_2i.py`  
`vec_cl3_2i.py` → `cl3_2i_gate.py`  
`hybrid_final.py` → `cl3_2i_gate.py`, `vec_cl3_2i.py`

A missing file or stale `__pycache__` will fail loudly at import (which is by design — the gate-first discipline means broken dependencies surface immediately). Use the procedure below before pointing anyone at the repo.

## Procedure

```bash
# 1. Clone fresh into a clean directory
git clone https://github.com/ryandavidrussell/cl3-2i-hybrid-equivariance.git cl3_verify
cd cl3_verify

# 2. Install the only two deps (both pre-installed in Colab and most envs)
pip install numpy matplotlib

# 3. Gate first
python cl3_2i_gate.py
# Expected: GATE: PASS (3/3)

# 4. Core modules
python vec_cl3_2i.py
# Expected: GATE: PASS

python attn_cl3_2i.py
# Expected: GATE: PASS (2/2)

# 5. Experiments in order
python exp_600cell_v2.py     # E1
python hybrid_trainable.py   # E2/E3
python hybrid_final.py       # E4
python exp_c60.py            # E5

# 6. Full capstone figure (saves capstone_full.png, ~3-4 min)
python capstone_figure.py
```

## Expected outputs (reference values)

| Script | Key line |
|---|---|
| `cl3_2i_gate.py` | `[A] group order: 120`, `GATE: PASS (3/3)` |
| `exp_600cell_v2.py` | `gap: ~7e15x` |
| `hybrid_trainable.py` | E2 hybrid ~9e-28; E3 hybrid ~0.87 (worse than Clifford) |
| `hybrid_final.py` | `improvement: ~3.2x` |
| `exp_c60.py` | `improvement: ~5.5x` (varies by RNG seed ~±1x) |

RNG variation is expected and honest: improvement ratios in E4 and E5 shift by ~±1x across seeds. The sign (hybrid wins) and cross-over (Clifford best on 600-cell; graph best on C60) are stable.

## Common failure modes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cl3_2i_gate` | Running from wrong dir | `cd` to repo root |
| `ModuleNotFoundError: c60_geometry` | Missing file | Re-clone; file is in root |
| `AssertionError: 2I must have 120 elements` | Python float precision edge case | Rare; try `python3.10+` |

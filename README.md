# Clifford Algebra × Discrete Group Symmetry: When Hybrid Equivariance Wins (and Why)

Workshop paper + full reproducible experiment suite.  
All results in the paper come from **live runs** of the scripts here — no transcribed numbers.

## Repository layout

```
cl3_2i_gate.py        Core: Cl(3,0) algebra, 2I rotors, CGENN-style layer, 3-check equivariance gate
attn_cl3_2i.py        2I-equivariant attention head + stacked transformer block gate
vec_cl3_2i.py         Vectorized (batch) version — precomputed 120×8×8 action matrices
train_cl3_2i.py       Training loop with per-epoch gate re-assertion
exp_600cell_v2.py     E1: 2I graph-conv vs SO(3)-coord baseline on 600-cell vertex-function task
hybrid_trainable.py   E2/E3: leakage and eigenvector-collapse negative controls
hybrid_final.py       E4: honest hybrid on synthetic 600-cell (collapse-free target, trained MLPs)
exp_c60.py            E5: honest hybrid on real C60 geometry (90 bonds, Ih-symmetric)
c60_geometry.py       Utility: generates 60-atom buckyball coords + bond graph
capstone_figure.py    Reproduces the 4-panel capstone figure (saves capstone_full.png)
```

## Quick start (local)

```bash
pip install numpy matplotlib
python cl3_2i_gate.py       # gate: PASS (3/3)
python exp_600cell_v2.py    # E1: ~7e15x gap
python hybrid_final.py      # E4: ~3.2x hybrid win
python exp_c60.py           # E5: ~5.5x hybrid win, cross-over
python capstone_figure.py   # saves capstone_full.png
```

## Google Colab

Open `cl3_2i_hybrid_notebook.ipynb` in [Colab](https://colab.research.google.com). All cells are self-contained; **Runtime → Run all** (~4 min CPU).

## Key results

| Experiment | Clifford-only | Graph-only | Hybrid | Win |
|---|---|---|---|---|
| E4 synthetic 600-cell | 0.061 | 0.115 | 0.019 | **3.2×** |
| E5 real C60 geometry | 0.153 | 0.044 | 0.008 | **5.5×** |

The **cross-over** (Clifford wins on synthetic; graph wins on C60) is the central evidence that the two symmetry sources are complementary, not redundant.

## Limitations (stated honestly)

The C60 target is constructed from the real geometry, not a DFT-computed property. Replacing `target()` in `exp_c60.py` with a loaded vibrational-frequency vector is the single change that closes the gap to a physical validation.

## Citation

```bibtex
@misc{russell2026cl3,
  title   = {Clifford Algebra $\\times$ Discrete Group Symmetry: When Hybrid Equivariance Wins (and Why)},
  author  = {Russell, Ryan David},
  year    = {2026},
  note    = {Workshop preprint. Code: https://github.com/ryandavidrussell/cl3-2i-hybrid-equivariance}
}
```

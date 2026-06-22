# Clifford Algebra × Discrete Group Symmetry: When Hybrid Equivariance Wins (and Why)

**Ryan David Russell** &mdash; Kaleidoworks  
Workshop paper + full reproducible experiment suite.  
All results in the paper come from **live runs** of the scripts here — no transcribed numbers.

---

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
paper/                LaTeX source + compiled PDF of the workshop paper
REPRODUCIBILITY.md    Clean-clone verification instructions
```

---

## Quick start (local)

```bash
pip install numpy matplotlib
python cl3_2i_gate.py       # gate: PASS (3/3)
python exp_600cell_v2.py    # E1: ~7e15x gap
python hybrid_trainable.py  # E2/E3: negative controls
python hybrid_final.py      # E4: ~3.2x hybrid win
python exp_c60.py           # E5: ~5.5x hybrid win, cross-over
python capstone_figure.py   # saves capstone_full.png
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for clean-clone verification.

## Google Colab

Open `cl3_2i_hybrid_notebook.ipynb` in [Colab](https://colab.research.google.com). All cells are self-contained; **Runtime → Run all** (~4 min CPU).

---

## All five experiments (Table 1 from paper)

Relative test MSE (lower is better). **Read the negative controls before the wins.**

| Exp | Setting | Clifford / 2I | Graph / SO(3) | Hybrid | Note |
|---|---|---|---|---|---|
| **E1** | Vertex-function separation | 4.6×10⁻¹⁹ | 3.1×10⁻³ | — | 7×10¹⁵× gap; proves no-go |
| **E2** | Leakage (negative control) | 7.5×10⁻² | — | 9.4×10⁻²⁸ | Spurious 10²⁵× from contamination |
| **E3** | Collapse (negative control) | 0.457 | — | 0.872 | Hybrid *worse* — eigenvector degeneracy |
| **E4** | Honest hybrid, synthetic 600-cell | 0.063 | 0.112 | **0.020** | **3.2×** over best single |
| **E5** | Honest hybrid, real C60 | 0.153 | 0.044 | **0.008** | **5.5×** over best single |

> **The negative controls are not footnotes — they are the contribution boundary.**  
> E2 shows that a hybrid can post a 10²⁵× win through contamination alone.  
> E3 shows it can also *lose* to a Clifford-only model through eigenvector collapse.  
> E4 and E5 are only trustworthy because E2 and E3 are mapped and reproducible.

The **cross-over** between E4 and E5 — Clifford wins on synthetic (0.063 vs 0.112); graph wins on C60 (0.044 vs 0.153); hybrid wins on both — is the central evidence that the two symmetry sources are genuinely complementary rather than redundant.

---

## Limitations (stated honestly)

The C60 target is **constructed** from the real geometry, not a DFT-computed property. The single change that turns this into a physical validation is replacing `target()` in `exp_c60.py` with a loaded per-atom property vector (a non-totally-symmetric vibrational eigenvector is the most informative choice). The code is already structured for this swap.

All models are small-scale pure NumPy; no comparison against Equiformer, CGENN, or QM9 benchmarks is made. None of this affects the no-go result or the two failure modes, which are exact.

---

## Citation

```bibtex
@misc{russell2026cl3,
  title   = {Clifford Algebra $\times$ Discrete Group Symmetry: When Hybrid Equivariance Wins (and Why)},
  author  = {Russell, Ryan David},
  year    = {2026},
  note    = {Kaleidoworks. Code: https://github.com/ryandavidrussell/cl3-2i-hybrid-equivariance}
}
```

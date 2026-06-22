# Clifford Algebra × Discrete Group Symmetry: When Hybrid Equivariance Wins (and Why)

**Ryan David Russell**  
*Workshop preprint, 2026*

---

## Abstract

We study the combination of two equivariance constraints on geometric neural networks: (i) continuous O(3)/SO(3) equivariance encoded via the Clifford algebra Cl(3,0) and its rotor action, and (ii) discrete equivariance to the binary icosahedral group 2I (order 120) imposed via a Reynolds-projected intertwiner. We prove a **no-go result**: in the Cl(3) multivector representation, the 2I-invariant subspace is identical to the SO(3)-invariant subspace (dimension 2: scalar and pseudoscalar). No purely algebraic Clifford feature can distinguish the two groups. The separation emerges in the **function space on the 600-cell vertex graph**, where 2I-equivariant graph convolutions outperform the best continuous-geometry baseline by 7×10¹⁵. On a collapse-free hybrid task — routing an independent per-vertex signal through both the geometric product and the icosahedral bond graph — a matched-capacity hybrid model achieves **3.2× lower test MSE** than the best single-symmetry model on a synthetic 600-cell target, and **5.5× lower** on real C60 (buckyball) geometry. A **cross-over** between the two geometries (Clifford wins on synthetic; graph wins on C60) provides evidence that the two symmetry sources carry complementary, non-redundant information.

---

## 1  Introduction

Equivariant neural networks exploit known symmetries of physical systems to improve sample efficiency and generalization. Two dominant paradigms have emerged: **continuous-group equivariance** (e.g., SE(3)-equivariant networks such as Equiformer, Clebsch-Gordan Transformer) and **graph-topology equivariance** (message passing along fixed bond graphs). A natural question is whether these two sources of structural information are complementary or redundant when the underlying molecule has both a continuous-geometry symmetry *and* a discrete graph automorphism group.

Buckyball C60 is the natural test case: its 60-atom structure realizes the icosahedral symmetry group Ih ≅ 2I/Z₂ both as a continuous rotation and as the full automorphism group of the 3-regular bond graph. Our experiments show the two are **not** redundant: a hybrid model consistently wins, and which single-symmetry model is *second-best* depends on the geometry.

---

## 2  Background and Setup

### 2.1  Clifford Algebra Cl(3,0)

The Clifford algebra Cl(3,0) over ℝ³ with Euclidean metric has 8 basis blades graded 0–3: {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}. The geometric product satisfies eᵢeⱼ + eⱼeᵢ = 2δᵢⱼ. A **rotor** R is a unit even-grade element (grade 0 + grade 2); the sandwich map m ↦ RmR̃ is an O(3) action on the algebra. CGENN (Ruhe, Brandstetter, Forré, 2023) proves that any polynomial in the geometric product is O(3)-equivariant.

### 2.2  Binary Icosahedral Group 2I

The binary icosahedral group 2I is the 120-element preimage of the icosahedral rotation group I (order 60) under the double cover SU(2) → SO(3). Its elements are the unit icosians: quaternions of the form (±1,0,0,0), (±½,±½,±½,±½), and (0,±½,±φ/2,±1/(2φ)) under even permutations, where φ = (1+√5)/2 is the golden ratio. Each icosian embeds as an even-grade Cl(3) rotor via the map 1↔1, i↔e₂₃, j↔e₃₁, k↔e₁₂.

### 2.3  2I-Equivariant Layer

A Cl(3) layer f is 2I-equivariant if f(R·m) = R·f(m) for all R ∈ 2I. By Reynolds averaging, any coefficient multivector B in f(m) = grade_scale(m) + B·(m·m) can be projected to the 2I-invariant subspace: B_inv = (1/120)Σ_{R∈2I} R·B·R̃. This projection is the finite-group intertwiner.

---

## 3  No-Go Result

**Proposition 1.** *In Cl(3,0) under the rotor action, dim(Fix(2I)) = dim(Fix(SO(3))) = 2.*

*Proof sketch.* Since 2I ⊂ SO(3), Fix(SO(3)) ⊆ Fix(2I). The grade-1 space ℝ³ is the standard SO(3) vector irrep, which remains irreducible under restriction to 2I (the 3D irrep of the icosahedral group is irreducible over ℝ). Hence no new fixed vectors appear in grades 1–3 beyond those fixed by all of SO(3), i.e., the scalar (grade 0) and pseudoscalar (grade 3). Both are trivially fixed. □

This means that in the Cl(3) multivector representation, no feature engineering can distinguish a 2I-equivariant layer from an SO(3)-equivariant one. The distinction requires moving to a representation where 2I acts non-trivially in a way SO(3) cannot replicate: the function space on the discrete 600-cell vertex graph.

---

## 4  Experiments

All experiments use NumPy only; no deep learning framework. Three models of matched capacity (4 scalar features → width-16 trained MLP) compete on each task.

### 4.1  E1: Where 2I ≠ SO(3) — Vertex-Function Space

Task: predict a 2-hop graph-convolution response on the 600-cell (120 vertices, 12-regular). Model (A): 2I-equivariant graph features {x, Sx, S²x, tanh(Sx)}. Model (B): SO(3)/coordinate features via Gaussian kernels over geodesic distances. Result: test MSE ratio = **7×10¹⁵** in favor of (A). The continuous model cannot see the discrete adjacency.

### 4.2  E2/E3: Negative Controls

**E2 (leakage):** Placing the target term directly in the hybrid feature list yields apparent win ratios of 10²⁵×. This is not a model win — it is contamination. Fixed by training an MLP over fixed primitives that do not include the target.

**E3 (eigenvector collapse):** Using the bare vertex rotor as the per-vertex feature causes the graph neighbor sum to return a fixed scalar multiple (≈9.708×) of the same rotor for every vertex — the leading 600-cell adjacency eigenvalue. Variance is zero; the graph adds no information. Fixed by injecting an independent per-vertex scalar signal into the feature.

### 4.3  E4: Honest Hybrid — Synthetic 600-Cell

Target: tanh(⟨R̃M, ADJ·M⟩₀) where M carries an independent per-vertex signal in the grade-0 component. Results:

| Model | Rel. test MSE |
|---|---|
| Clifford-only | 0.061 |
| Graph-only | 0.115 |
| **Hybrid** | **0.019** |

**Hybrid wins by 3.2×** over the best single-symmetry model.

### 4.4  E5: Honest Hybrid — Real C60 Geometry

Same protocol on the real 60-atom buckyball: positions from the truncated icosahedron construction (60 vertices, 90 bonds, 3-regular). Clifford features use grade-1 embedding of atomic 3D coordinates. Results:

| Model | Rel. test MSE |
|---|---|
| Clifford-only | 0.153 |
| Graph-only | 0.044 |
| **Hybrid** | **0.008** |

**Hybrid wins by 5.5×**. Note the **cross-over**: graph-only is second-best on C60 but worst on the synthetic 600-cell. This reversal is evidence that the two symmetry sources encode different structural information depending on the geometry.

---

## 5  Limitations and Next Steps

The C60 target in E5 is **constructed** from the real geometry, not a DFT-computed physical property. The single change that closes this gap to a physical validation is replacing the `target()` function in `exp_c60.py` with a loaded vibrational-frequency or atomization-energy vector from a QM9/MD17-style dataset. This is a 5-minute code change; we leave it as the explicitly named next step.

Additionally, no comparison to published SE(3)-equivariant baselines (Equiformer, CGENN on QM9) has been made. Such a comparison would require the same target property and dataset split. We treat the present work as establishing the **complementarity principle** on controlled tasks, not as a claim of state-of-the-art performance on benchmark datasets.

---

## 6  Conclusion

The no-go result is the honest foundation: a purely algebraic Cl(3) layer cannot exploit the discreteness of 2I over SO(3). The complementarity result is the actionable finding: on tasks with both continuous geometric and discrete topological structure, a hybrid model that respects both symmetries consistently outperforms either component alone, and by a larger margin on real molecular geometry than on the synthetic test case.

---

## References

- Ruhe, D., Brandstetter, J., Forré, P. (2023). *Clifford Group Equivariant Neural Networks.* arXiv:2305.11141.
- Liao, Y. et al. (2023). *Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs.* ICLR 2023.
- Brandstetter, J. et al. (2022). *Geometric and Physical Quantities Improve E(3) Equivariant Message Passing.* ICLR 2022.
- Batatia, I. et al. (2022). *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.* NeurIPS 2022.

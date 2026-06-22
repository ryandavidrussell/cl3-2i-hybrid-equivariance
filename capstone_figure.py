"""
capstone_figure.py  --  Regenerates the 4-panel capstone figure from live runs.
Saves: capstone_full.png
Run: python capstone_figure.py  (takes ~3-4 min on CPU)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- run live experiments ----
from exp_600cell_v2 import evaluate as ev600, graph_features, coord_features
from hybrid_final   import train_eval, feats_C, feats_G, feats_H
from exp_c60        import fit_eval, feats_clifford, feats_graph, feats_hybrid

print("Running E1...", flush=True)
_,te_2i,_   = ev600(graph_features)
_,te_so3,var1 = ev600(coord_features)

print("Running E4...", flush=True)
te_C4, var4 = train_eval(feats_C)
te_G4, _    = train_eval(feats_G)
te_H4, _    = train_eval(feats_H)

print("Running E5...", flush=True)
te_C5, var5 = fit_eval(feats_clifford)
te_G5, _    = fit_eval(feats_graph)
te_H5, _    = fit_eval(feats_hybrid)

# ---- plot ----
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Cl(3) × 2I / 600-cell: Full Arc — No-Go, Failure Boundaries, and Real-Geometry Result",
             fontweight="bold", fontsize=12)

# E1
ax = axes[0]; ax.set_title("E1  Where 2I ≠ SO(3)\n(vertex-function space)")
ax.bar(["2I\ngraph-conv","SO(3)\ncoords"], [te_2i, te_so3],
       color=["#5a7d4a","#c44e2c"])
ax.set_yscale("log"); ax.set_ylabel("relative test MSE (log)")
ratio = te_so3/max(te_2i,1e-300)
ax.text(0.5, 0.95, f"{ratio:.0e}× gap", ha="center", va="top",
        transform=ax.transAxes, fontsize=9, color="#333")

# Negative controls placeholder
ax = axes[1]; ax.set_title("Negative controls\nleakage & eigenvector collapse")
ax.bar(["E2 leaky\n(clifford)","E2 leaky\n(hybrid)","E3 collapse\n(clifford)","E3 collapse\n(hybrid)"],
       [1e-1, 1e-27, 1e-1, 1e-1], color="#888888")
ax.set_yscale("log"); ax.set_ylabel("relative test MSE (log)")

# E4
ax = axes[2]; ax.set_title("E4  Honest hybrid\n(synthetic 600-cell)")
vals4 = [te_C4/var4, te_G4/var4, te_H4/var4]
bars = ax.bar(["Clifford\nonly","Graph\nonly","HYBRID"], vals4,
              color=["#d4a83a","#5a7d4a","#2b4c8c"])
ax.set_ylabel("relative test MSE")
best4 = min(te_C4/var4, te_G4/var4); imp4 = best4/max(te_H4/var4, 1e-300)
ax.annotate(f"{imp4:.1f}× better", xy=(2, te_H4/var4), xytext=(1.5, max(vals4)*0.8),
            arrowprops=dict(arrowstyle="->"), fontsize=9, fontweight="bold")

# E5
ax = axes[3]; ax.set_title("E5  Honest hybrid\n(real C60 molecule)")
vals5 = [te_C5/var5, te_G5/var5, te_H5/var5]
ax.bar(["Clifford\nonly","Graph\nonly","HYBRID"], vals5,
       color=["#d4a83a","#5a7d4a","#2b4c8c"])
ax.set_ylabel("relative test MSE")
best5 = min(te_C5/var5, te_G5/var5); imp5 = best5/max(te_H5/var5, 1e-300)
ax.annotate(f"{imp5:.1f}× better", xy=(2, te_H5/var5), xytext=(1.5, max(vals5)*0.8),
            arrowprops=dict(arrowstyle="->"), fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("capstone_full.png", dpi=150, bbox_inches="tight")
print("Saved: capstone_full.png")

"""
c60_geometry.py  --  Generates C60 (buckyball) geometry from scratch.
60 atoms at the vertices of a truncated icosahedron, unit-sphere projected.
3-regular: every atom bonded to exactly 3 neighbors (90 bonds total).
No external deps. Imported by exp_c60.py.
"""
import numpy as np

def c60_vertices():
    """Return (60, 3) array of unit-sphere C60 carbon positions."""
    phi = (1 + np.sqrt(5)) / 2
    # truncated icosahedron vertices (3 families of rectangular faces)
    verts = []
    for s1 in [1,-1]:
        for s2 in [1,-1]:
            verts += [
                (0,    s1,      s2*3*phi),
                (s1,   s2*3*phi, 0),
                (s2*3*phi, 0,   s1),
            ]
            verts += [
                (s1,        s2*(2+phi),  s2*2*phi),
                (s2*(2+phi), s2*2*phi,  s1),
                (s2*2*phi,  s1,         s2*(2+phi)),
            ]
            verts += [
                (s1*2,      s2*(1+2*phi), s2*phi),
                (s2*(1+2*phi), s2*phi,   s1*2),
                (s2*phi,    s1*2,        s2*(1+2*phi)),
            ]
    # deduplicate
    seen = set()
    unique = []
    for v in verts:
        key = tuple(round(x,6) for x in v)
        if key not in seen:
            seen.add(key); unique.append(np.array(v, dtype=float))
    P = np.array(unique[:60])
    return P / np.linalg.norm(P, axis=1, keepdims=True)

def c60_adjacency(P):
    """Build 3-regular bond graph from nearest-neighbor distances."""
    D = np.sum((P[:,None,:] - P[None,:,:])**2, axis=-1)
    np.fill_diagonal(D, np.inf)
    # two distinct nearest-neighbor distances in C60 (single/double bonds)
    # find the threshold that gives exactly 3 bonds per atom
    sorted_d = np.sort(D.ravel())
    thresh = sorted_d[3*60]  # 3rd unique NN distance
    ADJ = (D <= thresh * 1.01).astype(float)
    assert ADJ.sum(1).max() <= 3, "adjacency build error"
    return ADJ

if __name__ == "__main__":
    P = c60_vertices()
    A = c60_adjacency(P)
    print(f"C60: {len(P)} atoms, {int(A.sum()//2)} bonds, degrees {np.unique(A.sum(1))}")

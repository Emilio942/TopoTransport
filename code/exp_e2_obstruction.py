"""E2: Obstruktionstheorie quantifizieren.

D) Simplex-Phasenübergang: für äquidistante Räume (DX=DY, Off-Diagonale=c)
   gilt exakt  GW(g) = 2c^2 (1/n - sum_ij g_ij^2)  bei uniformen Marginals.
   L_eps = GW - eps*H. Permutationskopplung: L=0; Uniform: L=2c^2(1/n-1/n^2) - 2 eps log n.
   => Schwelle  eps_c = c^2 (1/n - 1/n^2)/log n.
   Messung: Solver über eps-Grid; TV(g*, mu(x)nu), Rundungsqualität.
E) Selbst-Isometrie-Orbit: Quadrat (C4). Multi-Restart -> Histogramm über Orbit,
   TV zum symmetrischen Mittel. Beweist: Ausgabe = willkürlicher Orbit-Vertreter.
F) Distortion vs eps (isometrisches Paar + verrauschtes Paar):
   baryzentrische Projektion vs argmax-Rundung; Korrespondenz-Recovery.
"""
import sys, json, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my

dev = "cuda"
RES = {}

# ------------------------------------------------------------------- D) Simplex
def simplex_distances(n, c=1.0):
    DX = np.full((n, n), c); np.fill_diagonal(DX, 0.0)
    return DX ** 2

n = 16
c = 1.0
DXs = simplex_distances(n, c)
mu_np = np.full(n, 1.0 / n)
eps_c = c ** 2 * (1.0 / n - 1.0 / n ** 2) / np.log(n)   # theoretische Schwelle
print(f"D) Theoretische Schwelle eps_c = {eps_c:.5f} (n={n}, c={c})")

rows = []
for eps_rel in np.logspace(-3, 0, 13):
    eps = eps_rel * np.median(DXs)
    # Multi-Restart: Produkt-Init + B zufaellige Inits (GPU), bestes L_eps gewinnen.
    tDX = torch.tensor(DXs, dtype=torch.float64, device=dev)
    tmu = torch.tensor(mu_np, dtype=torch.float64, device=dev)
    B = 64
    G0 = torch.softmax(8 * torch.randn(B, n, n, generator=torch.Generator(device=dev),
                                      device=dev, dtype=torch.float64), dim=-1)
    Gb, _ = my.entropic_gw_batch(tDX, tDX, tmu, tmu, eps, n_outer=300,
                                 g0_batch=G0, n_sinkhorn=500)
    g_np = Gb.cpu().numpy()
    prod = mu_np[:, None] * mu_np[None, :]
    L_rand = [my.entropic_objective(g_np[b], DXs, DXs, eps) for b in range(B)]
    L_prod = my.entropic_objective(prod, DXs, DXs, eps)
    g = g_np[int(np.argmin(L_rand))] if min(L_rand) < L_prod else prod
    tv = 0.5 * float(np.abs(g - prod).sum())
    am = g.argmax(1)
    is_perm = len(set(am)) == n
    L = my.entropic_objective(g, DXs, DXs, eps)
    rows.append({"eps_rel": float(eps_rel), "eps_abs": float(eps), "TV_to_product": tv,
                 "argmax_is_perm": bool(is_perm),
                 "H_gamma": float(-(g * np.log(np.maximum(g, 1e-300))).sum()),
                 "L_eps": L, "gw": my.gw_cost(g, DXs, DXs),
                 "L_best_random_init": float(min(L_rand)),
                 "L_product_init": float(L_prod)})
    print(f"D) eps_rel={eps_rel:.4f}: TV(g*, Produkt)={tv:.4f}, argmax-Perm={is_perm}, "
          f"L_best={L:.4f} (uniform: {L_prod:.4f}, Theorie-Grenze eps_c={eps_c:.4f}), "
          f"H={rows[-1]['H_gamma']:.3f}")
RES["D_simplex_phase_transition"] = {"eps_c_theory": float(eps_c), "rows": rows}

# ------------------------------------------------------------------- E) Quadrat-Orbit
# Quadrat-Ecken in 2D: Selbst-Isometriegruppe D4 (8 Elemente); mit GW-Sicht nur die
# Distanzerhaltenden Permutationen (C4-Rotationen + Spiegelungen = 8) sind GW-optimal.
th = np.array([0, 1, 2, 3]) * np.pi / 2
Xq = np.stack([np.cos(th), np.sin(th)], axis=1)
DXq = np.sqrt(((Xq[:, None] - Xq[None]) ** 2).sum(-1)) ** 2
m = 4
muh = np.full(m, 1.0 / m)
orbit = []
import itertools
for p in itertools.permutations(range(4)):
    Yp = Xq[list(p)]
    DYp = np.sqrt(((Yp[:, None] - Yp[None]) ** 2).sum(-1)) ** 2
    if np.allclose(DXq, DYp):
        orbit.append(p)
print(f"E) Selbst-Isometrie-Orbit des Quadrats: {len(orbit)} Elemente (Theorie D4: 8)")

eps_rel = 0.05
eps = eps_rel * np.median(DXq)
sols = []
B = 64
tDX = torch.tensor(DXq, dtype=torch.float64, device=dev)
tmu = torch.full((m,), 1.0 / m, dtype=torch.float64, device=dev)
G0 = torch.softmax(8 * torch.randn(B, m, m, generator=torch.Generator(device=dev),
                                   device=dev, dtype=torch.float64), dim=-1)
Gb, _ = my.entropic_gw_batch(tDX, tDX, tmu, tmu, eps, n_outer=300, g0_batch=G0, n_sinkhorn=500)
ams = [tuple(x) for x in Gb.argmax(dim=2).cpu().numpy()]
from collections import Counter
hist = Counter(ams)
sym_mean = np.zeros((m, m))
for p in orbit:
    P = np.zeros((m, m)); P[range(m), p] = 1.0 / m  # P[i, p[i]] => Zeile i -> Spalte p[i]
    # Kopplungsrichtung i -> orbit-p: p als Abbildung i->p[i]; Yp=X[p[i]] => Y_{p[i]}=X_i korrekt
    sym_mean += P / len(orbit)
tvs = [0.5 * float(np.abs(Gb[b].cpu().numpy() - sym_mean).sum()) for b in range(B)]
gw_vals = []
for b in range(B):
    Gb_np = Gb[b].cpu().numpy()
    gw_vals.append(my.gw_cost(Gb_np, DXq, DXq))
RES["E_square_orbit"] = {
    "orbit_size": len(orbit), "eps_rel": eps_rel,
    "argmax_histogram": {str(k): int(v) for k, v in hist.most_common()},
    "TV_to_symmetric_mean_mean": float(np.mean(tvs)),
    "TV_to_symmetric_mean_max": float(np.max(tvs)),
    "gw_final_mean": float(np.mean(gw_vals)), "gw_final_std": float(np.std(gw_vals)),
}
print(f"E) eps_rel={eps_rel}: TV zum symmetrischen Mittel mean={np.mean(tvs):.3f} "
      f"(nicht 0 => Plan konvergiert NICHT zum Orbit-Mittel), "
      f"GW final mean={np.mean(gw_vals):.2e} std={np.std(gw_vals):.1e}")
print("   Histogramm:", dict(hist.most_common()))

# ------------------------------------------------------------------- F) Distortion
def circle(n=64, rot=True, noise=0.0, seed=5):
    rngl = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rngl.normal(0, 0.01, n)
    X = np.stack([np.cos(t), np.sin(t)], 1)
    if rot:
        t2 = t + 0.7
    else:
        t2 = t
    Y = np.stack([np.cos(t2), np.sin(t2)], 1)
    if noise > 0:
        Y = Y + rngl.normal(0, noise, Y.shape)
        t2 = t2 + rngl.normal(0, 0.02, n)          # GT-Korrespondenz leicht verrauscht? -> nein:
    return X, Y, t, t2

def distortion_metrics(DXn, DYn, G, eps):
    n, m = G.shape
    mux = G.sum(1); muy = G.sum(0)
    T = (G / mux[:, None]) @ np.diag(np.arange(m)) @ np.ones((m, 1))  # baryzentrisch auf Indizes
    # Indizes auf dem Kreis: y_j hat Winkel ~ 2 pi j / m
    # Distortion: ||T(x_i)-T(x_k)||_index vs |i-k| (zyklisch) -> relative mittlere Abweichung
    idx_x = np.arange(n)
    d_src = np.abs(((idx_x[:, None] - idx_x[None, :] + n / 2) % n) - n / 2)
    d_tgt = np.abs(((T[:, None] - T[None, :] + m / 2) % m) - m / 2)
    mask = d_src > 2
    rel = np.abs(d_tgt[mask] - d_src[mask] * (m / n)) / np.maximum(d_src[mask], 1)
    bar_dist = float(np.mean(rel))
    # Rounding:
    am = G.argmax(1)
    perm_ok = len(set(am)) == n
    d_tgt_r = np.abs(((am[:, None] - am[None, :] + m / 2) % m) - m / 2)
    rel_r = np.abs(d_tgt_r[mask] - d_src[mask] * (m / n)) / np.maximum(d_src[mask], 1)
    rnd_dist = float(np.mean(rel_r))
    return bar_dist, rnd_dist, perm_ok

res_f = {}
for noise in [0.0, 0.05, 0.15]:
    X, Y, t, t2 = circle(noise=noise)
    n_pts = len(X)
    DXc = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    DYc = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))
    muc = np.full(n_pts, 1.0 / n_pts)
    medc = np.median(DXc ** 2)
    for eps_rel in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        eps = eps_rel * medc
        g, _ = my.entropic_gw(DXc, DYc, muc, muc, eps, n_outer=300, n_sinkhorn=800)
        bd, rd, pk = distortion_metrics(DXc, DYc, g, eps)
        res_f[f"noise={noise},eps_rel={eps_rel}"] = {
            "barycentric_rel_distortion": bd, "rounding_rel_distortion": rd,
            "rounding_is_perm": pk}
        print(f"F) noise={noise:.2f} eps_rel={eps_rel:.3f}: baryzentrisch={bd:.4f} "
              f"rounding={rd:.4f} perm={pk}")
RES["F_distortion"] = res_f

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e2_obstruction.json"
out.write_text(json.dumps(RES, indent=1, default=float))
print("gespeichert ->", out)

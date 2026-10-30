"""E1: Zyklus 91 nachmessen — Nicht-Konvexität, Marginal-Konvexität, Fixpunkte, Basins.

A) Exakte 2x2-Landschaft: volle Parametrisierung des Kopplungs-Polytops,
   Konvexitätsverletzungen entlang Zufallssegenenten.
B) Zyklus-91-Behauptung (i): 'konvex in jeder Marginal einzeln' — Slice-Test.
C) Basins/Gauge-Mannigfaltigkeit: B parallele Zufalls-Inits (GPU) auf symmetrischem
   Problem; Verteilung der Finalwerte, Permutationen der argmax-Rundung, Spreizung.
"""
import sys, json, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my

rng = np.random.default_rng(42)
RES = {}

# ---------------------------------------------------------------- A) 2x2 exakt
# Geschlossene Form mit DX = DY = [[0,1],[1,0]]:
#   GW(g) = mu0^2+mu1^2 + nu0^2+nu1^2 - 4(a*d + b*c),  g=(a,b,c,d)=(g11,g12,g21,g22)
#   KL(g||mu(x)nu) = sum_ij g_ij log(g_ij / (mu_i nu_j))
def L2x2(p, eps):
    a, b, d = p[..., 0], p[..., 1], p[..., 2]
    c = 1.0 - a - b - d
    mu0, mu1 = a + b, c + d
    nu0, nu1 = a + c, b + d
    gw = mu0 ** 2 + mu1 ** 2 + nu0 ** 2 + nu1 ** 2 - 4.0 * (a * d + b * c)
    g = np.stack([a, b, c, d], axis=-1)
    prod = np.stack([mu0 * nu0, mu0 * nu1, mu1 * nu0, mu1 * nu1], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(g > 1e-300, g * np.log(np.maximum(g, 1e-300) / np.maximum(prod, 1e-300)), 0.0)
    return gw + eps * kl.sum(axis=-1)

res = {}
u = np.linspace(0.0, 1.0, 400)
UU, VV, WW = np.meshgrid(u, u, u, indexing="ij")
mask = (UU + VV + WW) <= 1.0
pts = np.stack([UU[mask], VV[mask], WW[mask]], axis=1)
for eps in [0.0, 0.05, 0.2, 1.0]:
    viol = 0
    tot = 100000
    idx = rng.integers(0, len(pts), size=(tot, 3))
    P = pts[idx]                                          # (tot,3,3) Eckpunkte
    w = rng.dirichlet(np.ones(3), size=tot)               # (tot,3)
    mid = (w[:, :, None] * P).sum(axis=1)                 # gewichteter Mittelwert
    mix = (w[:, None, :] * P).sum(axis=1)                 # Mittel der Funktionswerte
    f_mid = L2x2(mid, eps)
    f_mix = L2x2(mix, eps)
    violations = (f_mid > f_mix + 1e-12).mean()
    Lfull = L2x2(pts, eps)
    res[f"eps={eps}"] = {
        "convexity_violation_rate": float(violations),
        "max_violation": float(np.max(f_mid - f_mix)),
        "argmin_L": float(Lfull.min()),
        "argmin_at_g11_g12_g22": [float(x) for x in pts[Lfull.argmin()]],
    }
    print(f"A) eps={eps}: Konvexitätsverletzungsrate={violations:.4f}, "
          f"max Violation={np.max(f_mid - f_mix):.3e}, "
          f"argmin bei (g11,g12,g22)={np.round(pts[Lfull.argmin()],4)}")
RES["A_2x2"] = res

# ---------------------------------------------------------------- B) Slice-Konvexität
def slice_convexity(n=8, m=8, trials=4000, eps=0.1, scale=1.0):
    """Zyklus-91-Behauptung (i): L_eps konvex, wenn eine Row-Marginal fixiert ist?"""
    A = rng.normal(size=(n, 2)); Bp = rng.normal(size=(m, 2)) + 0.5
    DX = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1))
    DY = np.sqrt(((Bp[:, None] - Bp[None]) ** 2).sum(-1)) * scale
    viol = 0; tot = 0
    maxviol = 0.0
    for _ in range(trials):
        mu = rng.dirichlet(np.ones(n))                    # fixierte Row-Marginal
        nu1 = rng.dirichlet(np.ones(m)); nu2 = rng.dirichlet(np.ones(m))
        g1 = mu[:, None] * nu1[None, :]
        g2 = mu[:, None] * nu2[None, :]
        for t in rng.uniform(0.2, 0.8, size=4):
            gm = t * g1 + (1 - t) * g2
            f1 = my.entropic_objective(g1, DX, DY, eps)
            f2 = my.entropic_objective(g2, DX, DY, eps)
            fm = my.entropic_objective(gm, DX, DY, eps)
            lin = t * f1 + (1 - t) * f2
            tot += 1
            if fm > lin + 1e-10:
                viol += 1
                maxviol = max(maxviol, (fm - lin) / max(abs(lin), 1e-12))
    return viol / tot, maxviol

rate, maxv = slice_convexity()
RES["B_slice_convexity_fixed_row_marginal"] = {"violation_rate": rate, "max_rel_violation": maxv}
print(f"B) Slice-Konvexität (Row-Marginal fix): Verletzungsrate={rate:.3f}, max rel={maxv:.2e}")

# ---------------------------------------------------------------- C) Basins / Gauge-Mannigfaltigkeit (GPU)
def basins(n=16, eps_rel=0.1, B=256, n_outer=120, seed=7):
    dev = "cuda"
    tg = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1)).numpy()
    Y = X[torch.from_numpy(perm)]                          # isometrisch (Permutation)
    DX = torch.cdist(X, X) ** 2
    DY = torch.cdist(Y, Y) ** 2
    mu = torch.full((n,), 1.0 / n, dtype=DX.dtype)
    nu = torch.full((n,), 1.0 / n, dtype=DY.dtype)
    DX, DY, mu, nu = DX.to(dev), DY.to(dev), mu.to(dev), nu.to(dev)
    eps = eps_rel * float(DX.median())
    G0 = torch.softmax(6 * torch.randn(B, n, n, generator=tg, device=dev,
                                       dtype=torch.float64), dim=-1)
    Gb, vals = my.entropic_gw_batch(DX, DY, mu, nu, eps, n_outer=n_outer,
                                    g0_batch=G0, n_sinkhorn=300, device=dev)
    final = vals[:, -1]                                    # (B,) GW-Werte
    # argmax-Rundung -> Permutation je Init
    am = Gb.argmax(dim=2).cpu().numpy()                    # (B,n)
    perms = set()
    for row in am:
        if len(set(row)) == n:
            perms.add(tuple(row))
    # Spreizung der Loesungen (Frobenius-Abstand zum Batch-Mittel)
    Gc = Gb / Gb.sum(dim=(1, 2), keepdim=True)
    spread = (Gc - Gc.mean(dim=0, keepdim=True)).norm(dim=(1, 2))
    marg_err = float(((Gb.sum(2) - mu[None, :]).abs().max() + (Gb.sum(1) - nu[None, :]).abs().max()))
    return {
        "n_inits": B, "eps_rel": eps_rel,
        "gw_final_min": float(final.min()), "gw_final_max": float(final.max()),
        "gw_final_std": float(final.std()),
        "marginal_max_err": marg_err,
        "n_distinct_permutations": len(perms),
        "n_valid_permutations_found": int(sum(1 for row in am if len(set(row)) == n)),
        "solution_spread_mean": float(spread.mean()),
        "solution_spread_max": float(spread.max()),
    }

out = {}
for eps_rel in [0.5, 0.2, 0.1, 0.03, 0.01]:
    r = basins(eps_rel=eps_rel)
    out[f"eps_rel={eps_rel}"] = r
    print(f"C) eps_rel={eps_rel}: GW[min/max]={r['gw_final_min']:.4f}/{r['gw_final_max']:.4f} "
          f"std={r['gw_final_std']:.1e} | distinkte Permutationen={r['n_distinct_permutations']} "
          f"(von {r['n_valid_permutations_found']} gueltigen) | Spreizung={r['solution_spread_mean']:.3f}")
RES["C_basins"] = out

out_path = pathlib.Path(__file__).resolve().parent.parent / "results" / "e1_cycle91.json"
out_path.write_text(json.dumps(RES, indent=1))
print("gespeichert ->", out_path)

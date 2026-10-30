"""E1b: Gauge-Orbit vs. echte lokale Minima in den Basin-Solutionen.

Frage: Sind die gefundenen Permutationen (a) alle gleichoptimal (Gauge-Orbit /
Selbst-Isometrien) oder (b) unterschiedlich schlecht (spurious local minima)?
Zusätzlich: Gauge-Freiheitsgrad-Claim von Zyklus 91 analytisch pruefen:
alpha_i*beta_j-Reskalierung von gamma veraendert die Marginals => KEINE Symmetrie
des balancierten Problems. Messbar: marginals nach Reskalierung.
"""
import sys, json, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my

dev = "cuda"
n = 16
B = 512
seed = 11
tg = torch.Generator(device=dev).manual_seed(seed)

X = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
gt_perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1)).numpy()
gt_perm_inv = np.argsort(gt_perm)          # Y = X[gt_perm] => Isometrie i -> gt_perm^{-1}(i)
Y = X[torch.from_numpy(gt_perm)]
DX = torch.cdist(X, X) ** 2
DY = torch.cdist(Y, Y) ** 2
mu = torch.full((n,), 1.0 / n, dtype=torch.float64)
nu = torch.full((n,), 1.0 / n, dtype=torch.float64)
DX, DY, mu, nu = DX.to(dev), DY.to(dev), mu.to(dev), nu.to(dev)

RES = {}
for eps_rel in [0.1, 0.03, 0.01]:
    eps = eps_rel * float(DX.median())
    G0 = torch.softmax(6 * torch.randn(B, n, n, generator=tg, device=dev, dtype=torch.float64), dim=-1)
    Gb, vals = my.entropic_gw_batch(DX, DY, mu, nu, eps, n_outer=200,
                                    g0_batch=G0, n_sinkhorn=300)
    final = vals[:, -1]

    # 1) Cluster der Solver-Loesungen nach argmax-Permutation; GW je Cluster
    am = Gb.argmax(dim=2).cpu().numpy()
    perm_to_gw = {}
    for b in range(B):
        row = tuple(am[b])
        if len(set(row)) == n:
            perm_to_gw.setdefault(row, []).append(float(final[b]))
    cluster_stats = [
        {"perm_equal_gt": tuple(p) == tuple(gt_perm_inv),
         "gw_mean": float(np.mean(v)), "gw_std": float(np.std(v)), "count": len(v)}
        for p, v in sorted(perm_to_gw.items(), key=lambda kv: -len(kv[1]))
    ]

    # 2) GW-Wert der puren Ground-Truth-Permutationskopplung (Referenz):
    def gw_of_perm(perm):
        # perm = Abbildung Zeile i -> Spalte perm[i]; P[arange, perm] = 1/n
        P = torch.zeros(n, n, dtype=torch.float64, device=dev)
        P[torch.arange(n, device=dev), torch.from_numpy(np.asarray(perm)).to(dev)] = 1.0 / n
        DX2, DY2 = DX ** 2, DY ** 2
        return float(((mu @ DX2 @ mu) + (nu @ DY2 @ nu) - 2 * ((DX @ P @ DY) * P).sum()))

    gw_gt = gw_of_perm(gt_perm_inv)

    # 3) Gauge-Reskalierung alpha_i beta_j: Marginals danach (Zyklus-91-Check)
    g_star = Gb[int(final.argmin())].clone()
    alpha = torch.rand(n, dtype=torch.float64, device=dev) + 0.5
    beta = torch.rand(n, dtype=torch.float64, device=dev) + 0.5
    g_rescaled = g_star * alpha[:, None] * beta[None, :]
    marg_before = float((g_star.sum(1) - mu).abs().max() + (g_star.sum(0) - nu).abs().max())
    marg_after = float((g_rescaled.sum(1) - mu).abs().max() + (g_rescaled.sum(0) - nu).abs().max())
    gw_before = float(((DX @ g_star @ DY) * g_star).sum())
    gw_after = float(((DX @ g_rescaled @ DY) * g_rescaled).sum())

    RES[f"eps_rel={eps_rel}"] = {
        "n_inits": B,
        "solver_gw_min": float(final.min()),
        "n_valid_roundings": int(sum(1 for b in range(B) if len(set(am[b])) == n)),
        "n_distinct_permutations": len(cluster_stats),
        "clusters": cluster_stats[:8],
        "gw_pure_gt_perm": gw_gt,
        "gt_perm_recovered_rate": float(np.mean([
            tuple(am[b]) == tuple(gt_perm_inv) for b in range(B) if len(set(am[b])) == n
        ])) if any(len(set(am[b])) == n for b in range(B)) else 0.0,
        "gauge_rescaling_marginal_before": marg_before,
        "gauge_rescaling_marginal_after": marg_after,
        "gauge_rescaling_gw_change": gw_after - gw_before,
    }
    r = RES[f"eps_rel={eps_rel}"]
    print(f"eps_rel={eps_rel}: solver GW_min={r['solver_gw_min']:.5f}, "
          f"GW(pure GT-perm)={gw_gt:.5f}, GT-Permutation wiedergefunden="
          f"{r['gt_perm_recovered_rate']:.2%}, distinkte Permutationen={r['n_distinct_permutations']}")
    for c in r["clusters"][:5]:
        print(f"    cluster: count={c['count']:4d} gw_mean={c['gw_mean']:.5f} "
              f"gw_pure~{'GT' if c['perm_equal_gt'] else 'spurious'}")
    print(f"    Gauge-Reskalierung alpha*beta: marg_err {marg_before:.1e} -> {marg_after:.2f} "
          f"(GW-Änderung {gw_after - gw_before:+.4f})  => Marginals zerstoert: "
          f"{'JA' if marg_after > 1e-6 else 'nein'}")

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e1b_gauge_vs_minima.json"
out.write_text(json.dumps(RES, indent=1, default=float))
print("gespeichert ->", out)

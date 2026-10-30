"""E0: Validierung des eigenen entropischen GW-Solvers.

(1) Objektivformel gegen O(n^4)-Brute-Force
(2) Agreement mit POT entropic_gromov_wasserstein (PGD + PPA), mit Marginal-Check
(3) Konvergenz gegen POT exakten GW (unregularisiert) fuer eps -> 0
"""
import sys, json, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my
import ot

rng = np.random.default_rng(0)
res = {"objective_bf_maxerr": 0.0, "pot_agreement": [], "exact_limit": []}

# ---------- (1) Brute force
for _ in range(20):
    n = m = 5
    DX = rng.uniform(0.1, 2.0, (n, n)); DX = (DX + DX.T) / 2; np.fill_diagonal(DX, 0)
    DY = rng.uniform(0.1, 2.0, (m, m)); DY = (DY + DY.T) / 2; np.fill_diagonal(DY, 0)
    G = rng.dirichlet(np.ones(n * m)).reshape(n, m)
    fast = my.gw_cost(G, DX, DY)
    slow = sum(
        G[i, j] * G[k, l] * (DX[i, k] - DY[j, l]) ** 2
        for i in range(n) for k in range(n) for j in range(m) for l in range(m)
    )
    res["objective_bf_maxerr"] = max(res["objective_bf_maxerr"], abs(fast - slow))
print(f"(1) Objektiv vs BruteForce: max|err| = {res['objective_bf_maxerr']:.2e}")

# ---------- (2) Agreement mit POT (entropisch)
def marg_violation(G, mu, nu):
    return float(np.abs(G.sum(1) - mu).max() + np.abs(G.sum(0) - nu).max())

for trial in range(6):
    n = int(rng.integers(10, 22)); m = int(rng.integers(10, 22))
    A = rng.normal(size=(n, 2)); B = rng.normal(size=(m, 2)) + 0.5
    DX = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    DY = np.sqrt(((B[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    mu = np.full(n, 1.0 / n); nu = np.full(m, 1.0 / m)
    eps = 0.02 * float(np.median(DX)) ** 2

    G_mine, _ = my.entropic_gw(DX, DY, mu, nu, eps, n_outer=400)
    entry = {"n": n, "m": m,
             "gw_mine": my.gw_cost(G_mine, DX, DY),
             "Leps_mine": my.entropic_objective(G_mine, DX, DY, eps),
             "marg_mine": marg_violation(G_mine, mu, nu)}

    for solver, numit in [("PPA", 5000), ("PGD", 5000)]:
        G_pot = ot.gromov.entropic_gromov_wasserstein(
            DX, DY, mu, nu, "square_loss", epsilon=eps, solver=solver,
            max_iter=400, tol=1e-12, numItermax=numit, stopThr=1e-12)
        entry[f"gw_pot_{solver}"] = my.gw_cost(G_pot, DX, DY)
        entry[f"Leps_pot_{solver}"] = my.entropic_objective(G_pot, DX, DY, eps)
        entry[f"marg_pot_{solver}"] = marg_violation(G_pot, mu, nu)
    res["pot_agreement"].append(entry)
    print(f"(2) n={n:2d} m={m:2d} Leps mine={entry['Leps_mine']:.5f}  "
          f"PPA={entry['Leps_pot_PPA']:.5f} (marg {entry['marg_pot_PPA']:.1e})  "
          f"PGD={entry['Leps_pot_PGD']:.5f} (marg {entry['marg_pot_PGD']:.1e})")

# ---------- (3) Grenzwert eps -> 0 gegen exakten GW
for trial in range(3):
    n = m = 12
    A = rng.normal(size=(n, 2))
    th = rng.uniform(0, np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    B = A @ R
    DX = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    DY = np.sqrt(((B[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    mu = np.full(n, 1.0 / n); nu = np.full(m, 1.0 / m)
    T_ex = ot.gromov.gromov_wasserstein(DX, DY, mu, nu, "square_loss",
                                        max_iter=1000, tol=1e-12, log=True)[0]
    gw_ex = my.gw_cost(T_ex, DX, DY)
    row = {"gw_exact": gw_ex}
    for eps in [1e-2, 1e-3, 1e-4]:
        G, _ = my.entropic_gw(DX, DY, mu, nu, eps, n_outer=2000)
        row[f"gw_eps{eps:g}"] = my.gw_cost(G, DX, DY)
    res["exact_limit"].append(row)
    print(f"(3) GW_exact={gw_ex:.6f}  entropisch: " +
          "  ".join(f"eps={k.split('eps')[1]}: {v:.6f}" for k, v in row.items() if k != "gw_exact"))

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e0_validation.json"
out.write_text(json.dumps(res, indent=1))
print("gespeichert ->", out)

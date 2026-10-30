"""E3 (v2): Zero-Shot-Cross-Architecture-Transfer mit DISJUNKTEN Probe-Mengen.

A (h=16) und B (h=32) sehen nie dieselben Probe-Punkte (unabhaengige Samples).
GW-Kopplung zwischen den Hidden-Wolken. Metriken:
 (a) Label-Transfer via GW-Korrespondenz (argmax) vs wahre Labels von B
     Baselines: Zufallsperm., Produktkopplung, Oracle (LSA auf Label-Match)
 (b) Positionstransfer via Rounding: B_j erhaelt A-Koordinaten des GW-Partners
     -> linearer Probe A angewandt -> Accuracy vs B's wahren Labels
 (c) Baryzentrische Projektion B->A (Vergleich, erwartet schlecht)
"""
import sys, json, pathlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my

dev = "cuda"
RES = {}


def two_moons(n=2000, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    x1 = np.stack([1 - np.cos(t1), np.sin(t1) * 0.6 - 0.25], 1) + rng.normal(0, noise, (n1, 2))
    x2 = np.stack([1 + np.cos(t2), np.sin(t2) * 0.6 + 0.25], 1) + rng.normal(0, noise, (n - n1, 2))
    X = np.concatenate([x1, x2]); y = np.concatenate([np.zeros(n1, int), np.ones(n - n1, int)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


X_all, y_all = two_moons()
mu_, sd_ = X_all.mean(0), X_all.std(0)
X_all = (X_all - mu_) / sd_
rng = np.random.default_rng(1)
perm = rng.permutation(len(X_all))
X_tr, y_tr = X_all[perm[:1800]], y_all[perm[:1800]]
X_te, y_te = X_all[perm[1800:]], y_all[perm[1800:]]
X_tr = torch.tensor(X_tr, dtype=torch.float32, device=dev)
y_tr = torch.tensor(y_tr, dtype=torch.long, device=dev)
X_te = torch.tensor(X_te, dtype=torch.float32, device=dev)
y_te = torch.tensor(y_te, dtype=torch.long, device=dev)


def probe(n, seed):
    Xp, yp = two_moons(n=n, noise=0.15, seed=seed)
    Xp = (Xp - mu_) / sd_
    return torch.tensor(Xp, dtype=torch.float32, device=dev), torch.tensor(yp, device=dev)


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)

    def hidden(self, x):
        return torch.relu(self.net[2](torch.relu(self.net[0](x))))


def train(net, steps=2500, lr=1e-2):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(X_tr), (256,), device=dev)
        loss = nn.functional.cross_entropy(net(X_tr[idx]), y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


torch.manual_seed(0); netA = train(MLP(16).to(dev))
torch.manual_seed(1); netB = train(MLP(32).to(dev))

XpA, yA = probe(256, seed=123)          # Probe-Menge A
XpB, yB = probe(256, seed=456)          # DISJUNkte Probe-Menge B
with torch.no_grad():
    accA = (netA(X_te).argmax(1) == y_te).float().mean().item()
    accB = (netB(X_te).argmax(1) == y_te).float().mean().item()
    HA, HB = netA.hidden(XpA), netB.hidden(XpB)
    HAt = netA.hidden(X_tr)
print(f"Netz A (h=16): Test-Acc {accA:.3f} | Netz B (h=32): Test-Acc {accB:.3f}")
RES["model_acc"] = {"A": accA, "B": accB}


def zc(H):
    return (H - H.mean(0)) / (H.std(0) + 1e-8)


HAz, HBz = zc(HA), zc(HB)
n = 256
DXh = torch.cdist(HAz, HAz) ** 2
DYh = torch.cdist(HBz, HBz) ** 2
muh = torch.full((n,), 1.0 / n, dtype=torch.float64, device=dev)
DXh, DYh = DXh.double(), DYh.double()
medh = float(DXh.median())

# Linearer Probe auf A-Raum (Train-Hidden + Train-Labels)
clf = nn.Linear(16, 2).to(dev)
opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
for _ in range(1500):
    idx = torch.randint(0, len(HAt), (256,), device=dev)
    loss = nn.functional.cross_entropy(clf(HAt[idx]), y_tr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    probe_ref_A = (clf(HAz).argmax(1) == yA).float().mean().item()   # Referenz (kein Transfer)
print(f"Probe-Referenz (A-Raum, A-Punkte): {probe_ref_A:.3f}")

predA = clf(HAz).argmax(1).cpu().numpy()          # A-Probe-Praediktionen
from scipy.optimize import linear_sum_assignment
rows = []
yA_np, yB_np = yA.cpu().numpy(), yB.cpu().numpy()

for eps_rel in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    eps = eps_rel * medh
    Gb, _ = my.entropic_gw_batch(DXh, DYh, muh, muh, eps, n_outer=250,
                                 g0_batch=torch.softmax(torch.randn(32, n, n,
                                     generator=torch.Generator(device=dev), device=dev,
                                     dtype=torch.float64) * 4, dim=-1), n_sinkhorn=400)
    Gb_np = Gb.cpu().numpy()
    Ls = [my.entropic_objective(Gb_np[b], DXh.cpu().numpy(), DYh.cpu().numpy(), eps) for b in range(32)]
    G = Gb_np[int(np.argmin(Ls))]

    colA = G.argmax(0)                    # B_j -> A_i (GW-Partner)
    # Automorphismus-Diagnostik: Fraktion gleicher WAHRER Labels unter GW-Partnern
    same_true = float(np.mean(yA_np[colA] == yB_np))
    # GW-Kosten der zwei harten Korrespondenz-Varianten (Label-Cluster-Swap):
    # Variante 1: GW-Korrespondenz; Variante 2: gleiche Korrespondenz, aber Klassen getauscht
    # (permutiere die A-Seite innerhalb der Label-Gruppen)
    import itertools
    a_idx0 = np.where(yA_np == 0)[0]; a_idx1 = np.where(yA_np == 1)[0]
    G_swap = G.copy()
    G_swap[a_idx0] = G[a_idx1]
    G_swap[a_idx1] = G[a_idx0]
    DXn, DYn = DXh.cpu().numpy(), DYh.cpu().numpy()
    gw_orig = my.gw_cost(G, DXn, DYn)
    gw_swap = my.gw_cost(G_swap, DXn, DYn)
    # (a) Label-Transfer: predA(colA[j]) vs wahres Label von B_j
    acc_lbl = float(np.mean(predA[colA] == yB_np))
    acc_lbl_rand = float(np.mean(predA[np.random.permutation(n)] == yB_np))
    prod = np.outer(np.full(n, 1 / n), np.full(n, 1 / n))
    acc_lbl_prod = float(np.mean(predA[prod.argmax(0)] == yB_np))
    cost = (predA[:, None] != yB_np[None, :]).astype(float)
    ri, ci = linear_sum_assignment(cost)
    acc_lbl_oracle = 1 - cost[ri, ci].mean()
    # (b) Positionstransfer via Rounding: B_j -> A-Koordinaten des Partners
    pos_r = HAz[torch.tensor(colA)].float()
    with torch.no_grad():
        acc_pos = (clf(pos_r).argmax(1) == yB).float().mean().item()
    # (c) Baryzentrische Projektion B->A
    proj = (G.T / (G.sum(0)[:, None] + 1e-12)) @ HAz.double().cpu().numpy()
    with torch.no_grad():
        acc_bary = (clf(torch.tensor(proj, dtype=torch.float32, device=dev)).argmax(1) == yB).float().mean().item()

    rows.append({
        "eps_rel": eps_rel,
        "same_true_label_fraction": same_true,
        "gw_cost_original": gw_orig,
        "gw_cost_cluster_swapped": gw_swap,
        "label_transfer_gw": acc_lbl,
        "label_transfer_random": acc_lbl_rand,
        "label_transfer_product": acc_lbl_prod,
        "label_transfer_oracle_lsa": float(acc_lbl_oracle),
        "position_transfer_rounding_acc": acc_pos,
        "barycentric_projection_acc": acc_bary,
        "gw_value": my.gw_cost(G, DXn, DYn),
    })
    print(f"E3v2) eps_rel={eps_rel:.3f}: Label-GW={acc_lbl:.3f} (rand {acc_lbl_rand:.3f}, "
          f"prod {acc_lbl_prod:.3f}, oracle {acc_lbl_oracle:.3f}) | "
          f"Pos-Rounding={acc_pos:.3f} | Bary-Proj={acc_bary:.3f} | "
          f"gleiche-wahre-Labels={same_true:.3f} | GW(orig)={gw_orig:.4f} GW(swap)={gw_swap:.4f}")

RES["transfer"] = {"probe_reference_A": probe_ref_A, "rows": rows}
out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e3_transfer.json"
out.write_text(json.dumps(RES, indent=1, default=float))
print("gespeichert ->", out)

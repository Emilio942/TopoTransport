"""E3b: Kontrolle — Architektur-spezifische Metrik vs Solver-Artefakt.

Vergleicht GW-Kosten (kleines eps, Multi-Restart):
  1) GW(HA_probe1, HA_probe2)  gleiches Netz A, disjunkte Samples  -> Solvent-Floor
  2) GW(HB_probe1, HB_probe2)  gleiches Netz B
  3) GW(HA_probe1, HB_probe1)  zwischen Architekturen              -> Architektur-Gap
  4) GW-Kosten der inputnaechsten-Nachbar-Korrespondenz (label-consistent, Referenz)
"""
import sys, json, pathlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my

dev = "cuda"


def two_moons(n=2000, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1); t2 = rng.uniform(0, np.pi, n - n1)
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
X_tr = torch.tensor(X_all[perm[:1800]], dtype=torch.float32, device=dev)
y_tr = torch.tensor(y_all[perm[:1800]], dtype=torch.long, device=dev)  # Fix: alignierte Labels


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

Xp1, y1 = probe(256, seed=123)
Xp2, y2 = probe(256, seed=456)
with torch.no_grad():
    print(f"E3b) accA={float((netA(Xp1).argmax(1) == y1).float().mean()):.3f} "
          f"accB={float((netB(Xp2).argmax(1) == y2).float().mean()):.3f}")


def zc(H):
    return ((H - H.mean(0)) / (H.std(0) + 1e-8)).double()


with torch.no_grad():
    HA1, HA2, HB1, HB2 = zc(netA.hidden(Xp1)), zc(netA.hidden(Xp2)), zc(netB.hidden(Xp1)), zc(netB.hidden(Xp2))

n = 256
muh = torch.full((n,), 1.0 / n, dtype=torch.float64, device=dev)


def best_gw(H1, H2, eps_rel=0.02, B=64, n_outer=300):
    DX = torch.cdist(H1, H1) ** 2
    DY = torch.cdist(H2, H2) ** 2
    eps = eps_rel * float(DX.median())
    Gb, _ = my.entropic_gw_batch(DX, DY, muh, muh, eps, n_outer=n_outer,
                                 g0_batch=torch.softmax(torch.randn(B, n, n,
                                     generator=torch.Generator(device=dev), device=dev,
                                     dtype=torch.float64) * 4, dim=-1), n_sinkhorn=400)
    DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()
    Gb_np = Gb.cpu().numpy()
    Ls = [my.entropic_objective(Gb_np[b], DXn, DYn, eps) for b in range(B)]
    G = Gb_np[int(np.argmin(Ls))]
    return my.gw_cost(G, DXn, DYn), G


gw_AA, G_AA = best_gw(HA1, HA2)
gw_BB, G_BB = best_gw(HB1, HB2)
gw_AB, G_AB = best_gw(HA1, HB1)
print(f"GW(A-probe1, A-probe2) = {gw_AA:.3f}   [gleiches Netz, disjunkte Samples]")
print(f"GW(B-probe1, B-probe2) = {gw_BB:.3f}")
print(f"GW(A-probe1, B-probe1) = {gw_AB:.3f}   [zwischen Architekturen]")

# Referenz: input-NN-Korrespondenz (geteilter Input-Raum, label-consistent up to Net accuracy)
X1 = Xp1.cpu().numpy(); X2 = Xp2.cpu().numpy()
d = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
colNN = d.argmin(1)                     # A_i -> naechster B-Punkt im INPUT-Raum


def gw_of_hard(H1, H2, colmap):
    DX = torch.cdist(H1, H1) ** 2
    DY = torch.cdist(H2, H2) ** 2
    P = torch.zeros(n, n, dtype=torch.float64, device=dev)
    P[torch.arange(n, device=dev), torch.tensor(colmap, device=dev)] = 1.0 / n
    DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()
    mu = np.full(n, 1.0 / n); nu = np.full(n, 1.0 / n)
    constA = float(mu @ (DXn ** 2) @ mu); constB = float(nu @ (DYn ** 2) @ nu)
    cross = float(((DXn @ P.cpu().numpy()) * (P.cpu().numpy() @ DYn)).sum())
    return constA + constB - 2 * cross


gw_inputNN_AB = gw_of_hard(HA1, HB1, colNN)
gw_inputNN_AA = gw_of_hard(HA1, HA2, colNN)
print(f"GW-Kosten der input-NN-Korrespondenz: A<->B = {gw_inputNN_AB:.3f}, A<->A' = {gw_inputNN_AA:.3f}")

# Label-Konsistenz der Korrespondenzen (colmap: A_i -> B_j => y1[i] vs y2[colmap[i]])
y1n, y2n = y1.cpu().numpy(), y2.cpu().numpy()
colAB = G_AB.argmax(0)
print(f"Label-Match: input-NN A->B: {np.mean(y1n == y2n[colNN]):.3f} | "
      f"GW A->B: {np.mean(y1n == y2n[colAB]):.3f}")

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e3b_controls.json"
out.write_text(json.dumps({
    "gw_AA_same_net": gw_AA, "gw_BB_same_net": gw_BB, "gw_AB_cross_arch": gw_AB,
    "gw_cost_inputNN_AB": gw_inputNN_AB, "gw_cost_inputNN_AA": gw_inputNN_AA,
    "label_match_inputNN_AB": float(np.mean(y1n == y2n[colNN])),
    "label_match_GW_AB": float(np.mean(y1n == y2n[colAB])),
}, indent=1))
print("gespeichert ->", out)

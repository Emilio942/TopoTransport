"""E3c: Kosten der Label-Konsistenz im Hidden-Raum.

Vergleicht vier harte Korrespondenzen A-probe -> B-probe:
  1) GW-optimales hartes Matching (Hungarian auf GW-Gradient, iteriert)
  2) Label-restriktiertes GW-opt. Matching (LSA je Klasse, gleiche Iteration)
  3) Input-NN-Korrespondenz
  4) GW-Kosten des GW-Solvers (weich, gerundet) — Referenz aus e3b
"""
import sys, json, pathlib
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my

dev = "cuda"
n = 256


def two_moons(nn_pts=2000, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n1 = nn_pts // 2
    t1 = rng.uniform(0, np.pi, n1); t2 = rng.uniform(0, np.pi, nn_pts - n1)
    x1 = np.stack([1 - np.cos(t1), np.sin(t1) * 0.6 - 0.25], 1) + rng.normal(0, noise, (n1, 2))
    x2 = np.stack([1 + np.cos(t2), np.sin(t2) * 0.6 + 0.25], 1) + rng.normal(0, noise, (nn_pts - n1, 2))
    X = np.concatenate([x1, x2]); y = np.concatenate([np.zeros(n1, int), np.ones(nn_pts - n1, int)])
    idx = rng.permutation(nn_pts)
    return X[idx], y[idx]


X_all, y_all = two_moons()
mu_, sd_ = X_all.mean(0), X_all.std(0)
X_all = (X_all - mu_) / sd_
rng = np.random.default_rng(1); perm = rng.permutation(len(X_all))
X_tr = torch.tensor(X_all[perm[:1800]], dtype=torch.float32, device=dev)
y_tr = torch.tensor(y_all[perm[:1800]], dtype=torch.long, device=dev)  # Fix: alignierte Labels


def probe(num, seed):
    Xp, yp = two_moons(nn_pts=num, noise=0.15, seed=seed)
    return torch.tensor((Xp - mu_) / sd_, dtype=torch.float32, device=dev), yp


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
with torch.no_grad():
    _a = float((netA(X_tr).argmax(1) == y_tr).float().mean())
    print(f"E3c) Train-Acc A={_a:.3f}")
Xp1, y1 = probe(256, 123); Xp2, y2 = probe(256, 456)


def zc(H):
    return ((H - H.mean(0)) / (H.std(0) + 1e-8)).double()


with torch.no_grad():
    HA1, HB1 = zc(netA.hidden(Xp1)), zc(netB.hidden(Xp1))

DX = (torch.cdist(HA1, HA1) ** 2).cpu().numpy()
DY = (torch.cdist(HB1, HB1) ** 2).cpu().numpy()
mu = np.full(n, 1.0 / n)
y1n, y2n = np.asarray(y1), np.asarray(y2)


def gw_of_colmap(colmap):
    P = np.zeros((n, n)); P[np.arange(n), colmap] = 1.0 / n
    constA = float(mu @ (DX ** 2) @ mu); constB = float(mu @ (DY ** 2) @ mu)
    cross = float(((DX @ P) * (P @ DY)).sum())
    return constA + constB - 2 * cross


def hard_gw_matching(maskA=None, maskB=None, iters=80, init=None):
    """Iteriertes Hungarian auf dem GW-Gradienten (optional klassenrestriktiert)."""
    ia = np.arange(n) if maskA is None else maskA
    ib = np.arange(n) if maskB is None else maskB
    mA = np.full(len(ia), 1.0 / n)   # Punktegewichte wie im Gesamtproblem (Uniform)
    mB = np.full(len(ib), 1.0 / n)
    DXX, DYY = DX[np.ix_(ia, ia)], DY[np.ix_(ib, ib)]
    a_g = mA @ (DXX ** 2)
    b_g = mB @ (DYY ** 2)
    colk = init if init is not None else np.arange(len(ia)) % len(ib)
    for _ in range(iters):
        P = np.zeros((len(ia), len(ib))); P[np.arange(len(ia)), colk] = 1.0
        C = a_g[:, None] + b_g[None, :] - 2.0 * (DXX @ P @ DYY)
        _, c2 = linear_sum_assignment(C)
        if (c2 == colk).all():
            break
        colk = c2
    out = np.zeros(n, int)
    out[ia] = ib[colk]
    return out


# 1) GW-optimales hartes Matching (ungeschraenkt)
col_gw = hard_gw_matching()
# 2) Label-restriktiert (Klassenmassen bleiben im Problem; Matching nur gleich-label)
col_lab = np.zeros(n, int)
for k in [0, 1]:
    ia = np.where(y1n == k)[0]
    ib = np.where(y2n == k)[0]
    sub = hard_gw_matching(maskA=ia, maskB=ib)
    col_lab[ia] = sub[ia]
# 3) input-NN
X1n, X2n = Xp1.cpu().numpy(), Xp2.cpu().numpy()
d_in = ((X1n[:, None, :] - X2n[None, :, :]) ** 2).sum(-1)
col_in = d_in.argmin(1)

res = {}
for name, col in [("gw_hard_unconstrained", col_gw),
                  ("gw_hard_label_restricted", col_lab),
                  ("input_nn", col_in)]:
    res[name] = {"gw_cost": gw_of_colmap(col),
                 "label_match": float(np.mean(y1n == y2n[col]))}
    print(f"{name}: GW={res[name]['gw_cost']:.1f}  Label-Match={res[name]['label_match']:.3f}")

with open(pathlib.Path(__file__).resolve().parent.parent / "results" / "e3c_label_cost.json", "w") as fh:
    json.dump(res, fh, indent=1)
print("gespeichert -> results/e3c_label_cost.json")

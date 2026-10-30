"""E5: Konstruktives Gegenstück + finale Robustheit.

E5a: Gleich-Architektur-Kontrolle (h=16 vs h=16, verschiedener Seed) + Breitenpaare.
E5b: Zweitdatensatz sklearn-digits (10 Klassen) — Label-Transfer via GW vs Baselines.
E5c: Anker-Rettung: k gelabelte Anker pinnt man in der harten GW-Korrespondenz
     (iteratives Hungarian mit Fixpunkten) -> Label-Transfer vs k.
"""
import sys, json, pathlib
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my

dev = "cuda"
RES = {}
n = 256
muh = torch.full((n,), 1.0 / n, dtype=torch.float64, device=dev)


# ------------------------------------------------------------------ Utilities
def make_dataset(name, seed=0):
    if name == "moons":
        rng = np.random.default_rng(seed)
        n1 = 1000
        t1 = rng.uniform(0, np.pi, n1); t2 = rng.uniform(0, np.pi, 1000)
        x1 = np.stack([1 - np.cos(t1), np.sin(t1) * 0.6 - 0.25], 1) + rng.normal(0, 0.15, (n1, 2))
        x2 = np.stack([1 + np.cos(t2), np.sin(t2) * 0.6 + 0.25], 1) + rng.normal(0, 0.15, (1000, 2))
        X = np.concatenate([x1, x2]).astype(np.float32)
        y = np.concatenate([np.zeros(n1, int), np.ones(1000, int)])
        idx = rng.permutation(len(X))          # WICHTIG: Klassen mischen (Bugfix E5a)
        X, y = X[idx], y[idx]
    elif name == "digits":
        d = load_digits()
        X = d.data.astype(np.float32) / 16.0
        y = d.target.astype(int)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
    mu_, sd_ = X.mean(0), X.std(0) + 1e-8
    return (X - mu_) / sd_, y


class MoonsNet(nn.Module):
    def __init__(self, h, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(2, h)
        self.l2 = nn.Linear(h, h)
        self.out = nn.Linear(h, 2)

    def forward(self, x):
        return self.out(torch.relu(self.l2(torch.relu(self.l1(x)))))

    def hidden(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))


class DigitsNet(nn.Module):
    def __init__(self, h, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(64, h)
        self.l2 = nn.Linear(h, h)
        self.out = nn.Linear(h, 10)

    def forward(self, x):
        return self.out(torch.relu(self.l2(torch.relu(self.l1(x)))))

    def hidden(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))


def train_digits_net(h, seed, X_tr, y_tr, steps=3000, lr=1e-3):
    net = DigitsNet(h, seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(X_tr), (256,), device=dev)
        loss = nn.functional.cross_entropy(net(X_tr[idx]), y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


for h1, h2, s1, s2 in [(16, 16, 0, 1), (16, 16, 0, 5), (16, 32, 0, 1), (64, 8, 0, 1)]:
    def train_mlp(h, seed):
        net = MoonsNet(h, seed).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(2500):
            idx = torch.randint(0, len(X_tr), (256,), device=dev)
            loss = nn.functional.cross_entropy(net(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        return net


def zc(H):
    return ((H - H.mean(0)) / (H.std(0) + 1e-8)).double()


def best_plan(DX, DY, eps, B=32, n_outer=250):
    Gb, _ = my.entropic_gw_batch(DX, DY, muh, muh, eps, n_outer=n_outer,
                                 g0_batch=torch.softmax(torch.randn(B, n, n,
                                     generator=torch.Generator(device=dev), device=dev,
                                     dtype=torch.float64) * 4, dim=-1), n_sinkhorn=400)
    DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()
    Gb_np = Gb.cpu().numpy()
    Ls = [my.entropic_objective(Gb_np[b], DXn, DYn, eps) for b in range(B)]
    return Gb_np[int(np.argmin(Ls))]


def hard_gw_matching(DX, DY, anchors=None, iters=80, init_col=None):
    """Iteratives Hungarian auf GW-Gradient; anchors: Liste (i,j) fixierter Paare."""
    nn_ = DX.shape[0]
    a_g = (np.full(nn_, 1.0 / nn_) @ (DX ** 2))
    b_g = (np.full(nn_, 1.0 / nn_) @ (DY ** 2))
    col = init_col if init_col is not None else np.arange(nn_) % nn_
    free = np.ones(nn_, bool)
    if anchors:
        for i, j in anchors:
            col[i] = j
            free[i] = False
    for _ in range(iters):
        P = np.zeros((nn_, nn_))
        P[np.arange(nn_), col] = 1.0 / nn_
        C = a_g[:, None] + b_g[None, :] - 2.0 * (DX @ P @ DY)
        # Fixpunkte erzwingen (großer Malus für alles andere in Ankerzeilen):
        big = 1e9
        if anchors:
            for i, j in anchors:
                C[i, :] = big
                C[i, j] = 0.0
        ri, ci = linear_sum_assignment(C)
        if (ci == col).all():
            break
        col = ci
        if anchors:
            for i, j in anchors:
                col[i] = j
    return col


# ================================================================== E5a: Gleich-Architektur
rows = []
Xs, ys = make_dataset("moons")
rng = np.random.default_rng(1)
p = rng.permutation(len(Xs))
X_tr = torch.tensor(Xs[p[:1800]], dtype=torch.float32, device=dev)
y_tr = torch.tensor(ys[p[:1800]], dtype=torch.long, device=dev)
Xi, yi = make_dataset("moons", seed=123)
Xi = (Xi - Xs.mean(0)) / (Xs.std(0) + 1e-8)
XpA = torch.tensor(Xi, dtype=torch.float32, device=dev)[:n]
yA = torch.tensor(yi[:n], device=dev)
Xj, yj = make_dataset("moons", seed=456)
Xj = (Xj - Xs.mean(0)) / (Xs.std(0) + 1e-8)
XpB = torch.tensor(Xj, dtype=torch.float32, device=dev)[:n]
yB = torch.tensor(yj[:n], device=dev)
yAn, yBn = yA.cpu().numpy(), yB.cpu().numpy()
d_in = ((XpA.cpu().numpy()[:, None] - XpB.cpu().numpy()[None]) ** 2).sum(-1)
col_in = d_in.argmin(1)
lab_in = float(np.mean(yAn == yBn[col_in]))

for h1, h2, s1, s2 in [(16, 16, 0, 1), (16, 16, 0, 5), (16, 32, 0, 1), (64, 8, 0, 1)]:
    def train_mlp(h, seed):
        net = MoonsNet(h, seed).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(2500):
            idx = torch.randint(0, len(X_tr), (256,), device=dev)
            loss = nn.functional.cross_entropy(net(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        return net
    nA, nB = train_mlp(h1, s1), train_mlp(h2, s2)
    with torch.no_grad():
        HA, HB = zc(nA.hidden(XpA)), zc(nB.hidden(XpB))
    DX, DY = torch.cdist(HA, HA) ** 2, torch.cdist(HB, HB) ** 2
    eps = 0.02 * float(DX.median())
    G = best_plan(DX, DY, eps)
    lab_gw = float(np.mean(yAn == yBn[G.argmax(0)]))
    gw_val = my.gw_cost(G, DX.cpu().numpy(), DY.cpu().numpy())
    rows.append({"h1": h1, "h2": h2, "seed1": s1, "seed2": s2,
                 "label_match_gw": lab_gw, "label_match_inputNN": lab_in,
                 "gw_value": gw_val})
    print(f"E5a) A(h={h1},s={s1}) vs B(h={h2},s={s2}): GW-Label={lab_gw:.3f} "
          f"(inputNN {lab_in:.3f}) GW={gw_val:.2f}")
RES["E5a_same_arch"] = rows

# ================================================================== E5b: digits
dg = load_digits()
Xd, yd = dg.data.astype(np.float32) / 16.0, dg.target.astype(int)
rng = np.random.default_rng(7)
pd_ = rng.permutation(len(Xd))
Xd, yd = Xd[pd_], yd[pd_]
Xtr_d = torch.tensor(Xd[:1400], dtype=torch.float32, device=dev)
ytr_d = torch.tensor(yd[:1400], dtype=torch.long, device=dev)

rngA = np.random.default_rng(100); rngB = np.random.default_rng(200)
iA, iB = rngA.choice(np.arange(1400, len(Xd)), n, replace=False), rngB.choice(np.arange(1400, len(Xd)), n, replace=False)
XpAd = torch.tensor(Xd[iA], dtype=torch.float32, device=dev); yAd = torch.tensor(yd[iA], device=dev)
XpBd = torch.tensor(Xd[iB], dtype=torch.float32, device=dev); yBd = torch.tensor(yd[iB], device=dev)

netAd = train_digits_net(16, 3, Xtr_d, ytr_d)
netBd = train_digits_net(32, 4, Xtr_d, ytr_d)
with torch.no_grad():
    accAd = float((netAd(XpAd).argmax(1) == yAd).float().mean())
    accBd = float((netBd(XpBd).argmax(1) == yBd).float().mean())
    HAd, HBd = zc(netAd.hidden(XpAd)), zc(netBd.hidden(XpBd))
DXd, DYd = torch.cdist(HAd, HAd) ** 2, torch.cdist(HBd, HBd) ** 2
eps_d = 0.02 * float(DXd.median())
Gd = best_plan(DXd, DYd, eps_d)
col_gwd = Gd.argmax(0)
yAn_d, yBn_d = yAd.cpu().numpy(), yBd.cpu().numpy()
din = ((XpAd.cpu().numpy()[:, None] - XpBd.cpu().numpy()[None]) ** 2).sum(-1)
res_b = {
    "accA": accAd, "accB": accBd,
    "label_match_gw": float(np.mean(yAn_d == yBn_d[col_gwd])),
    "label_match_random": float(np.mean(yAn_d == yBn_d[np.random.permutation(n)])),
    "label_match_inputNN": float(np.mean(yAn_d == yBn_d[din.argmin(1)])),
    "gw_value": my.gw_cost(Gd, DXd.cpu().numpy(), DYd.cpu().numpy()),
}
RES["E5b_digits"] = res_b
print(f"E5b) digits: accA={accAd:.3f} accB={accBd:.3f} | GW-Label={res_b['label_match_gw']:.3f} "
      f"(Zufall ~0.1; random {res_b['label_match_random']:.3f}, inputNN {res_b['label_match_inputNN']:.3f})")

# ================================================================== E5c: Anker-Rettung (moons + digits)
def anchor_curve(DXn, DYn, yAn_, yBn_, col_entropic, n_anchor_grid):
    out = []
    rngl = np.random.default_rng(9)
    for k in n_anchor_grid:
        trials = []
        for t in range(8):  # 8 zufaellige Ankerwahl
            if k == 0:
                anchors = []
            else:
                anchors = [(int(i), int(col_entropic[i])) for i in
                           rngl.choice(n, size=min(k, n), replace=False)]
            col = hard_gw_matching(DXn, DYn, anchors=anchors, init_col=col_entropic.copy())
            trials.append(float(np.mean(yAn_ == yBn_[col])))
        out.append({"k": k, "label_match_mean": float(np.mean(trials)),
                    "label_match_min": float(np.min(trials)),
                    "label_match_max": float(np.max(trials))})
    return out

# Moons:
DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()   # aus E5a letztem Paar (16 vs 32)
curve_moons = anchor_curve(DXn, DYn, yAn, yBn, G.argmax(0), [0, 1, 2, 4, 8, 16])
print("E5c) moons:", [(c['k'], round(c['label_match_mean'], 3)) for c in curve_moons])
# digits:
curve_digits = anchor_curve(DXd.cpu().numpy(), DYd.cpu().numpy(), yAn_d, yBn_d, col_gwd,
                            [0, 1, 2, 4, 8, 16])
print("E5c) digits:", [(c['k'], round(c['label_match_mean'], 3)) for c in curve_digits])
RES["E5c_anchor_rescue"] = {"moons": curve_moons, "digits": curve_digits}

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e5_consolidation.json"
out.write_text(json.dumps(RES, indent=1, default=float))
print("gespeichert ->", out)

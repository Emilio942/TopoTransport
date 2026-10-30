"""E5c/d (v2): Anker-Rettung mit ECHTEN Korrespondenzen + Vier-Wege-Vergleich.

Szenario: praktikable 'Anchor' = wenige geteilte Input-Punkte, die durch BEIDE
Netze evaluiert werden (unlabelt!) => wahre Paare (i,i). Diese werden im harten
GW-Matching fixiert. Gemessen: Label-Match auf dem disjunkten Rest.

E5d: Vier-Wege-Vergleich am kanonischen Paar (16,32), moons + digits:
  1) weicher entropischer Plan (argmax-Rundung)
  2) hartes GW-Matching, init = entropische Rundung
  3) hartes GW-Matching, init = input-NN
  4) label-restriktiertes hartes Matching
jeweils Label-Match UND GW-Kosten -> saubere Geschichte statt Becken-Rauschen.
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
n = 256
muh = torch.full((n,), 1.0 / n, dtype=torch.float64, device=dev)
RES = {}


def two_moons(num=2000, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n1 = num // 2
    t1 = rng.uniform(0, np.pi, n1); t2 = rng.uniform(0, np.pi, num - n1)
    x1 = np.stack([1 - np.cos(t1), np.sin(t1) * 0.6 - 0.25], 1) + rng.normal(0, noise, (n1, 2))
    x2 = np.stack([1 + np.cos(t2), np.sin(t2) * 0.6 + 0.25], 1) + rng.normal(0, noise, (num - n1, 2))
    X = np.concatenate([x1, x2]).astype(np.float32)
    y = np.concatenate([np.zeros(n1, int), np.ones(num - n1, int)])
    idx = rng.permutation(num)
    return X[idx], y[idx]


class MoonsNet(nn.Module):
    def __init__(self, h, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(2, h); self.l2 = nn.Linear(h, h); self.out = nn.Linear(h, 2)

    def forward(self, x):
        return self.out(torch.relu(self.l2(torch.relu(self.l1(x)))))

    def hidden(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))


class DigitsNet(nn.Module):
    def __init__(self, h, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(64, h); self.l2 = nn.Linear(h, h); self.out = nn.Linear(h, 10)

    def forward(self, x):
        return self.out(torch.relu(self.l2(torch.relu(self.l1(x)))))

    def hidden(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))


def zc(H):
    return ((H - H.mean(0)) / (H.std(0) + 1e-8)).double()


def best_soft_plan(DX, DY, eps, B=32, n_outer=250):
    Gb, _ = my.entropic_gw_batch(DX, DY, muh, muh, eps, n_outer=n_outer,
                                 g0_batch=torch.softmax(torch.randn(B, n, n,
                                     generator=torch.Generator(device=dev), device=dev,
                                     dtype=torch.float64) * 4, dim=-1), n_sinkhorn=400)
    DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()
    Gb_np = Gb.cpu().numpy()
    Ls = [my.entropic_objective(Gb_np[b], DXn, DYn, eps) for b in range(B)]
    return Gb_np[int(np.argmin(Ls))]


def hard_gw_matching(DX, DY, anchors=None, iters=100, init_col=None):
    """Iteratives Hungarian auf GW-Gradient; anchors: Liste (i,j) fixiert."""
    m = DX.shape[0]
    a_g = np.full(m, 1.0 / m) @ (DX ** 2)
    b_g = np.full(m, 1.0 / m) @ (DY ** 2)
    col = init_col if init_col is not None else np.arange(m) % m
    for _ in range(iters):
        P = np.zeros((m, m)); P[np.arange(m), col] = 1.0 / m
        C = a_g[:, None] + b_g[None, :] - 2.0 * (DX @ P @ DY)
        big = 1e12
        if anchors:
            for i, j in anchors:
                C[i, :] = big; C[i, j] = 0.0
        ri, ci = linear_sum_assignment(C)
        if (ci == col).all():
            break
        col = ci
        if anchors:
            for i, j in anchors:
                col[i] = j
    return col


def gw_cost_colmap(DX, DY, col):
    m = DX.shape[0]
    P = np.zeros((m, m)); P[np.arange(m), col] = 1.0 / m
    mu = np.full(m, 1.0 / m)
    constA = float(mu @ (DX ** 2) @ mu); constB = float(mu @ (DY ** 2) @ mu)
    return constA + constB - 2.0 * float(((DX @ P) * (P @ DY)).sum())


# ================================================================ Moons (kanonisch 16 vs 32)
Xs, ys = two_moons(2000, seed=0)
mu_, sd_ = Xs.mean(0), Xs.std(0) + 1e-8
Xs = (Xs - mu_) / sd_
rng = np.random.default_rng(1); perm = rng.permutation(len(Xs))
X_tr = torch.tensor(Xs[perm[:1800]], dtype=torch.float32, device=dev)
y_tr = torch.tensor(ys[perm[:1800]], dtype=torch.long, device=dev)

nA, nB = MoonsNet(16, 0).to(dev), MoonsNet(32, 1).to(dev)
for net in [nA, nB]:
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(2500):
        idx = torch.randint(0, len(X_tr), (256,), device=dev)
        loss = nn.functional.cross_entropy(net(X_tr[idx]), y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()

# Probes: erste k Punkte GETEILT (gleiche Inputs), Rest disjunkt
k_shared_max = 16
rngp = np.random.default_rng(42)
poolA, y_poolA = two_moons(2000, noise=0.15, seed=123)
poolB, y_poolB = two_moons(2000, noise=0.15, seed=456)
shared = rngp.choice(len(poolA), k_shared_max, replace=False)   # geteilte Input-Punkte (aus A-Pool)
disjA = rngp.choice(np.setdiff1d(np.arange(len(poolA)), shared), n - k_shared_max, replace=False)
disjB = rngp.choice(np.setdiff1d(np.arange(len(poolB)), shared), n - k_shared_max, replace=False)

idxA = np.concatenate([shared, disjA]); idxB = np.concatenate([shared, disjB])
XpA = torch.tensor((np.concatenate([poolA[shared], poolA[disjA]]) - mu_) / sd_, dtype=torch.float32, device=dev)
XpB = torch.tensor((np.concatenate([poolB[shared], poolB[disjB]]) - mu_) / sd_, dtype=torch.float32, device=dev)
yAn = np.concatenate([y_poolA[shared], y_poolA[disjA]])
yBn = np.concatenate([y_poolB[shared], y_poolB[disjB]])
anchors_true = [(i, i) for i in range(k_shared_max)]   # wahre Korrespondenzen

with torch.no_grad():
    print("accA:", float((nA(XpA).argmax(1) == torch.tensor(yAn, device=dev)).float().mean()),
          " accB:", float((nB(XpB).argmax(1) == torch.tensor(yBn, device=dev)).float().mean()))
    HA, HB = zc(nA.hidden(XpA)), zc(nB.hidden(XpB))
DX, DY = torch.cdist(HA, HA) ** 2, torch.cdist(HB, HB) ** 2
eps = 0.02 * float(DX.median())
DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()

G_soft = best_soft_plan(DX, DY, eps)
col_soft = G_soft.argmax(0)
col_hard_ent = hard_gw_matching(DXn, DYn, init_col=col_soft.copy())
d_in_col = ((XpA.cpu().numpy()[:, None] - XpB.cpu().numpy()[None]) ** 2).sum(-1).argmin(1)
col_hard_in = hard_gw_matching(DXn, DYn, init_col=d_in_col.copy())

# label-restriktiert (innerhalb wahrer Klasse matchen; rechteckige LSA bei
# ungleichen Klassenzaehlen, eine LSA-Passage auf der GW-Linearisierung des
# weichen Plans)
col_lab = np.zeros(n, int)
for k in np.unique(yAn):
    ia = np.where(yAn == k)[0]; ib = np.where(yBn == k)[0]
    mA, mB = len(ia), len(ib)
    wA = np.full(mA, 1.0 / n); wB = np.full(mB, 1.0 / n)
    a_g = wA @ (DXn[np.ix_(ia, ia)] ** 2)
    b_g = wB @ (DYn[np.ix_(ib, ib)] ** 2)
    P0 = G_soft[np.ix_(ia, ib)]
    P0 = P0 / max(P0.sum(), 1e-300)
    C = a_g[:, None] + b_g[None, :] - 2.0 * (DXn[np.ix_(ia, ia)] @ P0 @ DYn[np.ix_(ib, ib)])
    ri, ci = linear_sum_assignment(C)
    col_lab[ia[ri]] = ib[ci]          # rechteckig: nur die gematchten Zeilen erhalten Partner

RES["E5d_moons"] = {
    "soft_rounding": {"label": float(np.mean(yAn == yBn[col_soft])),
                      "gw": gw_cost_colmap(DXn, DYn, col_soft)},
    "hard_from_entropic": {"label": float(np.mean(yAn == yBn[col_hard_ent])),
                           "gw": gw_cost_colmap(DXn, DYn, col_hard_ent)},
    "hard_from_inputNN": {"label": float(np.mean(yAn == yBn[col_hard_in])),
                          "gw": gw_cost_colmap(DXn, DYn, col_hard_in)},
    "label_restricted": {"label": float(np.mean(yAn == yBn[col_lab])),
                         "gw": gw_cost_colmap(DXn, DYn, col_lab)},
}
for k_, v_ in RES["E5d_moons"].items():
    print(f"E5d) moons {k_}: Label={v_['label']:.3f}  GW={v_['gw']:.1f}")

# Ankerkurve: k wahre Korrespondenzen fixieren, Rest hart matchen (init entropisch)
curve = []
for k in [0, 1, 2, 4, 8, 16]:
    tr = []
    for t in range(8):
        anch = anchors_true[:k]
        col = hard_gw_matching(DXn, DYn, anchors=anch, init_col=col_soft.copy())
        tr.append(float(np.mean(yAn == yBn[col])))
    curve.append({"k": k, "mean": float(np.mean(tr)), "min": float(np.min(tr)), "max": float(np.max(tr))})
    print(f"E5c) moons k={k:2d}: Label={np.mean(tr):.3f} [{np.min(tr):.3f},{np.max(tr):.3f}]")
RES["E5c_anchor_moons"] = curve

# ================================================================ Digits
dg = load_digits()
Xd, yd = dg.data.astype(np.float32) / 16.0, dg.target.astype(int)
rngd = np.random.default_rng(7)
pdd = rngd.permutation(len(Xd))
Xd, yd = Xd[pdd], yd[pdd]
Xtr_d = torch.tensor(Xd[:1400], dtype=torch.float32, device=dev)
ytr_d = torch.tensor(yd[:1400], dtype=torch.long, device=dev)
netAd, netBd = DigitsNet(16, 3).to(dev), DigitsNet(32, 4).to(dev)
for net in [netAd, netBd]:
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3000):
        idx = torch.randint(0, len(Xtr_d), (256,), device=dev)
        loss = nn.functional.cross_entropy(net(Xtr_d[idx]), ytr_d[idx])
        opt.zero_grad(); loss.backward(); opt.step()

sd_max = 16
rngq = np.random.default_rng(99)
shared_d = rngq.choice(np.arange(1400, len(Xd)), sd_max, replace=False)
rest = np.setdiff1d(np.arange(1400, len(Xd)), shared_d)
disjAd = rngq.choice(rest, n - sd_max, replace=False)
disjBd = rngq.choice(rest, n - sd_max, replace=False)
idxAd = np.concatenate([shared_d, disjAd]); idxBd = np.concatenate([shared_d, disjBd])
XpAd = torch.tensor(Xd[idxAd], dtype=torch.float32, device=dev)
XpBd = torch.tensor(Xd[idxBd], dtype=torch.float32, device=dev)
yAnd, yBnd = yd[idxAd], yd[idxBd]
anchors_d = [(i, i) for i in range(sd_max)]

with torch.no_grad():
    print("digits accA:", float((netAd(XpAd).argmax(1) == torch.tensor(yAnd, device=dev)).float().mean()),
          " accB:", float((netBd(XpBd).argmax(1) == torch.tensor(yBnd, device=dev)).float().mean()))
    HAd, HBd = zc(netAd.hidden(XpAd)), zc(netBd.hidden(XpBd))
DXd, DYd = torch.cdist(HAd, HAd) ** 2, torch.cdist(HBd, HBd) ** 2
eps_d = 0.02 * float(DXd.median())
DXdn, DYdn = DXd.cpu().numpy(), DYd.cpu().numpy()

Gd_soft = best_soft_plan(DXd, DYd, eps_d)
col_soft_d = Gd_soft.argmax(0)
col_hard_ent_d = hard_gw_matching(DXdn, DYdn, init_col=col_soft_d.copy())
col_lab_d = np.zeros(n, int)
for k in np.unique(yAnd):
    ia = np.where(yAnd == k)[0]; ib = np.where(yBnd == k)[0]
    mA, mB = len(ia), len(ib)
    wA = np.full(mA, 1.0 / n); wB = np.full(mB, 1.0 / n)
    a_g = wA @ (DXdn[np.ix_(ia, ia)] ** 2)
    b_g = wB @ (DYdn[np.ix_(ib, ib)] ** 2)
    P0 = Gd_soft[np.ix_(ia, ib)]
    P0 = P0 / max(P0.sum(), 1e-300)
    C = a_g[:, None] + b_g[None, :] - 2.0 * (DXdn[np.ix_(ia, ia)] @ P0 @ DYdn[np.ix_(ib, ib)])
    ri, ci = linear_sum_assignment(C)
    col_lab_d[ia[ri]] = ib[ci]

RES["E5d_digits"] = {
    "soft_rounding": {"label": float(np.mean(yAnd == yBnd[col_soft_d])),
                      "gw": gw_cost_colmap(DXdn, DYdn, col_soft_d)},
    "hard_from_entropic": {"label": float(np.mean(yAnd == yBnd[col_hard_ent_d])),
                           "gw": gw_cost_colmap(DXdn, DYdn, col_hard_ent_d)},
    "label_restricted": {"label": float(np.mean(yAnd == yBnd[col_lab_d])),
                         "gw": gw_cost_colmap(DXdn, DYdn, col_lab_d)},
}
for k_, v_ in RES["E5d_digits"].items():
    print(f"E5d) digits {k_}: Label={v_['label']:.3f}  GW={v_['gw']:.1f}")

curve_d = []
for k in [0, 1, 2, 4, 8, 16]:
    tr = []
    for t in range(8):
        col = hard_gw_matching(DXdn, DYdn, anchors=anchors_d[:k], init_col=col_soft_d.copy())
        tr.append(float(np.mean(yAnd == yBnd[col])))
    curve_d.append({"k": k, "mean": float(np.mean(tr)), "min": float(np.min(tr)), "max": float(np.max(tr))})
    print(f"E5c) digits k={k:2d}: Label={np.mean(tr):.3f} [{np.min(tr):.3f},{np.max(tr):.3f}]")
RES["E5c_anchor_digits"] = curve_d

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e5cd_anchor_rescue.json"
out.write_text(json.dumps(RES, indent=1, default=float))
print("gespeichert ->", out)

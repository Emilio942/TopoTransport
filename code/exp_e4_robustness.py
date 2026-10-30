"""E4: Robustheits-Checks zum Abschluss.

E4a: E3 (Automorphismus-Falle) ueber 4 unabhaengige Netzpaar-Seeds
E4b: eps_c-Phasenuebergang bei n=8 und n=32 (Skalierung der Theoriekurve)
E4c: POT-Cross-Check auf den echten A/B-Hidden-Wolken
E4d: Produkt-Init vs Zufalls-Init auf isometrischem (nicht-aequidistantem) Paar
"""
import sys, json, pathlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gw as my
import ot

dev = "cuda"
RES = {}
n = 256


def two_moons(num=2000, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n1 = num // 2
    t1 = rng.uniform(0, np.pi, n1); t2 = rng.uniform(0, np.pi, num - n1)
    x1 = np.stack([1 - np.cos(t1), np.sin(t1) * 0.6 - 0.25], 1) + rng.normal(0, noise, (n1, 2))
    x2 = np.stack([1 + np.cos(t2), np.sin(t2) * 0.6 + 0.25], 1) + rng.normal(0, noise, (num - n1, 2))
    X = np.concatenate([x1, x2]); y = np.concatenate([np.zeros(n1, int), np.ones(num - n1, int)])
    idx = rng.permutation(num)
    return X[idx], y[idx]


X_all, y_all = two_moons()
mu_, sd_ = X_all.mean(0), X_all.std(0)
X_all = (X_all - mu_) / sd_
rng = np.random.default_rng(1)
perm = rng.permutation(len(X_all))
X_tr = torch.tensor(X_all[perm[:1800]], dtype=torch.float32, device=dev)
y_tr = torch.tensor(y_all[perm[:1800]], dtype=torch.long, device=dev)  # y_all ist bereits intern permutiert & aligniert


def probe(num, seed):
    Xp, yp = two_moons(num=num, noise=0.15, seed=seed)
    return (torch.tensor((Xp - mu_) / sd_, dtype=torch.float32, device=dev),
            torch.tensor(yp, dtype=torch.long, device=dev))


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)

    def hidden(self, x):
        return torch.relu(self.net[2](torch.relu(self.net[0](x))))


def train(h, seed, steps=2500, lr=1e-2):
    torch.manual_seed(seed)                    # Seed vor Konstruktion -> kontrollierte Init
    net = MLP(h).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(X_tr), (256,), device=dev)
        loss = nn.functional.cross_entropy(net(X_tr[idx]), y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


XpA, yA = probe(n, seed=123)
XpB, yB = probe(n, seed=456)


def zc(H):
    return ((H - H.mean(0)) / (H.std(0) + 1e-8)).double()


def best_plan(DX, DY, eps, B=32, n_outer=250):
    Gb, _ = my.entropic_gw_batch(DX, DY, muh, muh, eps, n_outer=n_outer,
                                 g0_batch=torch.softmax(torch.randn(B, *DX.shape,
                                     generator=torch.Generator(device=dev), device=dev,
                                     dtype=torch.float64) * 4, dim=-1), n_sinkhorn=400)
    DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()
    Gb_np = Gb.cpu().numpy()
    Ls = [my.entropic_objective(Gb_np[b], DXn, DYn, eps) for b in range(B)]
    return Gb_np[int(np.argmin(Ls))]


muh = torch.full((n,), 1.0 / n, dtype=torch.float64, device=dev)

# ------------------------------------------------------------------ E4a: Seed-Robustheit
rows = []
for seed_net in [10, 11, 12, 13]:
    netA = train(16, seed_net)
    netB = train(32, seed_net + 100)
    with torch.no_grad():
        accA = (netA(XpA).argmax(1) == yA).float().mean().item()
        accB = (netB(XpB).argmax(1) == yB).float().mean().item()
        HA, HB = zc(netA.hidden(XpA)), zc(netB.hidden(XpB))
    DX = torch.cdist(HA, HA) ** 2
    DY = torch.cdist(HB, HB) ** 2
    eps = 0.02 * float(DX.median())
    G = best_plan(DX, DY, eps)
    colA = G.argmax(0)
    yAn, yBn = yA.cpu().numpy(), yB.cpu().numpy()
    lab_gw = float(np.mean(yAn == yBn[colA]))
    lab_rand = float(np.mean(yAn == yBn[np.random.permutation(n)]))
    d_in = ((XpA.cpu().numpy()[:, None, :] - XpB.cpu().numpy()[None, :, :]) ** 2).sum(-1)
    lab_in = float(np.mean(yAn == yBn[d_in.argmin(1)]))
    rows.append({"seed": seed_net, "accA": accA, "accB": accB,
                 "label_match_gw": lab_gw, "label_match_random": lab_rand,
                 "label_match_inputNN": lab_in,
                 "gw_value": my.gw_cost(G, DX.cpu().numpy(), DY.cpu().numpy())})
    print(f"E4a) seed={seed_net}: accA={accA:.3f} accB={accB:.3f} | "
          f"GW-Label={lab_gw:.3f} rand={lab_rand:.3f} inputNN={lab_in:.3f}")
RES["E4a_seed_robustness"] = rows

# ------------------------------------------------------------------ E4b: eps_c-Skalierung
def transition_scan(n_pts, c=1.0):
    DXs = np.full((n_pts, n_pts), c ** 2); np.fill_diagonal(DXs, 0.0)
    mun = np.full(n_pts, 1.0 / n_pts)
    eps_c = c ** 2 * (1.0 / n_pts - 1.0 / n_pts ** 2) / np.log(n_pts)
    prod = mun[:, None] * mun[None, :]
    lo, hi = None, None
    for eps_rel in np.logspace(-3, 0, 25):
        eps = eps_rel * np.median(DXs)
        tDX = torch.tensor(DXs, dtype=torch.float64, device=dev)
        tmu = torch.tensor(mun, dtype=torch.float64, device=dev)
        G0 = torch.softmax(8 * torch.randn(32, n_pts, n_pts,
            generator=torch.Generator(device=dev), device=dev, dtype=torch.float64), dim=-1)
        Gb, _ = my.entropic_gw_batch(tDX, tDX, tmu, tmu, eps, n_outer=250, g0_batch=G0, n_sinkhorn=500)
        Gb_np = Gb.cpu().numpy()
        Ls = [my.entropic_objective(Gb_np[b], DXs, DXs, eps) for b in range(32)]
        g = Gb_np[int(np.argmin(Ls))] if min(Ls) < my.entropic_objective(prod, DXs, DXs, eps) else prod
        tv = 0.5 * float(np.abs(g - prod).sum())
        if tv > 0.5 and lo is None:
            lo = eps_rel
        if tv < 0.05 and hi is None and lo is not None:
            hi = eps_rel
            break
    return eps_c, lo, hi

trans = {}
for n_pts in [8, 16, 32]:
    eps_c, lo, hi = transition_scan(n_pts)
    trans[f"n={n_pts}"] = {"eps_c_theory": eps_c, "last_perm_eps": lo, "first_product_eps": hi}
    print(f"E4b) n={n_pts}: eps_c-Theorie={eps_c:.5f}, Übergang gemessen in [{lo:.4f}, {hi:.4f}] "
          f"({'KONSISTENT' if lo < eps_c < hi else 'ABWEICHUNG'})")
RES["E4b_eps_c_scaling"] = trans

# ------------------------------------------------------------------ E4c: POT-Cross-Check auf echten Wolken
netA = train(16, 0)
netB = train(32, 1)
XpA, _ = probe(n, seed=123)
XpB, _ = probe(n, seed=456)
with torch.no_grad():
    HA, HB = zc(netA.hidden(XpA)), zc(netB.hidden(XpB))
DX = torch.cdist(HA, HA) ** 2
DY = torch.cdist(HB, HB) ** 2
eps = 0.02 * float(DX.median())
G_mine = best_plan(DX, DY, eps)
mun = np.full(n, 1.0 / n)
G_pot = ot.gromov.entropic_gromov_wasserstein(
    DX.cpu().numpy(), DY.cpu().numpy(), mun, mun, "square_loss", epsilon=float(eps),
    solver="PGD", max_iter=500, tol=1e-12, numItermax=5000, stopThr=1e-12)
DXn, DYn = DX.cpu().numpy(), DY.cpu().numpy()
v_mine = my.gw_cost(G_mine, DXn, DYn)
v_pot = my.gw_cost(G_pot, DXn, DYn)
marg_pot = float(np.abs(G_pot.sum(1) - mun).max() + np.abs(G_pot.sum(0) - mun).max())
print(f"E4c) POT-Plan-Marginalfehler: {marg_pot:.2e}")
colA = G_mine.argmax(0)
colA_pot = G_pot.argmax(0)
# Übereinstimmung der Rundungen:
agree = float(np.mean(colA == colA_pot))
RES["E4c_pot_crosscheck"] = {"gw_mine": v_mine, "gw_pot": v_pot,
                             "rounding_agreement": agree}
print(f"E4c) GW mine={v_mine:.2f} vs POT={v_pot:.2f} | argmax-Übereinstimmung={agree:.2%}")

# ------------------------------------------------------------------ E4d: Init-Vergleich (isometrisch, nicht-äquidistant)
tg = torch.Generator().manual_seed(7)
X = torch.randn(16, 2, generator=tg, dtype=torch.float64)
Y = X[torch.randperm(16, generator=torch.Generator().manual_seed(8))]
DXi = (torch.cdist(X, X) ** 2).to(dev)
DYi = (torch.cdist(Y, Y) ** 2).to(dev)
eps = 0.03 * float(DXi.median())
muh16 = torch.full((16,), 1.0 / 16, dtype=torch.float64, device=dev)
# Produkt-Init:
G0_prod = torch.einsum("i,j->ij", muh16.cpu(), muh16.cpu()).to(dev).unsqueeze(0)
Gb_p, _ = my.entropic_gw_batch(DXi, DYi, muh16, muh16, eps, n_outer=400,
                               g0_batch=G0_prod.repeat(1, 1, 1), n_sinkhorn=500)
gw_prod = float(my.gw_cost(Gb_p[0].cpu().numpy(), DXi.cpu().numpy(), DYi.cpu().numpy()))
# Zufalls-Inits (Best of 64):
G0_r = torch.softmax(6 * torch.randn(64, 16, 16, generator=torch.Generator(device=dev),
                                     device=dev, dtype=torch.float64), dim=-1)
Gb_r, _ = my.entropic_gw_batch(DXi, DYi, muh16, muh16, eps, n_outer=400, g0_batch=G0_r, n_sinkhorn=500)
gw_rand = float(min(my.gw_cost(Gb_r[b].cpu().numpy(), DXi.cpu().numpy(), DYi.cpu().numpy())
                    for b in range(64)))
RES["E4d_init_comparison"] = {"gw_product_init": gw_prod, "gw_best_random_init": gw_rand,
                              "product_reaches_global": bool(gw_prod < 1e-6)}
print(f"E4d) isometrisches Paar (n=16, eps_rel=0.03): Produkt-Init GW={gw_prod:.6f} | "
      f"Best-of-64-random GW={gw_rand:.6f}")

out = pathlib.Path(__file__).resolve().parent.parent / "results" / "e4_robustness.json"
out.write_text(json.dumps(RES, indent=1, default=float))
print("gespeichert ->", out)

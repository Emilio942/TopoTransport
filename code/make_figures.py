"""Figuren fuer den Forschungsbericht."""
import sys, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

# ---------------- Fig 1: 2x2-Landschaft (Zyklus 91)
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

u = np.linspace(0, 1, 300)
UU, VV, WW = np.meshgrid(u, u, u, indexing="ij")
mask = (UU + VV + WW) <= 1.0
pts = np.stack([UU[mask], VV[mask], WW[mask]], 1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, eps in zip(axes, [0.0, 0.05, 1.0]):
    L = L2x2(pts, eps)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=L, s=2, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="L_eps")
    i = L.argmin()
    ax.plot(pts[i, 0], pts[i, 1], "r*", ms=14, label=f"argmin ({pts[i,0]:.2f},{pts[i,1]:.2f})")
    ax.set_title(f"eps = {eps}")
    ax.set_xlabel("g11"); ax.set_ylabel("g12")
    ax.legend(fontsize=8)
fig.suptitle("Entropischer GW-Term im 2x2-Kopplungspolytop (Schnitt g22): nicht-konvex")
fig.tight_layout(); fig.savefig(FIG / "fig1_cycle91_landscape.png"); plt.close(fig)

# ---------------- Fig 2: Phasenuebergang (Simplex)
d = json.loads((RES / "e2_obstruction.json").read_text())
rows = d["D_simplex_phase_transition"]["rows"]
eps_c = d["D_simplex_phase_transition"]["eps_c_theory"]
x = [r["eps_rel"] for r in rows]
tv = [r["TV_to_product"] for r in rows]
perm = [1.0 if r["argmax_is_perm"] else 0.0 for r in rows]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogx(x, tv, "o-", label="TV(π*, μ⊗ν)")
ax.semilogx(x, perm, "s--", label="argmax ist Permutation")
ax.axvline(eps_c, color="crimson", ls=":", label=f"Theorie ε_c = {eps_c:.4f}")
ax.set_xlabel("ε_rel (= ε / median(D²))"); ax.set_ylabel("TV / Indikator")
ax.set_title("Simplex (n=16): Symmetrie-Auflösung kollabiert am theoretischen ε_c")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig2_phase_transition.png"); plt.close(fig)

# ---------------- Fig 3: Distortion (Kreis, Rauschen)
resf = d["F_distortion"]
eps_vals = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = {0.0: "tab:green", 0.05: "tab:orange", 0.15: "tab:red"}
for noise in [0.0, 0.05, 0.15]:
    bar = [resf[f"noise={noise},eps_rel={e}"]["barycentric_rel_distortion"] for e in eps_vals]
    rnd = [resf[f"noise={noise},eps_rel={e}"]["rounding_rel_distortion"] for e in eps_vals]
    ax.semilogx(eps_vals, bar, "o-", color=colors[noise], alpha=0.45,
                label=f"baryzentrisch δ={noise}")
    ax.semilogx(eps_vals, rnd, "s-", color=colors[noise], label=f"rounding δ={noise}")
ax.set_xlabel("ε_rel"); ax.set_ylabel("relative Distanz-Verzerrung")
ax.set_title("Kreis (isometrisch): Rounding vs. baryzentrische Projektion")
ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(FIG / "fig3_distortion.png"); plt.close(fig)

# ---------------- Fig 4: Transfer
t = json.loads((RES / "e3_transfer.json").read_text())
rows = t["transfer"]["rows"]
x = [r["eps_rel"] for r in rows]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogx(x, [r["label_transfer_gw"] for r in rows], "o-", label="GW-Korrespondenz")
ax.semilogx(x, [r["label_transfer_random"] for r in rows], ":", label="Zufallsperm.")
ax.semilogx(x, [r["label_transfer_product"] for r in rows], "--", label="Produktkopplung")
ax.semilogx(x, [r["label_transfer_oracle_lsa"] for r in rows], "-.", label="Oracle (LSA)")
ax.semilogx(x, [r["position_transfer_rounding_acc"] for r in rows], "s-", alpha=0.6, label="Positionstransfer (Rounding)")
ax.axhline(t["transfer"]["probe_reference_A"], color="gray", lw=1, ls=":", label="A-Referenz")
ax.set_xlabel("ε_rel"); ax.set_ylabel("Accuracy / Agreement")
ax.set_title("Zero-Shot-Transfer A(h=16) → B(h=32), disjunkte Probes")
ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(FIG / "fig4_transfer.png"); plt.close(fig)

# ---------------- Fig 5: Kosten der Label-Konsistenz
lc = json.loads((RES / "e3c_label_cost.json").read_text())
names = ["gw_hard_unconstrained", "gw_hard_label_restricted", "input_nn"]
labels = ["GW-optimal\n(ungeschraenkt)", "GW-optimal\n(label-restriktiert)", "Input-NN"]
gw = [lc[k]["gw_cost"] for k in names]
lab = [lc[k]["label_match"] for k in names]
fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
bars = ax1.bar(labels, gw, color=["tab:red", "tab:green", "tab:blue"], alpha=0.75)
ax1.set_ylabel("GW-Kosten der harten Korrespondenz")
ax2 = ax1.twinx()
ax2.plot(labels, lab, "ko", ms=9)
ax2.set_ylabel("Label-Match"); ax2.set_ylim(0, 1.05)
for b, l in zip(bars, lab):
    ax2.annotate(f"{l:.2f}", (b.get_x() + b.get_width() / 2, 0.93), ha="center", fontsize=10)
ax1.set_title("Kosten der Label-Konsistenz: +101% GW fuer 100% Semantik")
fig.tight_layout(); fig.savefig(FIG / "fig5_label_cost.png"); plt.close(fig)

print("Figuren geschrieben nach", FIG)

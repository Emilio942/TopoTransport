"""Entropie-regulaerter Gromov-Wasserstein: Referenz- (NumPy) und Batch-GPU- (Torch) Solver.

GW-Quadrat:        GW(g) = < L(g), g >  mit  L_ikjl = (DX_ik - DY_jl)^2
                   = const_X + const_Y - 2 < DX g DY, g >
Gradient (Marginals fix):  dGW/dg = -2 DX g DY
Entropisch:   min_g  GW(g) + eps * KL(g || mu (x) nu)
Algorithmus (Peyre/Cuturi/Solomon 2016): alternierend
   g <- Sinkhorn(mu, nu, C(g))  mit  C(g) = a 1' + 1 b' - 2 DX g DY,  a_i = (DX^2 mu)_i, b_j = (DY^2 nu)_j
Die Konstanten a,b sind noetig, damit der quadratische Term korrekt linearisiert wird
(aendern die Sinkhorn-Loesung nicht, aber den gemeldeten Kostenwert).
"""
import numpy as np


# ---------------------------------------------------------------- NumPy-Referenz
def sinkhorn(mu, nu, C, eps, n_iter=200, tol=1e-12, check_every=20):
    """Entropische OT in Log-Potential-Konvention: G = exp(phi_i + gamma_j - C_ij/eps).

    phi = f/eps, gamma = g/eps (ungeskaliert) — vermeidet den klassischen
    Skalierungsfehler 'M = g - lam' statt 'M = g/eps - lam'.
    Abbruch via Plan-Marginals (robust ggü. Potential-Shifts).
    """
    mu = np.asarray(mu, float)
    nu = np.asarray(nu, float)
    lam = np.asarray(C, float) / eps          # log K
    phi = np.zeros(len(mu))
    gam = np.zeros(len(nu))
    log_mu = np.log(mu)
    log_nu = np.log(nu)
    G = None
    for t in range(1, n_iter + 1):
        phi = log_mu - _logsumexp(gam[None, :] - lam, axis=1)
        gam = log_nu - _logsumexp(phi[:, None] - lam, axis=0)
        if t % check_every == 0 or t == n_iter:
            lse = phi[:, None] + gam[None, :] - lam
            lse -= lse.max()
            G = np.exp(lse)
            err = np.abs(G.sum(axis=1) - mu).max() + np.abs(G.sum(axis=0) - nu).max()
            if err < tol:
                break
    lse = phi[:, None] + gam[None, :] - lam
    lse -= lse.max()
    G = np.exp(lse)
    G /= G.sum()
    return G


def _logsumexp(A, axis=None):
    m = A.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out


def gw_cost(g, DX, DY):
    """GW(g) = sum_ikjl g_ij g_kl (DX_ik - DY_jl)^2, exakt via Expansion (O(n^3))."""
    mu = g.sum(axis=1)
    nu = g.sum(axis=0)
    constA = float(mu @ (DX ** 2) @ mu)
    constB = float(nu @ (DY ** 2) @ nu)
    cross = float(((DX @ g) * (g @ DY)).sum())  # sum_ij (DX g DY)_ij g_ij
    return constA + constB - 2.0 * cross


def gw_linear_cost(g, DX, DY):
    """Innerer OT-Kosten-Operator C(g)_ij = a_i + b_j - 2 (DX g DY)_ij."""
    mu = g.sum(axis=1)
    nu = g.sum(axis=0)
    a = DX ** 2 @ mu
    b = DY ** 2 @ nu
    return a[:, None] + b[None, :] - 2.0 * (DX @ g @ DY)


def entropic_gw(DX, DY, mu, nu, eps, n_outer=300, tol=1e-9, g0=None, n_sinkhorn=500):
    """Entropisch reguliertes GW via alternierender Linearisierung. Liefert (g, hist).

    Robustheit: trackt den besten Plan bzgl. L_eps = GW + eps*KL (gegen
    2-Zyklen/Oszillation der aesseren Schleife, die bei sehr kleinem eps
    (Untergrenze ~0.01*median(D^2), float64-Kernel-Underflow) auftreten).
    """
    DX = np.asarray(DX, float)
    DY = np.asarray(DY, float)
    mu = np.asarray(mu, float)
    nu = np.asarray(nu, float)
    if g0 is None:
        g = mu[:, None] * nu[None, :]
    else:
        g = g0 / g0.sum()
    hist = []
    best_g, best_val = None, np.inf
    prev = np.inf
    stall = 0
    for it in range(n_outer):
        C = gw_linear_cost(g, DX, DY)
        g_new = sinkhorn(mu, nu, C, eps, n_iter=n_sinkhorn)
        val = entropic_objective(g_new, DX, DY, eps)
        hist.append(val)
        if val < best_val:
            best_val, best_g = val, g_new.copy()
        delta = np.max(np.abs(g_new - g))
        g = g_new
        if abs(prev - val) < tol and delta < 1e-7:
            break
        if delta > 0.5:            # Oszillations-Schutz
            stall += 1
            if stall >= 3:
                break
        prev = val
    return best_g, np.array(hist)


def kl_to_product(g, mu, nu):
    prod = mu[:, None] * nu[None, :]
    mask = g > 1e-300
    return float((g[mask] * np.log(g[mask] / prod[mask])).sum())


def entropic_objective(g, DX, DY, eps):
    mu = g.sum(axis=1)
    nu = g.sum(axis=0)
    return gw_cost(g, DX, DY) + eps * kl_to_product(g, mu, nu)


# ---------------------------------------------------------------- Torch-Batch (GPU)
import torch


def sinkhorn_batch(mu, nu, logK, n_iter=100, check_every=10, tol=1e-9):
    """Batch-Sinkhorn, Log-Potential-Konvention. logK: (B,n,m); mu:(n,), nu:(m,).

    G = exp(phi_i + gamma_j + logK_ij); Abbruch via Plan-Marginals (max ueber Batch).
    logK wird direkt uebergeben (kein log(exp())-Roundtrip, kein Underflow).
    """
    log_mu = torch.log(mu)[None, :, None]
    log_nu = torch.log(nu)[None, None, :]
    phi = torch.zeros_like(logK[:, :, 0])          # (B,n)
    gam = torch.zeros_like(logK[:, 0, :])          # (B,m)
    P = torch.exp(logK)
    for t in range(1, n_iter + 1):
        phi = log_mu[:, 0, :] - torch.logsumexp(gam[:, None, :] + logK, dim=2)
        gam = log_nu[0, :, :] - torch.logsumexp(phi[:, :, None] + logK, dim=1)
        if t % check_every == 0 or t == n_iter:
            P = torch.exp(phi[:, :, None] + gam[:, None, :] + logK)
            err = (P.sum(dim=1) - mu[None, :]).abs().max() + (P.sum(dim=2) - nu[None, :]).abs().max()
            if err < tol:
                break
    return P


def entropic_gw_batch(DX, DY, mu, nu, eps, n_outer=200, g0_batch=None, n_sinkhorn=100, device="cuda"):
    """Loest B entropische-GW-Probleme parallel (gleiche DX,DY,eps, verschiedene Inits).

    DX:(n,n), DY:(m,m) torch auf device. g0_batch:(B,n,m) Startkopplungen.
    Liefert (G:(B,n,m), vals:(B,n_outer)) mit GW-Verlauf je Batch-Mitglied.
    """
    n, m = DX.shape[0], DY.shape[0]
    DX2 = DX ** 2
    DY2 = DY ** 2
    a = DX2 @ mu                                   # (n,)
    b = DY2 @ nu                                   # (m,)
    DXb = DX[None, :, :].expand(g0_batch.shape[0], n, n)
    DYb = DY[None, :, :].expand(g0_batch.shape[0], m, m)
    G = g0_batch.clone()
    vals = []
    for _ in range(n_outer):
        Tens = torch.bmm(torch.bmm(DXb, G), DYb)               # DX G_b DY
        C = a[:, None] + b[None, :] - 2.0 * Tens
        logK = -C / eps
        G = sinkhorn_batch(mu, nu, logK, n_iter=n_sinkhorn)
        Tens_new = torch.bmm(torch.bmm(DXb, G), DYb)
        cross = (Tens_new * G).sum(dim=(1, 2))
        GW_b = (mu @ DX2 @ mu) + (nu @ DY2 @ nu) - 2.0 * cross
        vals.append(GW_b)
    return G, torch.stack(vals, dim=1)

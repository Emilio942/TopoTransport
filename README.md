# TopoTransport — Entropic Gromov-Wasserstein, Symmetry Obstruction & Zero-Shot Transfer

Empirical research project investigating whether **entropy-regularized Gromov-Wasserstein optimal transport** can resolve hidden-layer gauge invariances / permutation symmetries $S_n$ between neural network architectures, and whether the resulting transport plan $\pi^*$ yields **metric-preserving activation maps** enabling **zero-shot cross-architecture representation transfer** without alignment losses or fine-tuning.

**Verdict: No — with precise boundaries, quantified.** Full report (German): [`REPORT.md`](REPORT.md).

## TL;DR

| Question | Answer | Evidence |
|---|---|---|
| Can entropic GW resolve permutation symmetries? | Only below a sharp threshold, and only modulo automorphisms | Exact phase transition $\varepsilon_c = c^2(1/n - 1/n^2)/\log n$, verified at $n=8,16,32$; orbit members chosen arbitrarily (D4 square: 8 optimal plans, $GW=0$) |
| Does $\pi^*$ yield metric-preserving activation maps? | Rounding: yes (0.0000 distortion, isometric pairs, $\varepsilon_{rel}\le 0.05$). Barycentric projection: no (~1.0 distortion) | Circle experiments, noise sweep |
| Does it transfer labels zero-shot between architectures? | **No** — GW correspondence is label-blind | Label match at chance level (0.07–0.58) across 5 network seeds, 2 datasets, all width pairs incl. same-architecture |
| What extra information fixes it? | **An initial symmetry-breaking prior**, not labels/anchors | 16 pinned true correspondences do NOT rescue (flat at chance); input-space NN init reaches 0.89–1.00 with zero labels |

## The mathematics in one line

$$\pi^* \text{ resolves } (D^X, D^Y) \text{ only modulo } \mathrm{Aut}(D^X)\times\mathrm{Aut}(D^Y), \qquad \text{semantics} \notin \sigma\big((D^X, D^Y)\big)$$

1. **Identifiability (O1):** $\mathcal{L}_\varepsilon = GW - \varepsilon H$ is invariant under simultaneous distance-preserving relabeling; minimizers form whole orbits. Labels are not a function of the distance matrices — information-theoretic, not algorithmic.
2. **Entropy does not break symmetry, it penalizes resolution:** for equidistant spaces $GW(\gamma) = 2c^2(1/n - \sum\gamma_{ij}^2)$ exactly; competition with max-entropy gives the exact threshold $\varepsilon_c$ — above it $\pi^*$ collapses to the product coupling $\mu\otimes\nu$ (maximum symmetry, zero resolution).
3. **Basin obstruction:** the label-consistent correspondence exists (+12–26% GW cost) but lives in a different basin; first-order alternating linearization cannot see it, and pinning up to 16 true correspondences does not steer local re-optimization into it.
4. **No isometry to preserve:** independently trained nets are not GW-isometric (3.6× above same-net floor); "metric-preserving" is the wrong demand — the metric encodes class boundaries only up to automorphism.

## Meta-analysis

66 prior research hypotheses on this question were coded and analyzed: **89.4% invalid**, novelty uncorrelated with validity (r = −0.06). Failure taxonomy: symmetry-persistence (13), non-functoriality (11), ill-posed manifolds (10), unprovable (10), metric errors (6), full-support/no-map (5), category errors (4). Two "valid" verdicts were re-examined: **Cycle 91** confirmed in its non-convexity claim (24–32% convexity violations in the exact 2×2 polytope) but its marginal-wise convexity and gauge-manifold claims are refuted; **Cycle 166**'s proof is defective (Gibbs form is the optimality condition of *linear* entropic OT, not quadratic GW).

## Repository layout

```
├── REPORT.md          full research report (German, with all tables/numbers)
├── code/
│   ├── gw.py                     entropic GW solver (NumPy reference + GPU-batched torch, float64)
│   ├── meta_analysis.py          cycle statistics
│   ├── exp_e0_validation.py      solver validation (brute force, POT cross-check, ε→0 limit)
│   ├── exp_e1_cycle91.py         exact 2×2 landscape, marginal-slice convexity, GPU basins
│   ├── exp_e1b_gauge_vs_minima.py gauge orbit vs spurious local minima
│   ├── exp_e2_obstruction.py     ε_c phase transition, square orbit, distortion
│   ├── exp_e3_transfer.py        zero-shot transfer, disjoint probes (two MLPs on GPU)
│   ├── exp_e3b_controls.py       same-net floor vs cross-architecture gap
│   ├── exp_e3c_label_cost.py     price of label consistency
│   ├── exp_e4_robustness.py      seed robustness, ε_c scaling, POT check, init comparison
│   ├── exp_e5_consolidation.py   same-architecture control, digits dataset
│   ├── exp_e5cd_anchor_rescue.py anchor rescue + basin structure (E5c/d)
│   └── make_figures.py           figures 1–6
├── results/           all experiment outputs (JSON)
└── figures/           fig1–fig6 (PNG)
```

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install numpy scipy matplotlib POT torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install POT scikit-learn
.venv/bin/python code/meta_analysis.py
.venv/bin/python code/exp_e0_validation.py
.venv/bin/python code/exp_e1_cycle91.py && .venv/bin/python code/exp_e1b_gauge_vs_minima.py
.venv/bin/python code/exp_e2_obstruction.py
.venv/bin/python code/exp_e3_transfer.py && .venv/bin/python code/exp_e3b_controls.py \
  && .venv/bin/python code/exp_e3c_label_cost.py
.venv/bin/python code/exp_e4_robustness.py
.venv/bin/python code/exp_e5_consolidation.py
.venv/bin/python code/exp_e5cd_anchor_rescue.py
.venv/bin/python code/make_figures.py
```

Requires one CUDA GPU (~4 h total on an RTX 3060; float64 GW inner loops). Numerical floor: $\varepsilon_{rel} \ge 0.01$ (float64 Gibbs-kernel underflow below that).

## Key figures

| Figure | Content |
|---|---|
| `fig1` | Exact entropic-GW landscape on the 2×2 coupling polytope (non-convexity) |
| `fig2` | Simplex phase transition: symmetry resolution collapses at theoretical $\varepsilon_c$ |
| `fig3` | Rounding vs. barycentric distortion across ε and noise |
| `fig4` | Zero-shot label transfer vs. baselines (disjoint probes) |
| `fig5` | Price of label consistency |
| `fig6` | Anchor rescue fails; init determines the basin |

## Honest caveats

- Two moons + digits as controlled scenarios; mechanistic (automorphism degeneracy), not dataset-scale claims.
- Global GW optimality is NP-hard in general; all claims use best-of-multi-restart with 32–512 parallel GPU inits, and the *non*-convexity itself is part of the findings (verified two ways: landscape enumeration + independent POT/heuristic cross-checks).
- Three methodological bugs were found and fixed during robustness checks (label shortcut, single-class probes, potential-scaling); all are documented in `REPORT.md` §3–5, and every headline number is from the corrected pipeline.

## License

See `LICENSE`.

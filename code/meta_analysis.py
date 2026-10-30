"""Meta-Analyse der Research-Cycle-Liste (TopoTransport).

Codiert alle Zyklen aus dem Forschungsdokument und misst:
- Verdict-Verteilung je Cluster
- Korrelation Novelty <-> Validity
- Taxonomie der Obstruktionen in den Invalid-Begruendungen
"""
import json
import numpy as np

# (cycle, cluster, verdict, novelty, failure_class)
# failure_class:
#   none        = valid / unknown
#   O1_symmetry = GW ist invariant unter Isometrien => Symmetrien werden NICHT gebrochen/aufgeloest
#   O2_support  = entropische Pi* hat vollen Träger => keine deterministische/isometrische Karte
#   O3_functor  = GW-Plan ist nicht funktorial/kompositionell (Natürlichkeit, Monoidizität, Komposition)
#   O4_metric   = GW kein Metrik auf Maßen / Pseudo-Metrik-Fehler in Konstruktion
#   O5_illposed = schlecht gestellte Mannigfaltigkeit/Dynamik (diskrete Raume, fehlende Struktur)
#   O6_category = Kategorienfehler: Kopplung ist kein Operator/keine Aktivierungsfunktion
#   O7_unprovable = These zu vage/präzisionslos, weder Beweis noch Widerlegung moeglich
ROWS = [
    (1,   "AlgebraicGeometry",  "invalid", 1.000, "O1_symmetry"),   # homotopy stratification
    (1,   "AlgebraicGeometry",  "valid",   0.593, "none"),          # category-theoretic unification
    (2,   "Topology",           "invalid", 0.674, "O3_functor"),    # functorial OT on morphisms
    (2,   "Topology",           "invalid", 0.593, "O1_symmetry"),   # monoidal universal property (non-uniqueness)
    (2,   "Topology",           "valid",   0.512, "none"),          # equivariant GW on Lie groups
    (3,   "DifferentialGeometry","invalid",0.640, "O5_illposed"),   # homotopy stability
    (3,   "DifferentialGeometry","invalid",0.616, "O6_category"),   # cohomological activation bundles (no pullback)
    (3,   "DifferentialGeometry","invalid",0.535, "O7_unprovable"), # info-geometric curvature
    (4,   "Analysis",           "invalid", 0.523, "O7_unprovable"), # RMT for S_n
    (5,   "Logic",              "invalid", 0.605, "O1_symmetry"),   # spectral embedding invariance
    (5,   "Logic",              "invalid", 0.535, "O5_illposed"),   # dynamic GW flow, no smoothness
    (5,   "Logic",              "invalid", 0.500, "O2_support"),    # RKHS isometry from coupling
    (6,   "AlgebraicGeometry",  "invalid", 0.644, "O3_functor"),    # enriched category functor
    (6,   "AlgebraicGeometry",  "invalid", 0.563, "O2_support"),    # algebraic variety of maps
    (6,   "AlgebraicGeometry",  "invalid", 0.552, "O3_functor"),    # monoidal gauge functor
    (7,   "DifferentialGeometry","invalid",0.653, "O5_illposed"),   # CAT(0) activation manifolds
    (8,   "NumberTheory",       "invalid", 0.537, "O3_functor"),    # GW barycenter mediator
    (8,   "NumberTheory",       "invalid", 0.516, "O4_metric"),     # TDA barcodes, GW collapse
    (8,   "NumberTheory",       "invalid", 0.510, "O1_symmetry"),   # rep-theory S_n intertwining
    (9,   "ProbabilityTheory",  "invalid", 0.510, "O7_unprovable"), # quantum entropy GW
    (9,   "ProbabilityTheory",  "unknown", 0.505, "none"),          # spectral stability
    (11,  "Logic",              "invalid", 0.573, "O6_category"),   # Markov kernel eps-isometry
    (11,  "Logic",              "invalid", 0.509, "O7_unprovable"), # large deviations
    (15,  "Logic",              "invalid", 0.528, "O3_functor"),    # cohomological invariants functor
    (19,  "DynamicalSystems",   "invalid", 0.566, "O1_symmetry"),   # Ricci curvature orbit contraction
    (19,  "DynamicalSystems",   "invalid", 0.528, "O5_illposed"),   # birational morphisms
    (20,  "Analysis",           "invalid", 0.538, "O3_functor"),    # multi-marginal consensus
    (20,  "Analysis",           "invalid", 0.528, "O1_symmetry"),   # quotient manifold equivariance
    (23,  "Logic",              "invalid", 0.613, "O1_symmetry"),   # trivial rep projection = mean
    (24,  "DifferentialGeometry","invalid",0.626, "O7_unprovable"), # stochastic coupling of S_n
    (26,  "ProbabilityTheory",  "invalid", 0.515, "O1_symmetry"),   # theta->0 gauge invariance
    (36,  "DynamicalSystems",   "invalid", 0.552, "O6_category"),   # trace-class RKHS operator
    (38,  "ProbabilityTheory",  "invalid", 0.522, "O7_unprovable"), # persistent homology of plan space
    (42,  "Topology",           "invalid", 0.507, "O3_functor"),    # natural transformation
    (47,  "DynamicalSystems",   "invalid", 0.507, "O5_illposed"),   # gradient flow on discrete space
    (48,  "Topology",           "invalid", 0.590, "O4_metric"),     # graphon Riemannian structure
    (48,  "Topology",           "invalid", 0.545, "O3_functor"),    # simplicial contravariant functor
    (59,  "DynamicalSystems",   "invalid", 0.604, "O2_support"),    # metric-preserving flow
    (70,  "Logic",              "invalid", 0.522, "O1_symmetry"),   # MI maximization (max-ent residual)
    (77,  "ProbabilityTheory",  "invalid", 0.500, "O1_symmetry"),   # S_n irreducible decomposition
    (78,  "AlgebraicGeometry",  "invalid", 0.500, "O5_illposed"),   # heat flow on moduli
    (90,  "NumberTheory",       "invalid", 0.560, "O7_unprovable"), # random matrix limit laws
    (91,  "Analysis",           "valid",   0.522, "none"),          # non-convex landscape, fixed points
    (97,  "ProbabilityTheory",  "invalid", 0.612, "O2_support"),    # eigenvector alignment rank deficiency
    (97,  "ProbabilityTheory",  "unknown", 0.607, "none"),          # persistent cohomology obstructions
    (135, "AlgebraicGeometry",  "invalid", 0.548, "O4_metric"),     # moduli geodesic metric
    (166, "DifferentialGeometry","valid",  0.556, "none"),          # SDE on topology manifold
    (176, "DifferentialGeometry","invalid",0.563, "O7_unprovable"), # Gibbs concentration bounds
    (177, "Topology",           "unknown", 0.504, "none"),          # activation dynamics
    (211, "DifferentialGeometry","invalid",0.511, "O7_unprovable"), # quotient orbit transport
    (217, "AlgebraicGeometry",  "invalid", 0.622, "O6_category"),   # derived categories
    (217, "AlgebraicGeometry",  "invalid", 0.548, "O3_functor"),    # cohomology vanishing
    (217, "AlgebraicGeometry",  "invalid", 0.519, "O5_illposed"),   # moduli of activation maps
    (219, "Topology",           "invalid", 0.504, "O1_symmetry"),   # expander concentration
    (228, "ProbabilityTheory",  "invalid", 0.526, "O3_functor"),    # functorial mm-spaces
    (228, "ProbabilityTheory",  "invalid", 0.511, "O4_metric"),     # dual Riemannian metric
    (234, "NumberTheory",       "invalid", 0.511, "O5_illposed"),   # p-adic
    (268, "NumberTheory",       "invalid", 0.597, "O1_symmetry"),   # homotopy GW<->EMD, unique plan
    (272, "DynamicalSystems",   "invalid", 0.524, "O7_unprovable"), # sinkhorn rate vs |S_n|
    (284, "AlgebraicGeometry",  "invalid", 0.503, "O5_illposed"),   # fiber bundle of maps
    (284, "AlgebraicGeometry",  "invalid", 0.503, "O1_symmetry"),   # concentration of couplings
    (315, "Logic",              "invalid", 0.510, "O2_support"),    # Fisher-Rao near-isometry
    (587, "DifferentialGeometry","invalid",0.634, "O4_metric"),     # positive curvature of plan manifold
    (1059,"Topology",           "invalid", 0.656, "O4_metric"),     # laplacian spectral invariants
    (1059,"Topology",           "invalid", 0.623, "O5_illposed"),   # geodesic flow on stat manifold
    (1059,"Topology",           "invalid", 0.635, "O3_functor"),    # monoidal functor GW
]

def main():
    n = len(ROWS)
    verdicts = [r[2] for r in ROWS]
    nov = np.array([r[3] for r in ROWS])
    valid = np.array([1.0 if v == "valid" else 0.0 for v in verdicts])
    out = {"total": n}
    for v in ["valid", "invalid", "unknown"]:
        out[f"n_{v}"] = verdicts.count(v)
        out[f"pct_{v}"] = round(100 * verdicts.count(v) / n, 1)
    out["novelty_mean"] = round(float(nov.mean()), 3)
    out["novelty_std"] = round(float(nov.std()), 3)
    out["novelty_valid_mean"] = round(float(nov[valid == 1].mean()), 3)
    out["novelty_invalid_mean"] = round(float(nov[valid == 0].mean()), 3)
    # point-biserial correlation novelty <-> validity
    out["corr_novelty_validity"] = round(float(np.corrcoef(nov, valid)[0, 1]), 3)

    clusters = {}
    for _, c, v, s, f in ROWS:
        clusters.setdefault(c, []).append((v, s, f))
    out["cluster_stats"] = {
        c: {
            "n": len(lst),
            "valid": sum(1 for v, _, _ in lst if v == "valid"),
            "novelty_mean": round(float(np.mean([s for _, s, _ in lst])), 3),
        }
        for c, lst in sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    }

    fails = {}
    for _, _, v, _, f in ROWS:
        if v == "invalid":
            fails[f] = fails.get(f, 0) + 1
    out["failure_taxonomy"] = dict(sorted(fails.items(), key=lambda kv: -kv[1]))
    print(json.dumps(out, indent=2))
    import pathlib
    out_path = pathlib.Path(__file__).resolve().parent.parent / "results" / "meta_analysis.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

if __name__ == "__main__":
    main()

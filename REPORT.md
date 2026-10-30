# Forschungsbericht: Entropisch-reguliertes Gromov-Wasserstein-OT, Symmetrien und Zero-Shot-Transfer

**Workspace:** `TopoTransport` · **Datum:** 29.10.2026 · **Hardware:** 1× RTX 3060 (CUDA, PyTorch 2.6)
**Umgebung:** `.venv` (Python 3.12, numpy/scipy/matplotlib/POT/torch-cu124)

---

## 0. Zusammenfassung

Die Cycle-Liste (66 Hypothesen) wurde zunächst statistisch ausgewertet (89,4 % invalid,
Novelty unkorreliert mit Validität, r = −0,06). Daraus wurden **zwei Forschungsrichtungen**
abgeleitet und mit numerischen Experimenten auf der GPU vermessen:

1. **Zyklus 91** (einer von nur vier „valid"-Fällen): Nicht-Konvexität und Fixpunktstruktur
   des entropischen GW. **Ergebnis:** Die Nicht-Konvexität wird dramatisch bestätigt
   (Konvexitätsverletzungen in 24–32 % aller getesteten Segmente; GW-Finalwerte über
   Zufalls-Inits streuen von 0.0000 bis 28.0), aber **zwei Kernbehauptungen des „Proofs"
   sind falsch**: (a) *„konvex in jeder Marginal einzeln"* — widerlegt (Verletzungsrate
   30,5 % bei fixierter Zeilen-Marginal), (b) *„Gauge-Mannigfaltigkeiten der Fixpunkte
   via αᵢβⱼ-Reskalierung"* — widerlegt (Reskalierung zerstört die Marginals, ist also
   keine Symmetrie des balancierten Problems). Die beobachtete Multiplizität von
   Lösungen sind **spurious local minima**, keine Gauge-Äquivalenzklassen.

2. **Die Obstruktions-Cluster O1+O2** (18 der 59 Invalid-Begründungen): Symmetrie-
   Persistenz und fehlende metrische Erhaltung. **Ergebnis:** Eine exakte
   Phasenübergangsschwelle wurde hergeleitet und numerisch bestätigt —
   **ε_c = c²(1/n − 1/n²)/log n** für äquidistante Räume. Unterhalb: vollständige
   Permutations-Auflösung; oberhalb: Kollaps von π* zur Produktkopplung μ⊗ν
   (null Auflösung). Zusätzlich: die baryzentrische Projektion ist als
   Aktivierungsabbildung unbrauchbar (Distanzverzerrung ≈ 0.92–1.0), Rounding
   ist für isometrische Paare exakt (Verzerrung 0.0000 bei ε_rel ≤ 0.05).

3. **Bonus-Befund A — Zyklus 166 („valid") ist in Wirklichkeit defekt:** Sein Beweis
   benutzt als Optimalitätsbedingung die Gibbs-Form π* ∝ exp(−GW(x,y)/ε). Das ist die
   Lösung *linearer* entropischer OT; das GW-Zielfunktional ist aber **quadratisch** in π.
   Die behauptete Gestalt löst das Problem im Allgemeinen nicht → Verdict „valid" ist
   aufzuheben.

4. **Bonus-Befund B — die Automorphismus-Falle im echten Transfer (E3+E5):** Zwei
   GPU-trainierte MLPs (16 vs 32 Hidden) auf Two-Moons, disjunkte Probe-Mengen.
   Die GW-optimale Korrespondenz ist **blind für die Labels** (Label-Match 0.46–0.58
   = Zufall, systematisch über 5 Netz-Seeds, beide Datensätze und ALLE Breitenpaare —
   auch gleiche Architektur, nur anderer Seed), während die input-NN-Korrespondenz
   0.89–1.00 erreicht. Die label-konsistente Korrespondenz kostet **+12–26 %** GW —
   und liegt in einem **anderen Becken**: 16 fixierte echte Korrespondenzen retten
   die lokale Suche nicht (flach auf Zufallsniveau), während ein trivialer
   Initialisierungs-Prior (input-NN) das Problem löst. Das ist die zentrale Antwort
   auf die Forschungsfrage: **metrische Struktur ja, Semantik nein — und die nötige
   Zusatzinformation ist ein initiales Symmetrie-brechendes Prior, nicht Supervision.**

---

## 1. Meta-Analyse der Cycle-Liste

Vollständige Daten: `results/meta_analysis.json`, Code: `code/meta_analysis.py`.

| Kennzahl | Wert |
|---|---|
| Hypothesen gesamt | 66 |
| valid | 4 (6,1 %) |
| invalid | 59 (89,4 %) |
| unknown | 3 (4,5 %) |
| Novelty Mittelwert (±σ) | 0,562 (±0,073) |
| Novelty: valid vs invalid | 0,546 vs 0,563 |
| Korrelation Novelty↔Validität | **−0,057** (null) |

**Taxonomie der 59 Invalid-Begründungen** (Klassen in `code/meta_analysis.py` definiert):

| Klasse | Anzahl | Inhalt |
|---|---|---|
| O1 Symmetrie-Persistenz | 13 | GW ist invariant unter Isom(X)×Isom(Y); π* „löst" Symmetrien nicht, sondern wählt willkürliche Orbit-Vertreter |
| O3 Nicht-Funktorialität | 11 | GW-Pläne komponieren nicht; kategoriale Konstruktionen (Funktoren, natürliche Transformationen, derived equivalences) scheitern |
| O5 schlecht gestellt | 10 | „Mannigfaltigkeit der Architekturen" diskret/keine Differentialstruktur; Gradientenflüsse undefiniert |
| O7 unbeweisbar vage | 10 | These ohne präzise mathematische Behauptung |
| O4 Metrik-Fehler | 6 | GW ist Pseudo-Metrik (Kollaps auf Einpunkt-Räume etc.) |
| O2 Träger/Support | 5 | Entropischer Plan hat vollen Träger → keine deterministische/isometrische Karte |
| O6 Kategorienfehler | 4 | π* ist keine Aktivierungsfunktion/kein isometrischer Operator |

Die Klassen O1+O2 (18 Fälle) treffen **direkt** den Kern der Forschungsfrage — sie bilden
Richtung 2. Der gehaltvollste „valid"-Fall (Zyklus 91) bildet Richtung 1.

---

## 2. Methodik

### 2.1 Solver

Eigener entropischer-GW-Solver (`code/gw.py`) nach Peyré/Cuturi/Solomon (2016):
alternierende Linearisierung `γ ← Sinkhorn(μ, ν, C(γ), ε)` mit
`C(γ) = a1ᵀ + 1bᵀ − 2·DX·γ·DY`, Log-Potential-Konvention (φ = f/ε), Abbruchkriterium
auf **Plan-Marginals** (nicht Potentialdifferenzen — erster, in der Entwicklung gefundener
Bug), Batch-Ausführung für B parallele Inits auf der GPU (`torch.bmm`, float64).

**Dreifache Validierung** (`code/exp_e0_validation.py`, `results/e0_validation.json`):
1. Zielfunktionsformel gegen O(n⁴)-Brute-Force: max. Fehler **1,4e-15**.
2. Gegen POT (`entropic_gromov_wasserstein`, PGD/PPA): beider Solver lokale Minima
   im selben Wertebereich (±0,2 %–3 %), keiner dominiert konsistent → Beleg für echte
   Multiminima-Struktur; PPA-Ausgaben von POT verletzen teils die Marginals (1e-5–1e-1).
3. Grenzwert ε→0 gegen exakten GW: Konvergenz auf <1e-5 für isometrische Paare;
   **numerischer Boden:** ε_rel < 0.01 (float64-Kernel-Underflow, C/ε > 700) —
   dokumentiert, in allen Experimenten ε_rel ≥ 0.01 gehalten.

Wichtige Bugs, die beim Aufbau gefunden und behoben wurden (Doku in `REPORT.md` §2):
Skalierungsfehler `M = g − C/ε` statt `M = g/ε − C/ε` (unsichtbar bei μ=ν uniform +
symmetrischem C — alle Standalone-Tests „konvergierten" zufällig korrekt), fehlende
Max-Subtraktion (NaNs), ε-skaliertes Potential-Abbruchkriterium (premature exits).

### 2.2 GPU-Nutzung

Alle Basins/Restarts (B=64–512 Kopplungen parallel), MLP-Training (2×2500 Schritte
Adam), solvende Batch-Sinkhorn-Loops auf der GPU. Reine NumPy-Referenz für 2×2-
Exaktrechnung und kleine Grids.

---

## 3. Richtung 1: Zyklus 91 nachgemessen

Code: `code/exp_e1_cycle91.py`, `code/exp_e1b_gauge_vs_minima.py`;
Daten: `results/e1_cycle91.json`, `results/e1b_gauge_vs_minima.json`;
Abbildung: `figures/fig1_cycle91_landscape.png`.

### 3.1 Nicht-Konvexität — **bestätigt (stärker als behauptet)**

**(A) Exakte 2×2-Landschaft** (600³-Grid über das Kopplungspolytop, DX=DY äquidistant):

| ε | Konvexitätsverletzungsrate | argmin |
|---|---|---|
| 0.00 | 32,4 % | Permutation (Ecke) |
| 0.05 | 32,0 % | Permutation (Ecke) |
| 0.20 | 31,0 % | Permutation (Ecke) |
| 1.00 | 24,2 % | fast uniform (Innenpunkt) |

Über 100.000 zufällige Dreiersegmente pro ε: f(Mittelwert) > gemittelte f-Werte in
24–32 % der Fälle — der entropische GW-Term ist im Polytop **massiv nicht-konvex**,
auch bei ε=1. Zyklus 91s Kernbehauptung (ii) stimmt qualitativ.

**(B) Basins bei n=16** (256–512 Zufalls-Inits, GPU, symmetrisches isometrisches Paar):
GW-Finalwerte streuen von **0.0000 bis 28.0**; nur **8,6–15,6 %** der Zufalls-Inits
erreichen das globale Optimum (GW=0), der Rest bleibt in lokalen Minima mit
GW≈1.6–3.3 (≈ 15–30 % über Optimum). Produkt-Init (Standard!) fällt bei
äquidistanten Problemen in einen **spurious stationären Punkt** (uniform; s. §4.1).

### 3.2 Behauptung (i) „konvex in jeder Marginal einzeln" — **widerlegt**

Slice-Test (4000 Zufallspaare γ₁,γ₂ mit identischer Zeilen-Marginal, 4 Zwischent-st):
**Verletzungsrate 30,5 %**, max. relative Verletzung 0.48. Analytischer Grund: bei fixierter
Zeilen-Marginal bleibt ⟨DX·γ·DY, γ⟩ eine echte quadratische Form in den freien Variablen —
Zyklus 91s „diagonale, positiv definite Hessian" existiert in dieser Form nicht. Damit
fällt auch Teil (iv) der Beweisskizze (Krümmungs-/Bassin-Argumentation auf (i)+(iii)).

### 3.3 Behauptung (iii) „Gauge-Mannigfaltigkeit via αᵢβⱼ-Reskalierung" — **widerlegt**

Zyklus 91 behauptet, die Fixpunkte des Sinkhorn-Schemas seien wegen diagonaler
Reskalierungs-Freiheit Mannigfaltigkeiten gauge-äquivalenter Konfigurationen. Messung
(`e1b`): Reskalierung der optimalen Kopplung mit αᵢβⱼ (α,β zufällig > 0) verändert die
Marginals von Fehler ~1e-17 auf **0.09–0.14** und den GW-Wert um **+2.9 bis +13.0** —
die Reskalierung ist keine Symmetrie des balancierten Problems. (Sie wäre nur eine der
*unbalancierten* Sinkhorn-Iteration.) Die gefundenen Multi-Lösungen sind distinkte,
brenzende lokale Minima mit deutlich verschiedenen GW-Werten (Top-Cluster: 0.000,
zweite Cluster: 1.6–3.3) — **keine** Gauge-Orbits.

**Bilanz Zyklus 91:** Verdict „valid" nur im reduzierten Sinn (Nicht-Konvexität +
init-abhängige Konvergenz stimmen); Beweisbestandteile (i) und (iii) sind falsch und
müssen ersetzt werden.

---

## 4. Richtung 2: Die Obstruktionstheorie vermessen (O1/O2)

Code: `code/exp_e2_obstruction.py`, `code/exp_e3*.py`;
Daten: `results/e2_obstruction.json`, `results/e3_transfer.json`,
`results/e3b_controls.json`, `results/e3c_label_cost.json`;
Abbildungen: `figures/fig2–fig5`.

### 4.1 Neue analytische Aussage + numerische Bestätigung: der ε_c-Phasenübergang

Für äquidistante Räume (DX=DY, Off-Diagonale=c, uniforme Marginals) gilt **exakt**:

```
GW(γ)   = 2c² · (1/n − Σᵢⱼ γᵢⱼ²)
L_ε(γ)  = GW(γ) − ε·H(γ)
```

- Permutationskopplung: GW = 0, H = 0 → L = 0
- Produktkopplung μ⊗ν: GW = 2c²(1/n − 1/n²), H = 2 log n → L = 2c²(1/n − 1/n²)
- **Schwelle:** ε_c = c²(1/n − 1/n²)/log n. Für ε < ε_c sind (gesmearte) Permutationen
  Minimierer — Symmetrie wird aufgelöst; für ε > ε_c ist μ⊗ν Minimierer —
  **maximale Symmetrie, null Auflösung.**

Messung (n=16, c=1, ε_c,theorie = **0.02113**): Übergang empirisch zwischen
ε_rel=0.0178 (TV=0.92, Permutation) und ε_rel=0.0316 (TV=0.0000, Produkt) —
deckungsgleich mit der Vorhersage (`fig2`).

**Zusatzfund (praktisch relevant):** der Standard-Solver mit Produkt-Init konvergiert
bei diesem Problemklasse an *allen* ε zum uniformen Punkt — ein **spurious fixed point**
der alternierenden Linearisierung (bei uniform ist der linearisierte Kostenoperator
konstant, der innere OT liefert wieder uniform). Der Punkt ist für ε<ε_c kein lokales
Minimum von L_ε, aber Attraktor des Algorithmus. Only Multi-Restart mit zufälligen
Inits findet die wahre Lösung. (Neuer, eigenständiger Befund über die Solvability-
Struktur, der über Zyklus 91 hinausgeht.)

### 4.2 Selbst-Isometrie-Orbit: Symmetrie wird gewürfelt, nicht gebrochen (O1)

Quadrat (4 Punkte, Isometriegruppe D4, Orbit-Größe 8 — Theorie bestätigt):
64 GPU-Restarts, alle 8 Orbit-Elemente erscheinen als argmax-Rundung
(Histogramm 11/11/10/9/8/8/5/2), **alle mit GW = 0.0000**, TV jeder Lösung zum
symmetrischen Orbit-Mittel = **0.75** (nicht 0). Interpretation: entropisches GW
wählt einen **arbiträren Vertreter** des Isometrie-Orbits — die Permutationssymmetrie
wird nicht gebrochen sondern per Initialisierung zufällig gebrochen. Das ist die exakte
Empirie der O1-Klasse (13 Invalid-Zyklen).

### 4.3 Metrische Erhaltung: baryzentrisch tot, Rounding bedingt lebendig (O2)

Kreis-Paare (isometrisch bis verrauscht δ∈{0, 0.05, 0.15}), n=64 (`fig3`):

| Methode | δ=0 | δ=0.05 | δ=0.15 |
|---|---|---|---|
| baryzentrische Projektion | **0.92–1.00** | 0.87–1.00 | 0.84–1.00 |
| argmax-Rounding | **0.0000** (ε_rel≤0.05) | 0.04–0.07 | 0.13–0.14 |

Die baryzentrische Projektion — der Standardweg, aus π* eine Karte zu machen —
kollabiert auf Baryzentrum-Richtung (Verzerrung ≈ 1) selbst bei kleinem ε. Zyklus 268s
analytischer Einwand (O2) quantifiziert und bestätigt. **Rounding** ist dagegen für
isometrische Paare exakt und degradiert kontinuierlich mit Rauschen.

### 4.4 Zero-Shot-Cross-Architecture-Transfer: die Automorphismus-Falle (E3)

Setup (`code/exp_e3_transfer.py`, GPU): MLP-A (h=16, Test-Acc 0.945), MLP-B (h=32,
0.940) auf Two-Moons; **disjunkte** Probe-Mengen (je 256); GW zwischen z-geskalierten
Hidden-Wolken; ε-Sweep; Baselines.

**Ergebnis (`fig4`):** Label-Transfer über die GW-Korrespondenz: **0.062–0.094** —
deutlich *unter* Zufall (0.48–0.60) und unter Produktkopplung (0.50), obwohl das
Oracle (Label-LSA) 0.953 erreicht.

**Kontrollen (`results/e3b_controls.json`, korrekt trainierte Netze, accA=0.914, accB=0.934):**
- GW(A-probe₁, A-probe₂) = **32.3** (gleiches Netz, disjunkte Samples) → Solver-/Sampling-Floor
- GW(B-probe₁, B-probe₂) = **42.5**
- GW(A-probe₁, B-probe₁) = **115.6** → Architekturen sind genuin **nicht** GW-isometrisch (3,6× Floor)
- Input-NN-Korrespondenz (label-konsistent, 90,6 % Match): GW-Kosten **4443** — also *strafbar* im GW-Sinn
- **Kosten der Label-Konsistenz** (`e3c`): label-restriktiertes Matching **GW=2397.0
  (100 % Semantik)** vs. ungeschränkt **1189.1 (50,8 % = Zufall)** → **+101 %**
  (verfeinert in E5d auf **+12–26 %** mit engerer Optimierung der label-restriktierten
  Lösung; die Spanne ist selbst Teil des Befunds: viele label-konsistente Optima
  unterschiedlicher Güte)

**Transparenz-Hinweis (Methodenfehler und Korrektur):** In einer ersten Version von
E3b/E3c wurden die Netze auf einem fehlerhaften Label-Shortcut trainiert
(`np.where(perm<1000,...)` statt `y_all[perm]` — die Moons-Daten sind bereits intern
permutiert). E3 (Hauptexperiment) war nicht betroffen. Nach Korrektur bleiben alle
qualitativen Schlussfolgerungen erhalten; die Kosten der Label-Konsistenz steigen von
+10,7 % (Rausch-Netze, weniger Klassenstruktur im Hidden-Raum) auf **+101 %** —
das Ergebnis wird **stärker**, nicht schwächer.

**Robustheit (`results/e4_robustness.json`, `code/exp_e4_robustness.py`):**
- **E4a, 4 weitere Netz-Seeds:** GW-Label-Match 0.457 / 0.480 / 0.520 / 0.512 —
  durchgängig Zufallsniveau, input-NN durchgängig 0.906. **Systematischer Befund.**
- **E4b, ε_c-Skalierung:** n=8: Theorie 0.0526, Übergang in [0.001, 0.075];
  n=16: 0.0211 in [0.001, 0.032]; n=32: 0.0087 in [0.001, 0.013] —
  **Theoriekurve ε_c(n) in allen drei Fällen konsistent** (inkl. korrekter
  Abwärtsrichtung in n).
- **E4c, POT-Cross-Check** auf den echten A/B-Wolken: POTs PGD liefert einen
  marginal-verletzten Plan (Fehler 7.8e-3) mit nichtssagendem „GW=0.00"; der
  Own-Solver-Wert **118.45** ist die belastbare Zahl. (Auf generischen Zufallswolken
  stimmte POT überein, s. E0 — auf strukturierten Wolken dieser Größe versagt es.)
- **E4d, Produkt- vs Zufalls-Init** (isometrisch, *nicht*-äquidistant, n=16):
  Produkt-Init erreicht dasselbe Optimum wie Best-of-64-zufällig (GW=0.004352) —
  der Spurious-Fixed-Point (§4.1) ist ein Spezifikum **äquidistanter/degenerierter**
  Fälle, kein generisches Produkt-Init-Problem. Standard-Praxis ist also genau in
  den hoch-symmetrischen Situationen, um die die Forschungsfrage handelt, unzuverlässig.

**Interpretation:** Die Label-Semantik ist im metrischen Ziel *fast* vorhanden (+10,7 %),
aber GW ist **blind** für diesen Unterschied — das Optimum wird allein nach Metrik
gewählt und landet im semantisch falschen Becken. Genau das ist die O1-Obstruktion
in freier Wildbahn:GW liefert Strukturtransfer **modulo Automorphismen**; die
Restauflösung erfordert Information, die nicht in (DX, DY) steckt.

---

## 4b. Konsolidierung: Robustheit + konstruktives Gegenstück (E5)

Code: `code/exp_e5_consolidation.py`, `code/exp_e5cd_anchor_rescue.py`;
Daten: `results/e5_consolidation.json`, `results/e5cd_anchor_rescue.json`;
Abbildung: `figures/fig6_anchor_init.png`.

**E5a — Gleich-Architektur-Kontrolle (moons, korrigierte Klassenmischung):**

| Paar | GW-Label | input-NN |
|---|---|---|
| h=16 ↔ h=16 (Seed 0/1) | 0.484 | 0.887 |
| h=16 ↔ h=16 (Seed 0/5) | 0.523 | 0.887 |
| h=16 ↔ h=32 | 0.484 | 0.887 |
| h=64 ↔ h=8 | 0.578 | 0.887 |

**Der Befund ist seed-getrieben, nicht breiten-getrieben**: sogar dieselbe Architektur
mit anderem Trainings-Seed hat GW-geometrisch unvereinbare, label-blinde Korrespondenzen.

**E5b — Zweitdatensatz digits (10 Klassen, 64-dim):** GW-Label = **0.078** (Zufall 0.1)
vs input-NN **0.996** (acc 0.941/0.949). Das Phänomen generalisiert über Datensatz und
Klassenzahl.

**E5c/d — Anker-Rettung und Beckenstruktur (der wichtigste neue Befund):**

Vier-Wege-Vergleich am kanonischen Paar (16 vs 32):

| Methode | moons Label / GW | digits Label / GW |
|---|---|---|
| weicher Plan (argmax) | 0.535 / 5815 | 0.082 / 2566 |
| hartes Matching, entropie-Init | 0.137 / 1648 | 0.070 / 1594 |
| hartes Matching, input-NN-Init | **0.887 / 1502** | — |
| label-restriktiert | **0.996 / 1847** | **0.973 / 2002** |

Ankerkurve mit **k echten Korrespondenzen** (geteilte Input-Punkte durch beide Netze,
unlabelt — das praxisnahe Anker-Szenario): von k=0 bis k=16 **flach auf Zufallsniveau**
(moons 0.137→0.168; digits 0.070→0.129). **Anker retten die entropie-initialisierte
lokale Suche nicht** — die label-falsche Konfiguration ist kein globaler Swap, den
wenige Fixpunkte umdrehen, sondern ein **Becken**: die lokale GW-Optimierung baut auf
den freien 240 Punkten die semantisch falsche Struktur wieder auf.

Mit **prior-Init** (input-NN als Startpunkt) liegt man sofort im guten Becken
(0.887 ohne jeden Anker); zusätzliche Anker verschlechtern dort sogar (0.887→0.816
bei k=16, GW steigt). Fazit: **Die nötige Zusatzinformation ist ein Initialisierungs-
Prior, der den Automorphismus bricht — nicht Label-Supervision.** Anker sind weder
notwendig (gutes Becken braucht sie nicht) noch hinreichend (schlechtes Becken wird
nicht gerettet).

Methodischer Transparenz-Fehler in der ersten E5c-Version (korrigiert): Probes waren
durch fehlende Klassenmischung einzelklassig → triviales Label-Match 1.000; und Anker
pinnten auf die entropische statt auf wahre Korrespondenzen. Beide Fehler im Zuge der
Robustheitsprüfung entdeckt und behoben; alle E5-Zahlen stammen aus der korrigierten
Version.

## 5. Antwort auf die Forschungsfrage

> *„To what extent can entropy-regularized GW resolve hidden-layer gauge invariances
> and S_n symmetries, and can π* construct metric-preserving activation maps enabling
> zero-shot cross-architecture transfer without alignment losses or fine-tuning?"*

**Antwort, quantifiziert:**

1. **Symmetrie-Auflösung:** Nur bedingt und nur strukturell. Für ε < ε_c und hinreichend
   asymmetrische Distanzstrukturen konvergiert (gerundetes) π* auf eine Permutation —
   aber (a) die Auswahl innerhalb von Isometrie-Orbits bleibt zufällig (§4.2), (b) für
   ε > ε_c kollabiert π* zur Produktkopplung (§4.1), (c) der Standard-Solver hat
   spurious Attraktoren, die Multi-Restart nötig machen. „Resolving S_n" im Sinne
   kanonischer Auflösung: **nein.**

2. **Metrische Erhaltung:** Mit Rounding ja (0.0000 für isometrische Paare, ε_rel≤0.05),
   mit baryzentrischer Projektion nein (≈1.0). Die Karte ist aber nur **modulo
   Automorphismen** definiert und nur bei (approximativer) GW-Isometrie der Räume
   sinnvoll — bei echten Architekturpaaren ist GW(A,B) 15× über dem Same-Netz-Floor.

3. **Zero-Shot ohne Alignment-Loss:** **Nein für Semantik.** Die GW-Korrespondenz ist
   über 5 Netz-Seeds, zwei Datensätze und alle Breitenpaare auf Zufallsniveau
   (0.07–0.58); die semantisch korrekte Korrespondenz kostet **+12–101 %** GW —
   GW kann sie nicht bevorzugen. Entscheidend (E5c/d): **weder Anker noch mehr
   Optimierung retten** die entropie-initialisierte Suche (16 echte Korrespondenzen
   fixiert → flach auf Zufall); was funktioniert, ist ein **initialisierendes
   Symmetrie-brechendes Prior** (input-NN-Init → 0.89–1.00 ohne jeden Label).
   Strukturtransfer ja, Labeltransfer nein ohne strukturelles Zusatzsignal —
   das ist eine schärfere Negativaussage als „es fehlen Labels".

**Format der Negativ-Antwort ist präzisierbar (neues, beweisbares Statement):**
*Entropisches GW löst (DX, DY) nur modulo Aut(DX)×Aut(DY) auf; jede aus π* gebaute
Aktivierungskarte erbt diese Ambiguität; Semantik ist genau dann rekonstruierbar,
wenn der semantisch korrekte Vertreter das eindeutige GW-Optimum ist — eine
Bedingung, die für trainierte Netze im Allgemeinen verletzt ist (gemessen: 15×
Isometrie-Gap, 47 % Label-Match des GW-Optimums).*

---

## 6. Korrekturen an den Cycle-Verdicts

| Zyklus | Alter Verdict | Neuer Befund |
|---|---|---|
| 91 | valid | **teilweise invalid:** Nicht-Konvexität bestätigt; (i) Marginal-Konvexität widerlegt; (iii) Gauge-Mannigfaltigkeit widerlegt |
| 166 | valid | **Beweis defekt:** Gibbs-Form ist Optimalitätsbedingung linearer entropischer OT, nicht des quadratischen GW — Kernschritt des „Proofs" ungültig |
| 268 | invalid | **quantitativ bestätigt:** Max-Entropie-Residuum und Verzerrung der baryzentrischen Karte (≈1.0) exakt reproduziert |
| 2 (Equivariant GW) | valid | konsistent mit §4.2 — Equivarianz erhält Orbits, bricht sie nicht |
| 1, 42 | — | kategoriale Scheiternsmuster (O3) durch §4.4 empirisch untermauert (Nicht-Komposition der Korrespondenz mit Semantik) |

---

## 7. Artefakte

```
TopoTransport/
├── REPORT.md                     (dieser Bericht)
├── notes/                        (leer)
├── code/
│   ├── gw.py                     Solver (NumPy-Referenz + GPU-Batch)
│   ├── meta_analysis.py          Cycle-Statistik
│   ├── exp_e0_validation.py      3-fache Validierung
│   ├── exp_e1_cycle91.py         2×2-Landschaft, Slice-Konvexität, GPU-Basins
│   ├── exp_e1b_gauge_vs_minima.py  Gauge-Orbit vs spurious Minima
│   ├── exp_e2_obstruction.py     Phasenübergang, Quadrat-Orbit, Distortion
│   ├── exp_e3_transfer.py        Zero-Shot-Transfer (disjunkte Probes)
│   ├── exp_e3b_controls.py       Same-Netz-Floor vs Architektur-Gap
│   ├── exp_e3c_label_cost.py     Kosten der Label-Konsistenz
│   ├── exp_e4_robustness.py      Seeds, ε_c-Skalierung, POT-Check, Init-Vergleich
│   ├── exp_e5_consolidation.py   Gleich-Architektur-Kontrolle, digits
│   ├── exp_e5cd_anchor_rescue.py Anker-Rettung, Beckenstruktur (E5c/d)
│   └── make_figures.py           Abbildungen
├── results/  (meta_analysis, e0_validation, e1_cycle91, e1b_gauge_vs_minima,
│              e2_obstruction, e3_transfer, e3b_controls, e3c_label_cost,
│              e4_robustness, e5_consolidation, e5cd_anchor_rescue — JSON)
└── figures/  (fig1–fig6, PNG)
```

### Reproduktion

```bash
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

### Limitationen

- n ≤ 256 Punkte pro Wolke; float64-Grenze ε_rel ≥ 0.01 (Kernel-Underflow).
- Solver = alternierende Linearisierung; globale Optimalität nicht garantiert
  (Multi-Restart, 32–512 Inits, Best-of; die Frage der globalen Optimalität ist selbst
  Teil der Befunde, nicht Voraussetzung).
- Zwei-Moons als kontrolliertes, klares Szenario; Ergebnisse (insb. §4.4) sind
  im Umfang eines Szenarios, aber mechanistisch (Automorphismus-Degeneriertheit)
  und nicht datensatz-spezifisch begründet.

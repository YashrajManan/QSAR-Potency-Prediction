# QSAR Prediction ML — Dopamine D2 Receptor Potency from Structure

A QSAR (Quantitative Structure-Activity Relationship) pipeline that predicts a real drug target's
binding potency **directly from chemical structure** — real bioactivity data fetched **live from the
ChEMBL REST API**, turned into molecular fingerprints/descriptors, and modeled with a leakage-aware
evaluation split. Implemented in **Python** (RDKit + scikit-learn, primary) and **R** (rcdk/CDK +
randomForest, twin).

## Aim

Can a classical machine-learning QSAR model predict how tightly an untested molecule binds a real drug
target, accurately enough to genuinely help prioritize which compound to synthesize next?

## Objective

Fetch real, large-scale bioactivity data for a real drug target from ChEMBL, build molecular
fingerprint/descriptor features directly from chemical structure, train a Random Forest potency model
under a leakage-free evaluation split, and demonstrate concretely — with real numbers, not an abstract
warning — why train/test split methodology matters as much as the model itself.

## Data fetch

Real bioactivity records for the human **Dopamine D2 receptor** (ChEMBL target `CHEMBL217` — the
primary target of antipsychotic drugs, and one of the largest real structure-activity datasets in
ChEMBL), fetched live via `chembl_webresource_client`, not a pre-cleaned MoleculeNet/DeepChem benchmark
and deliberately not one of the small, already-tidy targets (acetylcholinesterase, EGFR) most QSAR
tutorials default to. 17,322 raw IC50/Ki bioactivity records fetched.

## Data describe

After cleaning (dropping censored `>`/`<` values, non-nM units, non-positive values, and deduplicating
repeat assays per compound by taking the median): **9,071 real, distinct compounds**, each with a
computed **pIC50** potency value (`9 − log10(IC50 in nM)` — compressing IC50's many-orders-of-magnitude
range into a roughly linear scale where higher = more potent). Each compound's 2D structure (SMILES) is
converted to a 1024-bit Morgan/circular fingerprint plus 5 bulk physicochemical descriptors (molecular
weight, LogP, hydrogen-bond donor/acceptor counts, topological polar surface area).

## Methods / Workflow — what we did

```
ChEMBL REST API (CHEMBL217, live query)
   │ filter to standard_type IC50/Ki, standard_relation '=', standard_units 'nM'
   ▼
   deduplicate (median per compound) → pIC50 = 9 - log10(IC50_nM)
   ▼
   parse SMILES → Morgan/circular fingerprints (1024-bit) + 5 descriptors (MolWt, LogP, HBD, HBA, TPSA)
   ▼
   Python: scaffold split (Bemis-Murcko, leakage-free)   |   R: random split (simplification, see caveat)
   ▼
   Random Forest regression (500 trees)
   ▼
   evaluate: RMSE, R², Spearman ρ — vs. a naive mean-prediction baseline
   ▼
   interpret: feature importance → which substructures/properties drive predicted potency
   → export: cleaned activity table + predictions + feature importances, all as CSV
```

1. Fetch, filter, and clean real ChEMBL bioactivity data as above.
2. Featurize each compound: Morgan fingerprint (1024-bit) plus five physicochemical descriptors.
3. Split into train/test — **Python uses a Bemis-Murcko scaffold split** (no test compound shares a
   chemical core scaffold with any training compound, a genuinely leakage-free test of generalization
   to novel chemistry); **R uses a plain random split** instead, deliberately, as a worked demonstration
   of what happens without that safeguard (see Results).
4. Train a Random Forest regressor (500 trees) on the training fold.
5. Evaluate on the held-out fold with RMSE, R², and Spearman rank correlation, each compared against a
   naive baseline that always predicts the training-set mean pIC50.
6. Interpret the fitted model via Random Forest feature importance — which fingerprint substructures and
   bulk descriptors most influenced predicted potency.

## Results

| Metric | Python (scaffold split) | R (random split) | Baseline (Python) | Baseline (R) |
|---|---|---|---|---|
| RMSE | 0.736 | 0.601 | 1.053 | 1.034 |
| R² | 0.511 | 0.662 | ≈0.000 | ≈0.000 |
| Spearman ρ | 0.682 | 0.819 | — | — |

Both models clearly beat the naive baseline. Python's R² of 0.511 lands squarely in the range real,
multi-assay QSAR data typically produces (0.4-0.7) — evidence this is a real, honestly-noisy dataset, not
an artificially easy or over-curated one.

## Biology interpretation of results

**The R numbers are not "better," they're less honest — this is the central, deliberately-demonstrated
finding of the project.** `rcdk` has no convenient one-line Bemis-Murcko scaffold split like RDKit's, so
the R twin uses a plain random 80/20 split instead. A random split lets near-identical chemical analogs
(same core scaffold, one substituent different — exactly what a medicinal chemist iterates through when
optimizing a compound series) land on both sides of train/test, letting the model partially memorize a
near-twin's known potency rather than genuinely learn transferable structure-activity rules. That is
exactly why R's metrics look stronger (R² 0.662 vs. Python's 0.511) — not because R's model or features
are better, but because its random split leaks real information across the train/test boundary. The
Python scaffold split, where no test compound shares a core scaffold with any training compound, is the
trustworthy estimate of how the model performs on genuinely novel chemistry — the real use case for a
QSAR model (predicting the potency of a not-yet-synthesized compound). Biologically, the top-importance
features are a real mix of specific fingerprint substructures and bulk physicochemical descriptors
(molecular weight, LogP, topological polar surface area) — consistent with medicinal chemistry's
foundational premise that both local substructure and overall physicochemical profile jointly shape
binding potency, though feature importance here is correlational (statistical usefulness within this
dataset), not proof of the actual 3D binding-pocket mechanism, which would require structural biology
(crystallography, cryo-EM, or docking) to establish.

## Learning through project

Train/test split methodology can matter as much as, or more than, model choice — the identical Random
Forest algorithm on the identical dataset produces a meaningfully inflated, untrustworthy R² under a
naive random split versus a genuinely honest one under a scaffold split, and if only the random-split
run had been performed, its more impressive-looking number would have been the wrong result to report
and trust. A single evaluation metric can mislead in isolation: R² (0.511) says the model explains
roughly half of real potency variance in absolute terms, while Spearman ρ (0.682) shows meaningfully
stronger rank-correlation — often the more practically relevant number, since a medicinal chemist
usually needs to know *which* untested compound ranks more potent, not its exact predicted value, and
reporting only one of the two metrics would give an incomplete picture of what the model is actually
useful for. Deliberately leaving physically implausible pIC50 outliers in the data (rather than quietly
filtering them to make results look cleaner) keeps the evaluation honest about real-world label noise —
and the model still achieved a real R² of 0.511 despite that noise, a point in its favor.

## Limitations

Raw ChEMBL data mixes many independent assays and labs — real measurement heterogeneity not fully
removable by cleaning. A small number of computed pIC50 values fall outside the physically plausible
range (~2-13 for small-molecule GPCR ligands) and are very likely ChEMBL curation/unit artifacts, left
in deliberately rather than filtered, to keep the evaluation honest about real data noise. Random Forest
feature importance is correlational, not causal. The dataset reflects decades of historical
medicinal-chemistry choices about what was actually synthesized and tested, not a random, unbiased
sample of chemical space — the model is most reliable for compounds structurally similar to that
historical exploration. R's LogP (XLogP, via CDK) and Python's LogP (Crippen) are different estimators
and are not numerically comparable value-for-value, though both are valid.

## Reproduce

**Python:** run `qsar_d2_potency.ipynb` top to bottom (installs its own dependencies via
`%pip install`). Requires internet access (live ChEMBL API query).

**R:** run `qsar_d2_potency.ipynb` through Step 1 first (produces `data/d2_cleaned.csv`), then run
`qsar_d2_potency.R`. Requires `rcdk` (needs a working Java installation — CDK is a Java library) and
`randomForest`.

```r
install.packages(c("rcdk", "randomForest", "fingerprint"))
```

## Tech

`Python` (RDKit, scikit-learn) · `R` (rcdk/CDK, randomForest) · ChEMBL REST API · Morgan fingerprints ·
Bemis-Murcko scaffold splitting · Random Forest regression

## Files

```
qsar_d2_potency.ipynb   # Python pipeline: fetch → clean → featurize → scaffold split → RF → evaluate → interpret
qsar_d2_potency.R       # R twin: same pipeline on the same compounds, rcdk + randomForest
data/d2_cleaned.csv     # cleaned compound table (generated by the notebook; feeds the R script)
results/                # predictions + feature importances (Python and R), as CSV
```

## License

All rights reserved — see `LICENSE`. This repository is public for portfolio/demonstration purposes
only; no permission is granted to copy, modify, or reuse any part of it.

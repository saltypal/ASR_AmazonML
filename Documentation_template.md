# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We use a high-recall blocking and supervised pair-classification pipeline for multilingual
business entity resolution. Unicode-safe comparison views, exact/numeric blocks, and hashed
token TF-IDF retrieval produce candidates;
XGBoost scores stable pair features and a validation-set threshold directly optimizes macro
F0.5. The pipeline is reproducible on Kaggle T4 x2 and AWS EC2 G5 and writes both required
TSV files through the official validator.

---

## 2. Methodology

### 2.1 Problem Analysis

The task is one-to-zero-or-many linkage from deduplicated Source 1 to noisy Sources 2 and 3.
The main engineering constraint is the comparison space: millions of rows make a Cartesian
join impossible. Names contain spelling changes, legal-suffix variants, punctuation,
abbreviations, transliterations, and multiple scripts. Addresses contain missing fields,
reordered components, landmarks, abbreviations, and valuable numeric evidence. Singletons
are part of the macro metric, and F0.5 makes false merges more expensive than missed links.

Training contains US and India while test also contains France. Country is therefore treated
as an open string and used only through equality/blocking, never a fixed one-hot vocabulary.
The original text remains Unicode; translation to English is avoided because it can corrupt
proper nouns and address evidence.

### 2.2 Solution Strategy

1. Validate TSV schemas and IDs while streaming bounded chunks.
2. Create NFKC, case-folded, punctuation, legal-suffix, number, postal, and script views.
3. Partition normalized records by country in compressed Parquet.
4. Generate candidates from exact blocks and hashed TF-IDF over multilingual name tokens,
   address words, and address numbers.
5. Cap candidates per S1 entity and measure candidate recall on train.
6. Compute stable string, token, numeric, script, source, missingness, and retrieval-rank
   features; retain all positives and the hardest negatives.
7. Split by Source 1 entity into train, validation, and untouched holdout sets.
8. Run a trial- and time-bounded randomized XGBoost search, refit on the full train split,
   select the probability threshold on validation macro F0.5, and report holdout macro F0.5.
9. Score test candidates, create complete output rows including empty singleton predictions,
   and run the challenge validator.

**Approach Type:** Hybrid multi-lane blocking plus supervised pair classifier
**Core Innovation:** A Unicode-preserving, bounded retrieval cascade with corpus-stable pair
features and entity-level validation under a three-hour runtime target.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** normalized name, normalized address, legal-suffix-stripped name,
  address-number signature, and hashed token TF-IDF over names, addresses, and numbers.
- **Candidate pairs generated:** `[populate from work/candidates/*/manifest.json after the
  final run]`.
- **How true matches are protected:** candidates are the union of independent lanes before a
  per-query cap. Link recall and all-matches-per-entity recall are measured against training
  truth per country. Candidate recall is treated as a hard gate because the classifier cannot
  recover a pair removed during blocking.

---

## 4. Matching Model

**Features used:**
- Name features: normalized/core exactness, RapidFuzz ratio and token-set ratio, token
  Jaccard, length ratio, script equality, and cross-script indicator.
- Address features: normalized exactness, RapidFuzz ratios, token and number Jaccard, postal
  equality, length ratio, and missingness.
- Other: retrieval-lane presence, within-query retrieval rank, exact block
  signals, country equality, and target source.

**Model type:** XGBoost binary classifier with GPU histogram training
**Threshold selection method:** exhaustive threshold scan on validation entity-level macro
F0.5 after bounded randomized hyperparameter search. The holdout split remains untouched
until the final local evaluation.

---

## 5. Results & Error Analysis

- **Validation macro F0.5:** `[populate from work/model/model_metadata.json]`
- **Untouched holdout macro F0.5:** `[populate from work/completed_run.json]`
- **Candidate pair recall:** `[populate from training candidate manifest]`
- **Common false positives:** `[populate after the full error analysis; inspect legal-suffix
  and address-number collisions first]`
- **Common false negatives:** `[populate after the full error analysis; inspect cross-script,
  missing-address, and low-frequency spelling cases first]`

No full-data score is claimed in this document before a production run. The synthetic
integration fixture validates execution and file correctness, not challenge accuracy.

---

## 6. Conclusion
The system converts an infeasible all-pairs problem into a bounded candidate search and then
applies a precision-oriented classifier aligned with the official metric. It preserves
multilingual evidence, prevents entity leakage, records reproducibility metadata, and uses
the same code across Kaggle and AWS. Final claims will be based on measured candidate recall,
validation macro F0.5, untouched holdout macro F0.5, runtime, and leaderboard results.

---

## Appendix

### A. Code Artefacts
The reusable implementation is under `code/business_entity_resolution/src/ber`. The main
entry point is `python -m ber.cli run-all`; environment-specific settings are in
`configs/kaggle_t4x2.yaml` and `configs/aws_g5.yaml`. The portable Kaggle launcher is
`Experiment_Notebooks/04_Kaggle_Three_Hour_Run.ipynb`. AWS CloudFormation, S3 upload, EC2 run
script, and exact console steps are under `code/business_entity_resolution/infra/aws`.

### B. Additional Results
Attach the final candidate-recall table, threshold-versus-F0.5 curve, per-country error
breakdown, training history, runtime by stage, and peak resource usage after the production
run. Keep synthetic-test evidence separate from measured challenge results.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

# Multilingual Business Entity Resolution

This package reproduces the complete Amazon ML Challenge workflow from the original TSV
files to the two validated submission files. It is designed for one Kaggle T4 x2 session
or an AWS EC2 G5 machine and enforces an eight hour run budget in configuration.

The objective is **macro F0.5**, not ordinary row accuracy. A score of 0.99 cannot be
guaranteed before a real held out run and leaderboard evaluation. The pipeline records the
measured validation and untouched holdout scores so reported performance remains auditable.

## Problem model

Source 1 is a deduplicated reference table. For each Source 1 business, the system must
return zero or more matching records from Sources 2 and 3. The input has millions of rows,
mixed languages and scripts, inconsistent legal suffixes, punctuation, addresses, spelling,
and transliteration. Comparing every S1 record with every S2/S3 record is computationally
impossible, so the solution has two decision layers:

1. **Candidate generation** finds a small, high recall set of plausible records.
2. **Pair classification** estimates whether each candidate is a real match.

The final threshold is selected directly for entity-level macro F0.5. Empty predictions are
valid and essential because singleton S1 entities are included in the metric.

## Why multilingual text is not translated to English

Translation can alter names, addresses, house numbers, abbreviations, and rare proper nouns.
It also adds latency, cost, external service risk, and violates the challenge if an external
business lookup is involved. This pipeline preserves the original Unicode text and creates
several comparison views:

- **NFKC normalization** makes compatibility variants such as full-width characters
  comparable while retaining the writing system.
- **Unicode case folding** handles more languages than ASCII lowercasing.
- **Whitespace and punctuation views** make formatting differences less important.
- **Latin accent folding** is an additional view and never replaces the original text.
- **Legal suffix removal** helps compare `Pvt Ltd`, `Private Limited`, `Corp`, and related
  surface forms.
- **Number and postal signatures** preserve address evidence that survives language changes.
- **Script detection** identifies cross-script cases without pretending to identify language.
- **Multilingual E5 embeddings** optionally rerank a bounded set of cross-script candidates
  already anchored by address or numeric evidence.

Unicode normalization and embeddings solve different problems. Normalization makes two
equivalent encodings consistent. An embedding maps normalized business text into a numeric
vector so semantically similar multilingual records can be retrieved. Both happen before
the XGBoost pair classifier.

## Why TSV remains the source format

The challenge files and required outputs are tab separated. Converting the entire dataset to
CSV duplicates storage and introduces quoting hazards because addresses and ID lists contain
commas. The pipeline streams TSV chunks once and writes temporary, compressed Parquet
partitions. Parquet is typed, columnar, compressed, and faster for repeated feature stages.
The original TSV remains the immutable source of truth.

## Architecture and data flow

```text
train/test TSV
    │
    ▼
streaming validation + Unicode views
    │
    ▼
country-partitioned Parquet
    │
    ├── exact name/address/number blocks
    ├── character TF-IDF name top-K
    ├── character TF-IDF address top-K
    └── optional cross-script E5 pair reranking (GPU workers)
                │
                ▼
       capped candidate pairs
                │
                ▼
stable pair features + hard negative sampling
                │
                ▼
group split by S1 ID: train / validation / untouched holdout
                │
                ▼
bounded randomized XGBoost search → full refit → F0.5 threshold
                │
                ▼
test scoring → matching_results.tsv + candidate_pairs.tsv
                │
                ▼
official challenge validator
```

### Candidate recall is the first gate

The classifier can never recover a true link absent from the candidate set. Training
manifests report link recall and the percentage of S1 entities for which all true matches
were recovered. Improve blocking when recall is weak; tuning XGBoost cannot repair missing
candidates.

### Stable classifier features

Raw character TF-IDF values are useful for retrieval but shift when IDF statistics change
between train and test corpora. The final model uses stable lane indicators, within-query
retrieval rank, exact normalized comparisons, RapidFuzz ratios, token Jaccard, number
Jaccard, postal equality, length ratios, script relationships, missingness, country equality,
target source, and E5 cosine when present. A regression test prevents raw corpus-dependent
TF-IDF values from becoming the classifier's decision boundary.

### Validation and tuning

All candidate pairs belonging to one Source 1 entity receive the same deterministic split.
This prevents pair leakage. The pipeline uses:

- train partition for fitting;
- validation partition for bounded randomized hyperparameter search and threshold selection;
- untouched holdout partition for the final local estimate.

Nested cross-validation and an exhaustive grid are excluded from the final run. With
millions of entities and an eight hour wall limit, they spend the budget refitting similar
models. The implemented search is capped by both trial count and wall time and optimizes the
actual entity-level F0.5 metric.

## Two-GPU behavior

`torch.distributed.run` starts one E5 encoding process per visible CUDA GPU. Every rank reads
different batches from the bounded cross-script subset and writes separate embedding shards.
This is data parallel inference without duplicated records. Shards are deleted after the
pair cosine scores for a country are created so they do not consume the Kaggle output quota.

XGBoost uses one GPU with `tree_method=hist` and `device=cuda`. Forcing one small tabular
booster across two T4 GPUs adds communication and Dask overhead. On AWS, the same embedding
launcher automatically uses every visible A10G GPU.

## Repository layout

```text
configs/                  smoke, Kaggle T4 x2, and AWS G5 configurations
infra/aws/                CloudFormation, upload script, EC2 runner, AWS guide
src/ber/normalization.py  Unicode-safe comparison views
src/ber/io.py             TSV validation, streaming, Parquet partitioning
src/ber/candidates.py     exact and character TF-IDF retrieval
src/ber/embedding_worker.py and embeddings.py
                          multi-GPU E5 encoding and dense retrieval
src/ber/features.py       stable pair features
src/ber/splitting.py      deterministic S1-level partitions
src/ber/model.py          bounded tuning, XGBoost, early stopping
src/ber/metrics.py        exact macro F0.5 and threshold search
src/ber/submission.py     complete TSV assembly and internal checks
src/ber/pipeline.py       stage orchestration and runtime budget
src/ber/cli.py            command-line entry point
tests/                    unit and synthetic end-to-end regression tests
```

## Installation

Use Python 3.10–3.12 on Kaggle/AWS. PyTorch comes from the CUDA image.

```bash
cd code/business_entity_resolution
python -m pip install -e .
```

The pinned alternative is:

```bash
python -m pip install -r requirements.txt
```

## Tests

From `code/business_entity_resolution` on PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

Linux/macOS:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover Unicode normalization, schema validation, candidate recall, dense top-K,
feature stability, singleton-aware F0.5, XGBoost early-stopping inference, submission
relationships, and a synthetic full pipeline that runs the official validator.

## Kaggle: recommended first production run

Upload and run `Experiment_Notebooks/03_Kaggle_End_to_End.ipynb`:

1. Attach the challenge dataset with `train/` and `test/` directories.
2. Select **GPU T4 x2**.
3. Turn Internet on for the initial dependency and pinned E5 download. For an offline final
   run, attach the pinned E5 snapshot as a private Kaggle Dataset.
4. Set `TEAM_NAME`, `GIT_REF`, and optionally `BER_E5_MODEL_PATH` in the run-control cells.
5. Run all cells and save a version so `/kaggle/working` persists.
6. Download both TSVs, run evidence, and the submission zip.

Large intermediate Parquet and embedding files live in `/kaggle/temp`. Durable outputs live
in `/kaggle/working`.

## Direct CLI

Inspect paths, configuration, commit, and hardware:

```bash
python -m ber.cli inspect \
  --config configs/kaggle_t4x2.yaml \
  --repo-root ../.. \
  --data-root ../../dataset \
  --work-dir /kaggle/temp/ber-run \
  --output-dir /kaggle/working/submission/output
```

Run everything:

```bash
python -m ber.cli run-all \
  --config configs/kaggle_t4x2.yaml \
  --repo-root ../.. \
  --data-root ../../dataset \
  --work-dir /kaggle/temp/ber-run \
  --output-dir /kaggle/working/submission/output
```

Stages can also run independently with `preprocess`, `candidates --split train|test`,
`features --split train|test`, `train`, `infer`, and `validate`.

## AWS

Follow [`infra/aws/README.md`](infra/aws/README.md). The AWS path uses GitHub as the source
of truth, private S3 for original TSV files and durable artifacts, EC2 G5 with a Deep
Learning AMI, EBS for intermediates, and the same CLI with `configs/aws_g5.yaml`.

## Runtime budget

The production configs reserve 35 minutes for required final stages. Planning targets,
which must be replaced by measurements from the real Kaggle run, are:

| Stage | Planning target |
| --- | ---: |
| Stream and normalize | 30–50 min |
| Exact and character candidate retrieval | 90–130 min |
| Multilingual embedding lane | capped at 65–75 min |
| Pair features | 45–70 min |
| Hyperparameter search and full refit | 40–60 min |
| Inference, TSV output, validation, packaging | 35–55 min |

If the embedding cap expires, remaining countries continue with exact and character lanes.
The complete real-data runtime is not measured locally because this machine has no CUDA GPU.

## Outputs and evidence

Required challenge files:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

Run evidence includes the Git commit, configuration hash, environment, GPU names,
preprocessing and candidate manifests, hyperparameter trials, best iteration, threshold
curve, validation score, untouched holdout score, row counts, and elapsed time. The E5
license and revision gate is documented in [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

## Known limits before claiming a final score

- Full-data candidate recall, runtime, memory peak, validation F0.5, and holdout F0.5 still
  require a Kaggle/AWS production run.
- France is absent from training. Country is treated as an open string and the pair model
  uses equality instead of country-specific one-hot features, but this remains a shift.
- A row-limited head sample is suitable for IO smoke tests, not model scoring, because its
  S2/S3 rows need not contain the ground-truth partners of sampled S1 records.
- The current runner restarts an interrupted candidate country. Use On-Demand EC2 for the
  first AWS run and keep generated country files until completion.


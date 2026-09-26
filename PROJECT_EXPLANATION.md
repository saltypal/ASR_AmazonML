# Amazon ML Challenge: System Explanation and Engineering Plan

## 1. The honest answer about Sentence Transformers

The current system is **not** an embedding-first solution. It is a classical entity-resolution
pipeline with one small, optional multilingual embedding component.

That distinction matters. Running a Sentence Transformer over every record and searching the
entire corpus would add substantial GPU time, storage, indexing complexity, and failure risk.
Our implementation does something much narrower:

1. exact rules and character TF-IDF find plausible records;
2. only cross-script pairs already found by those classical methods are considered for E5;
3. at most `semantic_pairs_per_query` pairs, currently 3, are embedded for each query;
4. the E5 cosine score helps rank those candidates and becomes one feature for XGBoost;
5. temporary embeddings are deleted after the country is processed.

This is computationally defensible under an eight-hour limit. It is not automatically the
best choice. The present E5 lane **cannot discover a match that the exact and TF-IDF lanes
completely missed**, because it embeds only existing candidates. It can help distinguish or
rank cross-script candidates that were already surfaced by address, number, or partial lexical
evidence.

The correct engineering decision is therefore empirical:

| Version | Components | Purpose |
| --- | --- | --- |
| Baseline A | normalization + exact blocks + character TF-IDF + XGBoost | Establish speed, candidate recall, and F0.5 without neural overhead |
| Hybrid B | Baseline A + bounded cross-script E5 scores | Test whether E5 improves cross-script recall after the final cap or improves F0.5 |
| Rejected default | embed every row + global dense search | Too much additional compute and operational complexity before evidence that it is needed |

Keep E5 only if Hybrid B produces a meaningful improvement in untouched holdout macro F0.5
or a clearly important cross-script slice at acceptable runtime. If it does not, disable it.
There is no technical virtue in retaining a Transformer that does not improve the measured
objective.

The multilingual E5 family was built as a multilingual text-embedding model, and its authors
provide small, base, and large variants to trade quality against inference cost. The system
uses the small variant for that reason. Sentence Transformers also recommends multi-stage
retrieval and reranking for complex semantic-search systems rather than applying an expensive
model indiscriminately to every possible pair.

References:

- [Multilingual E5 technical report](https://arxiv.org/abs/2402.05672)
- [Sentence Transformers semantic-search documentation](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Sentence Transformers retrieve-and-rerank documentation](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

## 2. The problem as a story

Imagine that Amazon receives three enormous phone books.

- Source 1 is the reference phone book. Every row is believed to represent one distinct real
  business.
- Sources 2 and 3 come from other systems. Their rows may describe the same businesses, but
  there is no shared identifier.
- For every Source 1 business, we must return every corresponding record in Sources 2 and 3.
  A Source 1 business may have no match, one match, or several matches.

If a Source 1 row says `Sri Balaji Medicals, 14 MG Road` and another source says
`శ్రీ బాలాజీ మెడికల్స్, M.G. Rd No 14`, a human can combine the name, number, address, and
context. A computer sees different character sequences. Other rows differ only by punctuation,
legal suffixes, word order, missing address pieces, abbreviations, typing errors, accents, or
writing script.

This is **entity resolution**: deciding which noisy records refer to the same real entity.
It is not ordinary multiclass classification. There is no fixed label such as `restaurant` or
`pharmacy`; the output is a set of record-to-record links.

## 3. Why brute force is impossible

Suppose Source 1 has `N` rows and Sources 2 and 3 together have `M` rows. Comparing every
Source 1 row with every possible target row requires `N × M` pair comparisons. With millions
of rows, that becomes trillions of pairs.

No choice of XGBoost, neural network, GPU, or database makes that a sensible first operation.
The system must first answer a cheaper question:

> Which small set of target records is plausible enough to deserve expensive comparison?

This is the purpose of **candidate generation**, also called **blocking** or **retrieval**.
The quality of candidate generation creates a hard upper bound on the complete system. If a
true match never enters the candidate set, the classifier cannot recover it later.

## 4. The complete system in one picture

```text
Seven original TSV files
        │
        ▼
Schema and ID validation while streaming chunks
        │
        ▼
Unicode-preserving comparison views
        │
        ▼
Country-partitioned compressed Parquet
        │
        ├── exact name/address/number blocks
        ├── character n-gram TF-IDF on names
        ├── character n-gram TF-IDF on addresses
        └── optional bounded E5 scoring for cross-script candidates
                         │
                         ▼
             at most 30 candidates per S1 query
                         │
                         ▼
             stable pair-comparison features
                         │
                         ▼
     S1-grouped train / validation / holdout split
                         │
                         ▼
       bounded XGBoost tuning + threshold selection
                         │
                         ▼
       test scoring and zero/one/many predicted links
                         │
                         ▼
matching_results.tsv + candidate_pairs.tsv + validator
```

The system is a cascade. Every stage reduces or enriches the data before the next stage.

## 5. What happens to multilingual input

### 5.1 We do not translate everything to English

Business names and addresses are identifiers as much as they are language. Translation can
change or omit the exact clues we need:

- a proper name may be transliterated inconsistently;
- `Road`, `Rd`, and a local-language equivalent may be rewritten differently;
- house numbers, postal codes, and branch names may move or disappear;
- rare local names may be mistranslated;
- an external translation service adds time, cost, network dependence, and reproducibility
  risk.

The pipeline therefore keeps the original Unicode text and creates additional comparison
views. The original value remains available.

### 5.2 What Unicode normalization does

Unicode gives characters numeric code points, but visually equivalent text can sometimes be
encoded in different ways. Normalization converts defined equivalent or compatibility forms
into a consistent representation.

The pipeline uses `NFKC` followed by Unicode `casefold()`:

- `NFKC` makes compatibility variants such as full-width Latin letters comparable;
- `casefold()` is a Unicode-aware form of case normalization and is broader than ASCII
  lowercasing;
- whitespace is collapsed;
- punctuation and symbol-stripped views are created separately;
- Latin accent folding is retained as an extra view rather than replacing the original;
- number, probable postal-code, legal-suffix-free name, and script signatures are extracted.

Unicode normalization does **not** translate Hindi to English, Tamil to Latin, or French to
English. It standardizes representation. Embeddings, when enabled, solve a different problem:
they provide a numerical similarity signal for text that is not lexically identical.

### 5.3 Views produced for each source record

For a record containing business name, address, country, and entity ID, preprocessing creates:

- normalized business name;
- punctuation-normalized name;
- Latin accent-folded name;
- legal-suffix-stripped core name;
- normalized and punctuation-normalized address;
- address-number signature;
- probable postal token;
- name and address script signatures;
- a flag showing whether the name contains non-Latin script.

These are comparison views, not new ground-truth facts.

## 6. Why TSV is not converted to CSV

The supplied data is already valid tab-separated data. Converting it to CSV would duplicate
several gigabytes, create extra quoting problems for comma-heavy addresses and ID lists, and
provide no modeling benefit.

The system treats the seven TSV files as immutable input. It reads them in bounded chunks and
writes temporary Parquet partitions because later stages repeatedly need selected columns and
country subsets. Parquet is compressed, typed, and column-oriented, which makes repeated local
processing substantially more practical.

The flow is:

```text
TSV source of truth → streamed validation/normalization → temporary Parquet working data
```

The required final outputs are written back to TSV.

## 7. Candidate generation: where most scalability is won

### 7.1 Country partition

Records are processed within country. This prevents obvious cross-country comparisons and
keeps individual retrieval problems smaller. Country remains an open string rather than a
fixed training-only category, because test data may contain countries absent from training.

### 7.2 Exact blocks

Exact blocks join records sharing strong normalized evidence:

- normalized name;
- normalized address;
- legal-suffix-free name;
- address-number signature.

Very common values are excluded from exact blocking using `max_exact_block_size`; otherwise a
generic value could generate a huge block and destroy efficiency.

Exact matches are fast and precise but cannot handle substantial spelling or script changes.

### 7.3 Character n-gram TF-IDF

A character n-gram breaks text into overlapping character fragments. For example, a simplified
3-gram view of `amazon` contains fragments such as `ama`, `maz`, `azo`, and `zon`. Misspellings
and formatting changes can preserve many fragments even when the whole strings differ.

TF-IDF increases the influence of informative fragments and reduces the influence of fragments
that occur everywhere. The implementation uses character-within-word n-grams of lengths 3 to 5,
sparse matrices, similarity thresholds, and top-K retrieval. The name and address lanes are
separate because they contain different kinds of noise.

This is a strong entity-resolution baseline because it is:

- sparse rather than dense;
- much cheaper than a Transformer pass;
- naturally tolerant of small spelling changes;
- language-agnostic at the character level;
- easy to inspect and debug.

It is weaker when two matching values use completely different scripts or have almost no
shared characters. The implementation uses scikit-learn's sparse TF-IDF representation and
top-N sparse multiplication. See the official
[TfidfVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

### 7.4 The precise role of multilingual E5

E5 converts a piece of text into a fixed-length dense vector. Similar vector directions imply
similarity under the model's learned representation.

In this codebase, E5 is intentionally restricted:

- it runs only when embeddings are enabled and a GPU is visible;
- it considers cross-script pairs only by default;
- it selects only the strongest few classical candidates per query;
- it encodes query and candidate subsets in parallel across visible GPUs;
- it calculates cosine scores only for those pairs;
- it deletes transient country-level embeddings afterward.

This avoids global dense indexing. It also creates the limitation stated at the beginning:
E5 currently **reranks and enriches** existing candidates; it is not an independent dense
retriever.

### 7.5 Candidate union and cap

All retrieval lanes are unioned. Duplicate pairs are merged and retain their strongest lane
signals. A retrieval score combines name TF-IDF, address TF-IDF, E5 cosine, and exact-match
bonuses. The result is capped at 30 candidates per Source 1 query in the Kaggle configuration.

The cap is crucial: every additional candidate becomes a feature row, a training example, and
an inference decision. Increasing the cap can improve recall but increases memory, runtime,
and false-positive opportunity.

## 8. The first success gate: candidate recall

Before tuning any classifier, measure:

- **pair recall:** fraction of all true links present in the candidate set;
- **all-matches entity recall:** fraction of matched Source 1 entities for which every true
  target record is present;
- mean candidates per query;
- candidate recall by country and, ideally, by same-script versus cross-script group.

Interpretation:

| Observation | Meaning | Next action |
| --- | --- | --- |
| Low candidate recall | Retrieval removed true links | Improve blocks, thresholds, top-K, or add a genuine dense-retrieval lane |
| High candidate recall, weak F0.5 | Classifier or threshold is making poor decisions | Improve features, labels, hard negatives, tuning, or calibration |
| Hybrid and baseline have equal recall/F0.5 | E5 is not earning its cost | Disable E5 |
| Hybrid improves cross-script F0.5 enough | E5 adds useful evidence | Retain it with a measured runtime cap |

## 9. Pair features: evidence shown to the classifier

For every candidate pair, the system builds 28 stable numerical features. Major groups are:

- retrieval-lane flags and within-query retrieval rank;
- E5 cosine score when available;
- exact normalized name, core name, address, and number matches;
- RapidFuzz name and address ratios;
- token-set ratios and token Jaccard similarities;
- address-number Jaccard and postal equality;
- country and writing-script agreement;
- cross-script indicator;
- missing-address flags;
- name and address length ratios;
- target-source indicator.

The final classifier does not use raw text or raw TF-IDF vector dimensions. That is deliberate:
raw TF-IDF dimensions and IDF weights depend on the fitted corpus and can shift between train
and test. Stable comparison features are easier to audit and more robust.

## 10. Why XGBoost is the final model

After candidate generation, the problem is ordinary binary classification over structured
pair features: match or non-match.

XGBoost is suitable because:

- it models nonlinear interactions such as “similar name plus matching postal code”;
- it handles mixed continuous and binary features;
- it trains efficiently with histogram trees and CUDA;
- it provides early stopping;
- it works well without converting pair features into an end-to-end deep architecture;
- its predictions are inexpensive during test scoring.

The original XGBoost paper describes the scalable tree-boosting system and its sparsity-aware
and approximate tree-learning design: [Chen and Guestrin, 2016](https://doi.org/10.1145/2939672.2939785).

XGBoost uses one GPU in this project. The two Kaggle GPUs are used for data-parallel E5
encoding. Forcing a single tabular XGBoost model across two T4s would introduce distributed
coordination overhead and is not automatically faster.

## 11. Training labels and hard negatives

Ground truth lists the correct Source 2 and Source 3 IDs for every Source 1 entity. Candidate
pairs found in that list receive label 1; other retrieved pairs receive label 0.

The negative class can be enormous. Keeping every trivial negative would waste memory and let
easy cases dominate training. The system retains all positives and only the strongest configured
number of negative candidates per Source 1 entity. These are **hard negatives**: plausible
records that the model must learn to reject.

## 12. Splitting without leakage

All candidate pairs for one Source 1 entity must stay in the same split. If pairs from one
entity appeared in both train and validation, the model would be evaluated on variants of an
entity it already saw.

The deterministic split is:

- training partition for fitting;
- validation partition for hyperparameter and threshold selection;
- untouched holdout partition for the final local estimate.

The split unit is the Source 1 entity, not an individual pair row.

## 13. Why the metric is macro F0.5 rather than accuracy

Candidate data contains many more non-matches than matches. A model that predicts “non-match”
almost everywhere can obtain misleadingly high row accuracy while failing the actual task.

For each Source 1 entity, predicted and expected target-ID sets produce precision and recall.
With `β = 0.5`:

```text
F0.5 = (1 + 0.5²) × precision × recall / (0.5² × precision + recall)
```

Because beta is below 1, precision receives more weight than recall. False links are therefore
especially costly.

The implementation calculates F0.5 separately for every Source 1 entity and averages the
scores. When both expected and predicted sets are empty, the singleton receives 1.0. When the
expected set is empty but the model invents a link, it receives 0.0.

The probability threshold is scanned from the configured minimum to maximum, and the threshold
with the best validation macro F0.5 is selected. If scores tie, the higher threshold wins,
which is consistent with the precision-heavy objective.

## 14. Hyperparameter tuning under eight hours

An exhaustive grid or nested cross-validation would repeatedly train large models on very
similar configurations. It provides stronger statistical estimation in some settings, but it
is poorly matched to one eight-hour production run over millions of candidate rows.

The implemented tuner uses:

- a reproducible sequence of randomized parameter candidates;
- at most 12 trials in the Kaggle configuration;
- a 40-minute search timeout;
- early stopping within each trial;
- a bounded tuning sample selected by complete Source 1 groups;
- direct validation macro F0.5 for trial comparison;
- one final refit on the complete training partition.

The tuned parameters include tree depth, learning rate, minimum child weight, row and column
subsampling, and L1/L2 regularization. The notebook prints every completed trial, selected
threshold, F0.5, best iteration, and elapsed time.

## 15. Stage-by-stage execution contract

### Stage 1: Input validation and preprocessing

- **Input:** seven TSV files.
- **Work:** validate file presence, columns, ID prefixes, missing IDs, and duplicate IDs while
  streaming; create Unicode comparison views; partition by country.
- **Output:** compressed Parquet partitions and preprocessing manifest.
- **Acceptance check:** all seven schemas pass and manifests contain plausible row totals.

### Stage 2: Training candidates

- **Input:** normalized training partitions and ground truth.
- **Work:** exact and TF-IDF retrieval, optional bounded E5 scoring, lane union, query cap.
- **Output:** candidate pairs per country plus recall diagnostics.
- **Acceptance check:** candidate recall is high enough to justify classifier training.

### Stage 3: Test candidates

- **Input:** normalized test partitions.
- **Work:** apply the same retrieval logic without ground-truth labels.
- **Output:** capped test candidate pairs.
- **Acceptance check:** candidate counts and per-query means remain within memory/runtime bounds.

### Stages 4 and 5: Pair features

- **Input:** training/test candidates plus normalized records.
- **Work:** construct stable pair features in query-aligned chunks; attach train labels and
  entity-level splits; retain hard negatives.
- **Output:** Parquet feature partitions.
- **Acceptance check:** training and validation each contain both classes; all positives found
  during candidate generation remain present.

### Stage 6: Tune and train XGBoost

- **Input:** labeled training features and ground-truth sets.
- **Work:** bounded trial search, early stopping, final refit, threshold optimization, holdout
  scoring.
- **Output:** XGBoost model, all trials, training history, threshold curve, validation and
  holdout macro F0.5.
- **Acceptance check:** holdout metric is reported separately from the selected validation score.

### Stage 7: Inference, output, and validation

- **Input:** trained model and test features.
- **Work:** score candidate pairs, apply the threshold, retain empty singleton predictions,
  combine country parts, and invoke the official validator.
- **Output:** `matching_results.tsv`, `candidate_pairs.tsv`, manifests, and submission archive.
- **Acceptance check:** required columns, IDs, relationships, and Source 1 coverage pass the
  validator.

## 16. Runtime and storage strategy

The Kaggle configuration has a 480-minute budget and reserves 35 minutes for required final
work. Optional E5 processing has its own time cap. The runtime guard can stop optional work
before it consumes the output reserve.

Storage is separated by purpose:

- original TSV files remain under the immutable input mount;
- large Parquet and embedding intermediates live under `/kaggle/temp`;
- final output, model metadata, configuration, and evidence live under `/kaggle/working`;
- Git stores source code, never challenge data;
- AWS S3 stores immutable inputs and durable run artifacts;
- AWS EBS stores disposable high-throughput intermediate files.

## 17. Kaggle and AWS in parallel

Kaggle and AWS do not require separate ML implementations. Git is the source of truth, and
both environments invoke the same Python package with different YAML configurations.

### Kaggle path

1. Push the tested branch to GitHub.
2. Attach the challenge dataset.
3. Select T4 x2.
4. Run `Experiment_Notebooks/03_Kaggle_End_to_End.ipynb`.
5. The notebook deletes its temporary code checkout and fetches the latest configured branch.
6. It discovers all seven TSV files, prints the configuration, streams progress, reports F0.5,
   validates outputs, and packages durable artifacts.

### AWS path

1. Provision the private S3 bucket and EC2 instance role using CloudFormation.
2. Upload the original TSV files to S3.
3. Launch an EC2 G5 instance with adequate EBS storage.
4. Fetch an exact tested Git commit for the final auditable run.
5. Run the same package with `configs/aws_g5.yaml`.
6. Upload output, logs, model, configuration, and metrics to a unique S3 run prefix.
7. Stop or terminate the GPU instance after artifacts are safely in S3.

Kaggle is appropriate for the first measured baseline because its GPU session is already
available and bounded. AWS is useful for controlled reruns, larger GPU configurations, durable
artifact storage, and team infrastructure. Hardware can change runtime, but the commit,
configuration, transformations, and metric definition must remain traceable.

## 18. The experiment sequence that should decide the final model

### Experiment 0: pipeline smoke test

- Use synthetic or intentionally small coherent data.
- Validate schemas, stage contracts, model save/load, output format, and official validator.
- Do not report this score as challenge accuracy.

### Experiment 1: full TF-IDF baseline

- Disable embeddings.
- Run exact blocks, character TF-IDF, stable features, and XGBoost.
- Record candidate recall, validation F0.5, holdout F0.5, stage runtimes, and peak storage.

### Experiment 2: bounded E5 ablation

- Use the same split, seed, candidate settings, and XGBoost search budget.
- Enable only the bounded cross-script E5 lane.
- Compare total and cross-script metrics with Experiment 1.

### Decision rule

Retain E5 only if:

1. holdout macro F0.5 improves beyond ordinary run-to-run noise;
2. the improvement is traceable to useful cross-script cases rather than leakage or a lucky
   threshold;
3. total runtime remains inside the production budget with the final-output reserve intact;
4. output validation and reproducibility checks still pass.

If candidate recall remains weak specifically for cross-script matches, the next research step
is not “increase the reranker.” The next step is to design a genuine bounded dense-retrieval
lane that can introduce candidates absent from TF-IDF, then measure its recall and cost.

## 19. Failure modes to watch

### Low recall from blocking

The candidate set is the bottleneck. Inspect missed true pairs by country, script relationship,
missing address, and number agreement. Increase top-K or add a targeted lane only for the slice
that needs it.

### False merges from generic names

Businesses with names such as `General Store` can collide. Require stronger address/number
evidence, reduce oversized blocks, and inspect the selected threshold.

### Train/test country shift

France appears in test but not training. Avoid fixed country one-hot categories and inspect
whether generic string features behave consistently. A leaderboard score is still required to
measure the real shift.

### Excess memory or disk usage

Candidate caps, hard-negative limits, feature chunks, Parquet compression, and deletion of
temporary embeddings are the main controls. A 200 GB warning should be treated seriously;
monitor `/kaggle/temp` and `/kaggle/working` separately.

### Misleading validation

Never split candidate rows randomly. Keep all pairs for one Source 1 entity together. Never
select parameters on the holdout set. Keep synthetic scores separate from real-data metrics.

### Overclaiming 0.99

No architecture guarantees 0.99. The only defensible claims are measured candidate recall,
validation macro F0.5, untouched holdout macro F0.5, runtime, and leaderboard score produced by
a recorded commit and configuration.

## 20. Important repository files

| File | Responsibility |
| --- | --- |
| `Experiment_Notebooks/03_Kaggle_End_to_End.ipynb` | Visible Kaggle launcher, input inventory, progress, results, and packaging |
| `src/ber/normalization.py` | Unicode and comparison-view construction |
| `src/ber/io.py` | TSV validation, chunk streaming, and Parquet partitioning |
| `src/ber/candidates.py` | Exact and character TF-IDF candidate retrieval |
| `src/ber/embeddings.py` | Multi-GPU embedding orchestration |
| `src/ber/embedding_worker.py` | Per-GPU E5 encoding worker |
| `src/ber/features.py` | Stable pair-feature construction |
| `src/ber/splitting.py` | Deterministic Source 1 entity split |
| `src/ber/model.py` | Bounded XGBoost tuning, refit, and persistence |
| `src/ber/metrics.py` | Singleton-aware macro F0.5 and threshold scan |
| `src/ber/submission.py` | Output construction and relationship checks |
| `src/ber/pipeline.py` | Seven-stage orchestration and runtime budget |
| `configs/kaggle_t4x2.yaml` | Eight-hour Kaggle configuration |
| `configs/aws_g5.yaml` | AWS G5 configuration |
| `infra/aws/` | S3, IAM, EC2, and run automation |

Paths under `src`, `configs`, and `infra` are relative to
`code/business_entity_resolution/`.

## 21. Reproducible commands

Install the package:

```powershell
cd D:\Bunker\BaseCamp\AmazonML\code\business_entity_resolution
python -m pip install -e .
```

Run the tests:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

Inspect a local configuration without training:

```powershell
python -m ber.cli inspect `
  --config configs\smoke.yaml `
  --repo-root ..\.. `
  --data-root ..\..\dataset `
  --work-dir artifacts\work `
  --output-dir artifacts\output
```

The full Kaggle run is launched from:

```text
Experiment_Notebooks/03_Kaggle_End_to_End.ipynb
```

## 22. Current evidence status

The codebase and notebook implement the complete flow, input validation, progress reporting,
metric calculation, output validation, and artifact recording. Local inspection confirms the
seven TSV files and notebook execution path.

The following are still **unmeasured until the Kaggle or AWS production run completes**:

- full-data candidate recall;
- TF-IDF-only validation and holdout macro F0.5;
- hybrid E5 validation and holdout macro F0.5;
- cross-script ablation benefit;
- full stage timings, peak RAM, GPU memory, and disk use;
- leaderboard score.

This separation is intentional. Architecture is a hypothesis. The recorded experiments decide
whether the hypothesis is good.

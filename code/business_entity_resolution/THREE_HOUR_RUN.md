# Three-hour Kaggle run

Use `Experiment_Notebooks/04_Kaggle_Three_Hour_Run.ipynb` with the challenge dataset attached, Internet enabled, and Kaggle T4 x2 selected. Run all cells from the top. The first setup cell deletes the previous temporary Git checkout, fetches the latest `main` commit, and prints the exact commit hash. It uses `configs/kaggle_t4x2.yaml`; change the config in Git and push before starting a new notebook run.

The notebook logs each preprocessing file and chunk, country and source retrieval stage, feature partition, XGBoost trial, threshold, validation and holdout macro F0.5, and official TSV validation. Its log is `/kaggle/working/ber_pipeline.log`; its two required results are `/kaggle/working/submission/output/matching_results.tsv` and `candidate_pairs.tsv`. After the run, download those files and the ZIP from Kaggle Output. The notebook can be rerun from scratch against a newer Git commit.

## Why this configuration

The original character n-gram TF-IDF search stalled on the first full India source. The deadline configuration uses exact name, address, core-name, and address-number blocks plus sparse hashed TF-IDF on Unicode name tokens, address words, and address numbers. It prunes very common hash buckets and searches queries in bounded chunks. No translation or embedding download is required. XGBoost trains on a random 50,000 Source-1 queries per training country against the complete Source-2 and Source-3 pools; all test Source-1 queries are processed. The model uses one GPU. The second T4 does not accelerate a single booster in this configuration.

Local checks on 2026-09-27, before a Kaggle production run:

- Full India Source-2 pool: 2,000,000 targets and 5,000 sampled Source-1 queries. Candidate generation after record preparation took 63.8 seconds and recovered 7,573 of 8,304 known links (91.20%) within the configured 24 candidates per query. This is **candidate recall**, not model F0.5 or leaderboard accuracy.
- The optimized feature builder processed about 100,000 real candidate pairs in 4.2 seconds locally, compared with 19.8 seconds before that change.
- The synthetic end-to-end integration test and candidate tests run with `python -m pytest -q code/business_entity_resolution/tests` from the repository root.

The three-hour `time_budget_minutes` begins when the pipeline starts, after Git setup and package installation. The pipeline reserves 20 minutes for output, but the reserve is a guard, not a promise that Kaggle will finish on time. Full test feature generation, model tuning, and the final portal score remain unmeasured until the production run. A 0.99 score cannot be promised: the measured retrieval recall already caps how many true links the classifier can recover.

## Run and inspect

In Kaggle, attach the dataset at `/kaggle/input/datasets/satyapaladugu/amazon-ml-challenge/`, select **T4 x2**, enable **Internet**, upload the new notebook, and use **Save Version > Save & Run All**. Watch Stage 1 through Stage 7 in the output. After it finishes, check `completed_run.json`, the validation and untouched holdout F0.5 printed in the notebook, both TSV files, and the official validator result. Submit `matching_results.tsv` to the portal; retain `candidate_pairs.tsv` and the ZIP for the full package.

The Kaggle CLI can upload a kernel and follow logs only after local CLI authentication. A stale API key returns HTTP 401; notebook logs cannot be fetched remotely in that state. Keep the notebook output open or paste its latest log when debugging until CLI access is restored.

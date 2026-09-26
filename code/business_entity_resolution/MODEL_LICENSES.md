# Model and Library License Gate

The challenge rule allows model architectures and pretrained weights released under
MIT or Apache 2.0 and limits models to 8 billion parameters.

| Component | Role | License | Size / parameter status | Allowed |
| --- | --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` at revision `fd1525a9fd15316a2d503bf26ab031a61d056e98` | Optional cross-script pair reranking | MIT | About 118M parameters, below 8B | Yes |
| XGBoost 2.1.4 | Final supervised pair classifier | Apache 2.0 | Gradient boosted trees, not a pretrained parameter model | Yes |
| scikit-learn 1.6.1 | Character TF-IDF candidate retrieval | BSD-3-Clause library; no pretrained weights | Fitted only on challenge data | Yes |

No external business lookup, geocoder, registry, search API, or external labeled data is
used. The E5 weights are general pretrained language weights. Pin the exact revision in
Kaggle/AWS setup so the audited artifact cannot silently change.

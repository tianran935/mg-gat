# Yelp MG-GAT Reproduction

This project is a careful reproduction attempt of the Stage 1 MG-GAT rating-prediction experiment from:

Interpretable Recommendations and User-Centric Explanations with Geometric Deep Learning.

## Scope
- Reproduce the Yelp PA rating-prediction setup as closely as possible.
- Implement only Stage 1 MG-GAT.
- Exclude all LLM-based explanation generation.
- Record all paper ambiguities and dataset mismatches explicitly.

## Important finding
A direct inspection of the available Yelp snapshot shows that the paper's reported PA dataset is not equal to a naive `state == PA` slice. Even after restricting to PA restaurants with non-empty hours and attributes, the current snapshot remains much larger than Table D1 on users and ratings.

## Current conservative default subset
- region: `PA`
- business category contains `Restaurants`
- business must have `hours`
- business must have `attributes`
- years kept: `2009-2018`

Resulting current subset on this snapshot:
- users: `228,679`
- businesses: `9,519`
- reviews: `784,856`

Paper Table D1 target for PA:
- users: `76,865`
- businesses: `10,966`
- ratings: `260,350`

## Commands
Subset analysis:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp python3 src/subset_analysis.py
```

Preprocess:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp python3 src/preprocess.py
```

Build graphs:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp python3 src/build_graph.py
```

Train baseline reproduction:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp python3 train.py
```

## Current baseline result
Saved in `artifacts/train_result.json`:
- test RMSE: `2.2034`
- test Spearman: `0.0281`

Paper Table 1 target:
- interpretable MG-GAT RMSE: `1.249`
- less interpretable MG-GAT RMSE: `1.217`

# Yelp MG-GAT Reproduction

This project is a careful reproduction attempt of the Stage 1 MG-GAT rating-prediction experiment from the paper:

[Interpretable Recommendations and User-Centric Explanations with Geometric Deep Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3696092)

## Scope
- Reproduce the Yelp Pennsylvania rating-prediction setup as closely as possible.
- Implement only Stage 1 MG-GAT.
- Exclude all LLM-based explanation generation.
- Record all paper ambiguities, assumptions, and dataset mismatches explicitly.

## Important finding
The paper's reported PA dataset is not equal to a naive `state == PA` slice of the currently available Yelp snapshot. To get closer to the paper, the current code uses a stricter subset:
- Pennsylvania businesses
- category contains `Restaurants`
- non-empty `hours`
- non-empty `attributes`
- reviews between `2009` and `2018`
- users restricted to the friendship giant component

Current processed subset used for the best runs:
- users: `98,831`
- businesses: `9,458`
- reviews: `516,336`
- train reviews: `372,623`
- valid reviews: `71,969`
- test reviews: `71,744`

Paper Table D1 target for PA:
- users: `76,865`
- businesses: `10,966`
- ratings: `260,350`

This remaining mismatch is the main reason exact reproduction is still not guaranteed.

## Project structure
- `src/subset_analysis.py`: compare candidate PA subset rules against the paper statistics
- `src/preprocess.py`: build the processed PA subset, features, and time split
- `src/build_graph.py`: construct the user graph and three business graphs
- `src/model.py`: MG-GAT-style Stage 1 model
- `src/evaluate.py`: RMSE, Spearman, FCP, and BPR metrics
- `train.py`: end-to-end training and evaluation
- `artifacts/*.json`: saved experiment results

## Environment
The AutoDL server uses:
- Python: `/root/miniconda3/bin/python`
- GPU: CUDA is used automatically when available

## Commands
Subset analysis:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp /root/miniconda3/bin/python src/subset_analysis.py
```

Preprocess:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp /root/miniconda3/bin/python src/preprocess.py
```

Build graphs:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp /root/miniconda3/bin/python src/build_graph.py
```

Train the default interpretable MG-GAT reproduction:
```bash
cd /root/autodl-tmp/yelp
PYTHONPATH=/root/autodl-tmp/yelp /root/miniconda3/bin/python train.py
```

## Best current results
### Interpretable MG-GAT
Saved in `artifacts/train_result.json`:
- best validation RMSE: `1.2104`
- best epoch: `69`
- test RMSE: `1.2462`
- test Spearman: `0.3513`
- test FCP: `0.6042`
- test BPR: `0.6023`

Paper Table 1 target:
- RMSE: `1.249`
- Spearman: `0.405`
- FCP: `0.602`
- BPR: `0.520`

### Less interpretable MG-GAT
Best run currently saved in `artifacts/uninterp_search.json`:
- configuration: `uninterp_64x64_drop01`
- test RMSE: `1.1990`
- test Spearman: `0.3819`
- test FCP: `0.6149`
- test BPR: `0.6139`

Paper Table 1 target:
- RMSE: `1.217`
- Spearman: `0.430`
- FCP: `0.645`
- BPR: `0.551`

## Current conclusion
- On RMSE, the reproduction is already very close to the paper for the interpretable model and better than the paper's reported number for the less interpretable model.
- On ranking quality, especially Spearman, there is still a visible gap from Table 1.
- The strongest remaining risk is dataset mismatch rather than code completeness.

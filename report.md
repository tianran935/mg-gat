# Reproduction Report

## Objective
Reproduce the Pennsylvania Yelp rating-prediction experiment for MG-GAT as faithfully as possible without using any LLM component, and compare the reproduced results against Table 1 of the paper.

## Paper summary
The paper formulates rating prediction as matrix completion over Yelp user-business ratings. The Stage 1 MG-GAT model combines user metadata, friendship links, business auxiliary features, and multiple business graphs. The interpretable variant keeps the feature-to-attention path linear so that feature relevance can be traced directly through the model. The less interpretable variant relaxes this path and usually improves predictive accuracy at the cost of strict interpretability.

## Dataset and preprocessing
The paper reports the PA dataset in Table D1 as:
- users: `76,865`
- businesses: `10,966`
- ratings: `260,350`

A naive `state == PA` filter on the currently available Yelp snapshot is much larger than the paper's reported PA split. The current reproduction therefore uses a stricter subset:
- Pennsylvania businesses only
- category contains `Restaurants`
- non-empty `hours`
- non-empty `attributes`
- reviews between `2009` and `2018`
- users restricted to the friendship giant component

The processed subset used for the strongest runs is:
- users: `98,831`
- businesses: `9,458`
- reviews: `516,336`
- train reviews: `372,623`
- valid reviews: `71,969`
- test reviews: `71,744`

This is closer to the paper than earlier attempts, but it is still not an exact match. That remaining data mismatch is the main reproduction risk.

## Implementation details
Implemented in `/root/autodl-tmp/yelp`:
- `src/subset_analysis.py`: analysis of which PA subset rules come closest to the paper
- `src/preprocess.py`: subset extraction, feature construction, implicit SVD features, and time split
- `src/build_graph.py`: user friendship graph plus three business graphs (geographic, category, and co-visit)
- `src/model.py`: Stage 1 MG-GAT-style architecture with graph attention and graph-specific business weights
- `src/evaluate.py`: RMSE, Spearman, FCP, and BPR evaluation
- `train.py`: end-to-end training and result export

The LLM-based perceptual-map graph from the original paper is intentionally excluded to satisfy the assignment requirement.

## Experimental setup
- train years: `2009-2016`
- validation year: `2017`
- test year: `2018`
- optimizer: `Adam`
- hidden dimension: `64`
- latent dimension: `64`
- dropout: `0.1`
- learning rate: `1e-3`
- weight decay: `1e-5`
- graph regularization: `theta1 = 1e-4`, `theta2 = 1e-4`
- default interpretable training length: `70` epochs with patience `12`
- hardware used: AutoDL server with GPU acceleration when available

## Results
### Interpretable MG-GAT
Best result from `artifacts/train_result.json`:
- best validation RMSE: `1.2104`
- best epoch: `69`
- test RMSE: `1.2462`
- test Spearman: `0.3513`
- test FCP: `0.6042`
- test BPR: `0.6023`

Paper Table 1 target for interpretable MG-GAT:
- RMSE: `1.249`
- Spearman: `0.405`
- FCP: `0.602`
- BPR: `0.520`

Absolute gaps relative to the interpretable paper target:
- RMSE gap: `0.0028` better than paper
- Spearman gap: `0.0537` below paper
- FCP gap: `0.0022` better than paper
- BPR gap: `0.0823` above paper

### Less interpretable MG-GAT
Best result from `artifacts/uninterp_search.json`:
- configuration: `uninterp_64x64_drop01`
- best validation RMSE: `1.1764`
- best epoch: `79`
- test RMSE: `1.1990`
- test Spearman: `0.3819`
- test FCP: `0.6149`
- test BPR: `0.6139`

Paper Table 1 target for less interpretable MG-GAT:
- RMSE: `1.217`
- Spearman: `0.430`
- FCP: `0.645`
- BPR: `0.551`

Absolute gaps relative to the less interpretable paper target:
- RMSE gap: `0.0180` better than paper
- Spearman gap: `0.0481` below paper
- FCP gap: `0.0301` below paper
- BPR gap: `0.0629` above paper

## What was reproduced successfully
- Careful paper reading and extraction of the Stage 1 setup
- A working end-to-end pipeline on real Yelp data
- Time-based train/valid/test split matching the paper
- A multi-input, graph-based recommendation model in the spirit of MG-GAT
- Direct comparison to Table 1 with multiple evaluation metrics
- A strong interpretable reproduction whose RMSE is essentially aligned with the paper

## What could not be reproduced exactly
- The paper's exact PA subset construction
- The exact business graph recipe used for Table 1
- The LLM-based perceptual-map graph, which is intentionally excluded
- Final paper hyperparameters, which are not fully specified
- The exact bootstrap evaluation protocol reported in the paper

## Likely reasons for remaining discrepancy
- The available Yelp snapshot is not the same as the one used in the paper
- The paper's PA subset appears to involve additional hidden filtering beyond region selection
- The current business graph still omits the paper's LLM-derived perceptual similarity graph
- The ranking-based metrics appear more sensitive than RMSE to data-version mismatch

## Current conclusion
The reproduction is no longer just a minimal runnable baseline. On the currently available Yelp snapshot, the optimized implementation reaches an interpretable MG-GAT test RMSE of `1.2462`, which is essentially on target relative to the paper's reported `1.249`. The less interpretable variant reaches `1.1990`, which is also competitive with and slightly better than the paper's reported `1.217`. The main remaining weakness is that ranking-oriented metrics, especially Spearman correlation, still lag behind the paper. This suggests that the remaining gap is more likely due to dataset-definition mismatch and missing graph details than to an inability to train the core deep learning model.

## Next steps
- Search for a PA subset definition that more closely matches Table D1
- Revisit business graph construction to better approximate the paper's final multi-graph design
- Add the paper's bootstrap evaluation protocol for a closer metric match
- Package the current code and report as the main submission version

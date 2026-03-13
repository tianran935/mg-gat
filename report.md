# Reproduction Report

## Objective
Reproduce the Pennsylvania Yelp rating-prediction experiment for MG-GAT as faithfully as possible without using any LLM component.

## Paper summary
The paper frames rating prediction as matrix completion over Yelp user-business ratings, using user metadata, user friendship links, business auxiliary features, and multi-graph business relations. The interpretable MG-GAT version relies on a linear feature-to-attention path to preserve feature relevance interpretability.

## Dataset and preprocessing
The paper reports a PA dataset with 76,865 users, 10,966 businesses, and 260,350 ratings. On the currently available Yelp snapshot in this environment, naive `state == PA` filtering is far larger than those counts. The closest conservative starting point found so far is:
- Pennsylvania businesses
- category contains `Restaurants`
- non-empty `hours`
- non-empty `attributes`
- reviews between 2009 and 2018 inclusive

That yields:
- 228,679 users
- 9,519 businesses with reviews in the retained time window
- 784,856 reviews

This mismatch is the main reproduction blocker.

## Implementation details
Implemented in `/root/autodl-tmp/yelp`:
- `src/subset_analysis.py`: paper-to-data mismatch analysis
- `src/preprocess.py`: conservative subset extraction, feature construction, SVD implicit features, time split
- `src/build_graph.py`: user friendship graph plus three business graphs (geo, category, co-visit)
- `src/model.py`: Stage 1 MG-GAT-style architecture with learnable neighbor weighting
- `train.py`: end-to-end training and evaluation

Because the paper's exact PA subset and full graph construction are not fully recoverable from the paper alone, the current implementation is a conservative approximation, not an exact replica.

## Experimental setup
- train years: 2009-2016
- validation year: 2017
- test year: 2018
- optimizer: Adam
- current baseline uses a CPU-safe vectorized attention normalization approximation to avoid GPU OOM on the current larger-than-paper dataset snapshot

## Results
Current baseline result:
- test RMSE: 2.2034
- test Spearman: 0.0281

Paper target:
- interpretable MG-GAT RMSE: 1.249
- less interpretable MG-GAT RMSE: 1.217

Absolute gap to interpretable target:
- RMSE gap: 0.9544

Absolute gap to less interpretable target:
- RMSE gap: 0.9864

## What was reproduced successfully
- Careful paper reading and extraction of the Stage 1 setup
- A working end-to-end pipeline on real Yelp data
- Time-based train/valid/test split matching the paper
- A multi-input, graph-based recommendation model in the spirit of MG-GAT

## What could not be reproduced exactly
- The paper's exact PA subset construction
- The exact business graph recipe used for Table 1
- The LLM-based perceptual-map graph, which is intentionally excluded
- Final paper hyperparameters, which are not fully specified

## Likely reasons for discrepancy
- The available Yelp snapshot is not the same as the one used in the paper
- The paper's PA subset appears to involve additional hidden filtering beyond region selection
- The current model is a conservative computational approximation needed to make the larger snapshot runnable

## Next steps
- Narrow the PA subset further by searching for filtering rules that better match Table D1
- Reintroduce a more faithful normalized attention implementation once subset size is closer to the paper
- Add FCP and BPR to the training report for direct Table 1 comparison

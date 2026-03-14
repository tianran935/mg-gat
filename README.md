# MG-GAT Yelp PA Reproduction

This project is a careful reproduction of the Stage 1 rating-prediction component from the paper:

*Interpretable Recommendations and User-Centric Explanations with Geometric Deep Learning*

The goal is to reproduce the Yelp Pennsylvania (PA) MG-GAT experiment in Table 1 as faithfully as possible while removing all LLM-related components and keeping only the deep learning recommendation structure.

## Scope

This reproduction only keeps the Stage 1 model.

Included:
- Yelp PA rating prediction
- user auxiliary features and business auxiliary features
- user friendship graph
- multi-graph business structure
- interpretable MG-GAT and less interpretable MG-GAT variants
- RMSE, Spearman, FCP, and BPR evaluation

Excluded:
- any LLM-based perceptual graph construction
- any LLM-based explanation generation
- any OpenAI / GPT / Qwen / Claude API usage
- any prompt-based or instruction-tuned pipeline

## Research Objective

The main question is whether the core Stage 1 MG-GAT architecture can be reproduced on Yelp PA data without any LLM component, and whether the reproduced result can approach the MG-GAT values reported in Table 1 of the paper.

## Paper Target

For PA, Table 1 reports:

| Model | RMSE | Spearman | FCP | BPR |
|---|---:|---:|---:|---:|
| Interpretable MG-GAT | 1.249 | 0.405 | 0.602 | 0.520 |
| Less interpretable MG-GAT | 1.217 | 0.430 | 0.645 | 0.551 |

## Current Reproduction Result

Current best results from the optimized reproduction are:

| Model | RMSE | Spearman | FCP | BPR |
|---|---:|---:|---:|---:|
| Reproduced interpretable MG-GAT | 1.2462 | 0.3513 | 0.6042 | 0.6023 |
| Reproduced less interpretable MG-GAT | 1.1990 | 0.3819 | 0.6149 | 0.6139 |

Interpretation:
- On RMSE, the reproduction is already very close to the paper for the interpretable model and slightly better than the paper result for the less interpretable model.
- On ranking-oriented metrics, especially Spearman, there is still a visible gap from the paper.
- The remaining discrepancy is more likely driven by dataset-definition mismatch and graph-construction details than by an inability to implement the Stage 1 model itself.

## Dataset Notes

The paper reports PA dataset statistics of:
- users: 76,865
- businesses: 10,966
- ratings: 260,350

However, the currently available Yelp snapshot does not match the paper under a naive `state == PA` filter. The working reproduction subset is therefore built more conservatively.

Current processed subset used in the strongest runs:
- region: PA
- business category contains `Restaurants`
- non-empty `hours`
- non-empty `attributes`
- review years restricted to 2009--2018
- users restricted to the friendship giant component

Processed data size:
- users: 98,831
- businesses: 9,458
- ratings: 516,336

This mismatch is the main limitation of the reproduction.

## Method Summary

The implemented pipeline follows the paper's Stage 1 idea:

1. Filter and construct the PA subset.
2. Build user features and business features.
3. Build the user friendship graph.
4. Build three non-LLM business graphs:
   - geographic proximity graph
   - category similarity graph
   - co-visit graph
5. Map features into a shared representation space.
6. Compute graph-based neighbor importance and aggregation.
7. Produce user and business embeddings.
8. Predict ratings and evaluate against the paper.

Two variants are implemented:
- interpretable MG-GAT
- less interpretable MG-GAT

## Key Files

- `src/preprocess.py`: subset construction, feature preparation, implicit SVD features, split generation
- `src/build_graph.py`: user graph and business multi-graph construction
- `src/model.py`: Stage 1 MG-GAT-style model
- `src/evaluate.py`: RMSE, Spearman, FCP, and BPR metrics
- `train.py`: end-to-end training and evaluation
- `artifacts/train_result.json`: best interpretable result
- `artifacts/hparam_search.json`: interpretable and initial comparison runs
- `artifacts/uninterp_search.json`: less interpretable search results

## How To Run

Train the default interpretable configuration:

```bash
cd your_projectpath
python train.py
```

This script:
- rebuilds the processed dataset
- rebuilds the graphs
- trains the default interpretable MG-GAT configuration
- writes results to `artifacts/train_result.json`

## Main Limitation

The largest unresolved issue is not the deep learning implementation itself, but the fact that the paper's exact PA subset construction and full graph recipe are not fully disclosed. In particular:
- the paper's PA statistics cannot be exactly matched from the current Yelp snapshot
- the LLM-based perceptual graph is excluded by design
- the bootstrap evaluation protocol from the paper is not fully specified

## Conclusion

Within the no-LLM constraint, this project successfully reproduces the core Stage 1 MG-GAT logic and reaches a result that is already close to the published MG-GAT performance on PA in terms of RMSE. The remaining gap is concentrated in ranking-sensitive metrics and should be understood primarily as a data-definition and evaluation-protocol issue rather than a failure to reconstruct the recommendation model.

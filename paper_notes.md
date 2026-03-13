# Paper Notes

## Core task
Predict missing Yelp user-business ratings with an interpretable deep learning model.

## Table 1 PA targets
- MG-GAT interpretable: RMSE 1.249, Spearman 0.405, FCP 0.602, BPR 0.520
- MG-GAT less interpretable: RMSE 1.217, Spearman 0.430, FCP 0.645, BPR 0.551

## Confirmed paper details
- Time split: train 2009-2016, validation 2017, test 2018
- Yelp subset: Pennsylvania (PA), but exact construction is narrower than naive state filtering
- Inputs: user metadata, friendship graph, business attributes/categories/hours/location/check-ins, implicit features from SVD of binarized interactions
- Model: linear feature projection, interpretable attention (NIG), feature relevance (FR), neighbor aggregation, nonlinear dense layer, final graph-regularized embedding
- Loss: squared error plus graph regularization
- Optimizer: Adam
- Hyperparameter search: Hyperopt

## Risks
- Paper does not specify the final hyperparameters clearly
- Business graph construction is internally inconsistent
- One graph is LLM-based in the paper and must be excluded here
- PA subset construction is not fully specified and needs conservative inference

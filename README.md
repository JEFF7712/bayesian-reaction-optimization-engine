# Bayesian Reaction Optimization Engine

Uses machine learning (random forest) to predict reaction yields and Bayesian Optimization to intelligently select experimental conditions.

## Architecture
1. Featurization: Converts raw SMILES strings (Ligands, Additives, Bases) into 2048-bit Morgan Fingerprints (ECFP4) using `RDKit`.
2. Modeling: Uses a Random Forest Regressor to map chemical features to Reaction Yield (%).
3. Bayesian Optimization: Implements an Upper Confidence Bound (UCB) acquisition function to balance Exploitation (high yield) vs. Exploration (high uncertainty).

**Data:** Doyle-Buchwald High-Throughput Experimentation Dataset

## Performance
RMSE: 6.65% (Average prediction error)
R² Score: 0.943 (Correlation)
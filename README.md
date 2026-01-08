# Bayesian Reaction Optimization Engine

Uses machine learning (random forest) to predict reaction yields and Bayesian Optimization to intelligently select experimental conditions.

## Architecture
1. Featurization: Converts raw SMILES strings (Ligands, Additives, Bases) into 2048-bit Morgan Fingerprints (ECFP4) using `RDKit`.
2. Modeling: Uses a Random Forest Regressor to map chemical features to Reaction Yield (%).
3. Bayesian Optimization: Implements an Upper Confidence Bound (UCB) acquisition function to balance Exploitation (high yield) vs. Exploration (high uncertainty).

**Data:** Doyle-Buchwald High-Throughput Experimentation Dataset

## Result
- In a simulation using real-world data, this system found a 92.1% yield reaction on its very first try, vastly outperforming standard random guessing.
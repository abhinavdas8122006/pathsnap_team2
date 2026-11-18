#!/bin/bash
# Runs the whole training pipeline in order. Usage (inside pathsnap, venv active):  bash run_pipeline.sh
set -e
python scripts/02_build_features.py
python scripts/models/train_logistic_regression.py
python scripts/models/train_gaussian_nb.py
python scripts/models/train_random_forest.py
python scripts/models/train_hist_gradient_boosting.py
python scripts/models/train_mlp.py
python scripts/04_compare_models.py
python scripts/05_bayesian_optimisation.py
python scripts/06_hmm_viterbi.py
echo "PIPELINE FINISHED"

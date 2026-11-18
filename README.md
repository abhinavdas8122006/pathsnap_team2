# PathSnap – Team 2
Map-matching noisy GPS points to road segments (ML classifier + HMM/Viterbi).

## Setup
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Put the dataset in data/train and data/student_test.

## Run
python scripts/02_build_features.py   # preprocessing, candidates, labels, features
python scripts/03_train_models.py     # 5 models, GroupKFold CV, student-test evaluation
notebooks/01_eda.py                   # dataset exploration + Folium map (run cells in VS Code)

## Folders
src/ – reusable code (loading, features, metrics)
scripts/ – pipeline steps
notebooks/ – exploration
outputs/ – results, plots, maps
models/ – trained models

AI assistance was used for guidance and code scaffolding; all experiments were run by the team.

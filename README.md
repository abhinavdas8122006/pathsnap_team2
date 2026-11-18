# PathSnap – Team 2
Map-matching noisy GPS trajectories to a road network: candidate generation → HistGradientBoosting
candidate classifier (Bayesian-tuned with Optuna) → HMM + Viterbi route reconstruction, served by a
FastAPI backend (Laptop 1) and a Streamlit evaluator interface (Laptop 2).

AI assistance was used for guidance and code scaffolding; all experiments were run by the team and
all reported numbers come from the outputs in this repository.

## Results (student_test, 10 trips)
| Stage | Top-1 accuracy | Route continuity |
|---|---|---|
| HistGradientBoosting, untuned | 89.4% | 93.5% |
| + Bayesian optimisation | 90.2% | 93.9% |
| + HMM/Viterbi (final system) | **94.3%** | **97.9%** |

Source files: `outputs/model_comparison.csv`, `outputs/optimisation/tuned_vs_untuned.csv`,
`outputs/hmm/classifier_vs_hmm.csv`, `outputs/hmm/hmm_per_trip.csv`.

## Repository layout
| Folder | Contents |
|---|---|
| `src/` | `data_loader.py` (reading files, degrees→metres), `features.py` (cleaning, candidates, label alignment, features), `evaluation.py` (metrics), `train_common.py` (GroupKFold CV + evaluation), `hmm.py` (HMM + Viterbi) |
| `scripts/` | `02_build_features.py`, `models/` (one training script per model), `04_compare_models.py`, `05_bayesian_optimisation.py`, `06_hmm_viterbi.py` |
| `notebooks/` | `01_eda.py` – dataset exploration and Folium map (VS Code cells) |
| `api/` | `main.py` – FastAPI backend (Laptop 1) |
| `ui/` | `app.py` – Streamlit evaluator interface (Laptop 2) |
| `models/` | trained models, `HistGradientBoosting_tuned.joblib`, `hmm_config.json` |
| `outputs/` | results tables, plots, confusion matrices, optimisation record, HMM results, Folium maps; `v1/` = first feature version |

## Installation (Python 3.13)
```bash
git clone https://github.com/abhinavdas8122006/pathsnap_team2.git
cd pathsnap_team2
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Place the dataset (not committed) as `data/train/<trip_id>/...` and `data/student_test/<trip_id>/...`.

## Reproducing the experiments
```bash
bash run_pipeline.sh
```
Runs, in order: feature building → 5 model scripts (5-fold GroupKFold on the 80 train trips,
evaluation on the 10 student_test trips) → model comparison → Bayesian optimisation (Optuna TPE,
20 trials) → HMM decoder tuning (on train trips, out-of-fold) and classifier-vs-HMM evaluation.
Seed 42 throughout. Evaluator-test data is never used.

## Deployment (two laptops, same network)
**Laptop 1 – model + API**
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000          # serves data/evaluator_test
# practice with the student trips:
PATHSNAP_PARTITION=student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0                                     # address to give Laptop 2
```
Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. Request logs are printed
in this terminal. Ground-truth routes stay on Laptop 1: prediction runs with labels disabled and only
accuracy figures are returned.

**Laptop 2 – evaluator UI** (needs only the `ui/` folder)
```bash
python3.13 -m venv uienv && source uienv/bin/activate
pip install -r ui/requirements.txt
streamlit run ui/app.py
```
Enter `http://<Laptop 1 address>:8000` in the sidebar, choose a trip, press **Reconstruct route**.

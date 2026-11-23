# Verification record — 2026-11-23

Executed on Linux with Python 3.12 in an isolated virtual environment. Exact installed dependency versions are in `requirements-tested.txt` (NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, Streamlit 1.67.0, pytest 9.2.0).

- Actual PoissonRegressor training, calibration and held-out inference executed on the original generated data.
- `python -m pytest -q`: **20 passed**. Tests cover deterministic data, daily chronology, exact finite-sample rank, malformed inputs, feature/target isolation, overlapping calibration rejection, positive predictions and valid intervals, training-only baseline comparison, held-out target perturbation invariance, and Streamlit AppTest control interactions.
- `python -m pip check`: no broken requirements.
- `python -m compileall -q app.py demand.py evaluate.py tests`: passed.
- Streamlit launched as an actual subprocess bound to loopback; `/_stcore/health` returned **HTTP 200, ok**; stopped after checking. AppTest additionally executed application logic, rendered widgets and evaluation data, and reran after rain/holiday controls changed.
- Reproducible benchmark: 438 training, 146 calibration, 146 test rows; MAE 11.9794 vs weekday baseline 33.8661; empirical interval coverage 141/146 (96.58%); mean width 70.8493 rentals.

No real bike-rental dataset, third-party weather API, public deployment, Windows/macOS run, or manual browser visual audit was performed. CSV download controls are present but browser download behavior was not end-to-end tested. HTTP health is a server-liveness check, not a substitute for UI testing. No real-world accuracy or interval-coverage guarantee is claimed.

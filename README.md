# BikeWindow ML 🚲

Estimate a fictional bike station's daily rental count **and a calibrated uncertainty window**, then inspect prediction error and empirical interval coverage on a later time block. Runs locally without API keys, external data downloads, or paid services after dependency installation.

This is an educational ML project built around **original synthetic data**. It is not a Bengaluru station forecast, a real transport model, or an inventory recommendation.

## Run

Python 3.12 is the tested version (Linux). Python 3.11+ should work; Windows/macOS and other versions have not been verified.

```bash
git clone https://github.com/AdarshVasth/bikewindow-ml.git
cd bikewindow-ml
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements-tested.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local address printed by Streamlit. The included CSV is enough to run the demo. Change date, forecast temperature, forecast rain, and holiday status; download one estimate or the held-out predictions. No real weather service is connected.

`requirements-tested.txt` records exact verified dependencies; `requirements.txt` describes supported version ranges for a fresh resolver. The first installation needs internet access. Subsequent inference/training is local; telemetry is disabled in the project configuration. No user data or serialized model files are uploaded or loaded.

## Reproduce and test

```bash
python evaluate.py
python -m pytest -q
python -m pip check
```

`evaluate.py` deterministically regenerates the sample CSV, metrics JSON and held-out prediction CSV. Floating point results may differ slightly across platforms/library versions. Training is quick on this small dataset. A container may emit a harmless joblib physical-core detection warning; optionally set `LOKY_MAX_CPU_COUNT=2` in that environment.

## Architecture

- `demand.py`: schema checks, calendar/weather features, seeded simulator, chronological partitions, trained estimator, residual calibration, evaluation.
- `app.py`: cached local model and three demo tabs: scenario, evaluation, learning.
- `evaluate.py`: reproducible data and benchmark artifacts.
- `tests/`: invalid-input checks, exact quantile rank, chronology, target leakage guard, fitted-model performance, and interactive Streamlit smoke test.
- `data/`: 730 original fictional daily observations and reproducible benchmark outputs.
- `docs/LEARNING.md`: a short internship-oriented guide and experiments.

The model uses `StandardScaler → PoissonRegressor(alpha=0.05, max_iter=1000)` with a log link, so expected demand is positive. Seven features describe temperature, a quadratic temperature term, rainfall, weekend, holiday, and annual sine/cosine seasonality. Date is a calendar input, not a demand-label shortcut. Targets never enter feature construction.

## Training, calibration and evaluation

Dates run consecutively from 2024-01-01. The first 438 rows train the scaler and regression; the next 146 rows set interval width; the last 146 rows are untouched evaluation data. There is no random time shuffle. A weekday-mean baseline is fitted on the training block only. The displayed model stays fixed after calibration; it is not refitted on test labels.

For calibration absolute residuals `r`, choose the sorted residual at 1-based rank `ceil((n+1)*0.9)`. The daily window is `[max(0, floor(prediction-q)), ceil(prediction+q)]`. If a requested coverage cannot be supported by the number of residuals, the helper rejects it instead of silently using an interpolated quantile. This is a split-conformal-style calibration rule. The app fixes nominal coverage at 90%.

Measured on the included 146-day synthetic test block, with the recorded environment:

| Metric | Result |
|---|---:|
| Model mean absolute error | 11.98 rentals/day |
| Training weekday-mean baseline MAE | 33.87 rentals/day |
| Empirical interval coverage | 96.58% (141/146 days) |
| Mean interval width | 70.85 rentals |

These are **simulation results**, not evidence of real-world accuracy. The generator deliberately has smooth calendar/weather effects similar to the model's feature family. This makes learning easier. Results are one fixed split, not cross-validation or a production backtest. Wider windows can cover more days while being less informative; evaluate width and coverage together.

## Limitations

- Standard split-conformal marginal guarantees require exchangeability of calibration and future observations. Chronological series, seasonality, dependence and distribution shift do not generally satisfy that assumption. **No 90% coverage guarantee is claimed here**, especially for any specific date or weather subgroup.
- The residual window has constant width and does not explicitly model higher variance at larger expected counts. Outward integer rounding is conservative.
- Weather is assumed supplied in advance. Forecast uncertainty, station capacity, empty docks, events, trends and real rider behavior are absent. Rentals are not necessarily unconstrained demand in real operations.
- Allowed input bounds (-5–45°C, 0–80 mm) are validation limits, not evidence of training support. The UI warns for unusual weather, but this is not a formal out-of-distribution detector.
- Public-holiday status is manual; no country-specific holiday calendar exists. Dates are timezone-free daily values.
- No custom CSV upload, authentication, model-serving API, or cloud deployment is included. Never use this to make operational allocation decisions without real data validation.

## References and API checks

Official documentation checked on 2026-11-23:

- [scikit-learn PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html): count targets, regularization, log link.
- [Streamlit date_input](https://docs.streamlit.io/develop/api-reference/widgets/st.date_input) and [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
- [NumPy quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html): quantile conventions; our exact order statistic is implemented explicitly.
- [Ryan Tibshirani's conformal prediction lecture notes](https://www.stat.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal.pdf): calibration rank and exchangeability assumptions.

Code and all generated sample data are provided under the MIT license. No personal records or third-party dataset is bundled. See [validation](docs/VALIDATION.md) and [learning guide](docs/LEARNING.md).

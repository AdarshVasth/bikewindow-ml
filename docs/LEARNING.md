# Learn BikeWindow in one sitting

1. **Read the data (10 min).** Inspect `data/daily_demand.csv`. Identify the target and inputs. Explain why using same-day realized weather would be an optimistic evaluation if only imperfect weather forecasts were available in production.
2. **Trace the model (15 min).** Read `features` and `fit_model`. A Poisson regression uses `exp(w·x+b)` for a positive expected count. Scaling fits on training data only. Coefficients are learned, not hard-coded predictions. The simulator uses a Poisson/log-normal mixture, so it is overdispersed relative to a pure Poisson model.
3. **Understand the three blocks (10 min).** Training learns weights. Calibration measures residual size. Test evaluates the fixed result. Explain why reusing calibration labels as test labels makes measured coverage optimistic.
4. **Compute a window by hand (10 min).** For nine residuals 1–9 and nominal 80% coverage, rank is `ceil(10*0.8)=8`, so q=8. For predicted count 20.2, the rounded window is [12,29]. Explain why a prediction interval is different from uncertainty in the mean estimate.
5. **Run the tests (10 min).** Find the test that adds 5,000 to test labels. Predictions and calibration width must remain identical. This is a leakage check, not an accuracy check.
6. **Discuss the limitations (10 min).** Why is 96.58% empirical coverage different from a guarantee? Why can seasonality and time dependence break exchangeability? Why is an 80-rental-wide interval not automatically useful?

## Small extensions

- Evaluate several chronological rolling windows; reserve a final block before selecting hyperparameters.
- Compare coverage separately on rainy and dry days, and report the sample sizes.
- Inject an event shock into the final block. Freeze the original model; measure how coverage changes.
- Compare against a lag-7 baseline carefully: ensure the past observation would exist at the actual forecast horizon. The included baseline avoids this horizon issue by using training weekday means.
- After choosing a licensed real dataset, replace the simulator and use weather forecasts available at prediction time. Keep a data provenance/license file. Do not simply relabel simulation metrics as real results.

## Internship explanation

“I built a count-regression demo and calibrated prediction windows on a separate time block. I tested that targets never enter features and that changing test labels cannot affect model predictions or interval width. I compared error against a training-only weekday baseline. The result works on original synthetic data, but chronological dependence means I do not claim the usual exchangeable-data conformal guarantee.”

Be ready to explain MAE, chronological splitting, the log link, regularization, coverage versus interval width, distribution shift and why synthetic performance is limited evidence.

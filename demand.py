"""Local count regression and chronological residual calibration."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLUMNS = ['date', 'temperature_c', 'rain_mm', 'holiday', 'rentals']


def validate(frame, target=False):
    required = COLUMNS if target else COLUMNS[:-1]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError('Missing required columns or empty data.')
    df = frame.copy()
    try:
        df['date'] = pd.to_datetime(df['date'], errors='raise', format='ISO8601')
        if df['date'].isna().any() or df['date'].dt.tz is not None:
            raise ValueError('Use timezone-free dates.')
        if not (df['date'] == df['date'].dt.normalize()).all():
            raise ValueError('Use daily dates without a time component.')
        for col in required[1:]:
            df[col] = pd.to_numeric(df[col], errors='raise')
            if not np.isfinite(df[col]).all():
                raise ValueError('Numeric inputs must be finite.')
    except (TypeError, OverflowError) as exc:
        raise ValueError('Invalid input types.') from exc
    if not df.temperature_c.between(-5, 45).all():
        raise ValueError('Temperature must be -5 to 45 C.')
    if not df.rain_mm.between(0, 80).all():
        raise ValueError('Rain must be 0 to 80 mm.')
    if not df.holiday.isin([0, 1]).all():
        raise ValueError('Holiday must be 0 or 1.')
    if target and ((df.rentals < 0).any() or (df.rentals % 1 != 0).any()):
        raise ValueError('Rentals must be nonnegative integer counts.')
    return df


def features(frame):
    df = validate(frame)
    day = df.date.dt.day_of_year.to_numpy()
    temp = df.temperature_c.to_numpy()
    return np.column_stack([
        temp, (temp - 22) ** 2, df.rain_mm,
        (df.date.dt.day_of_week >= 5).astype(int), df.holiday,
        np.sin(2 * np.pi * day / 365.25),
        np.cos(2 * np.pi * day / 365.25),
    ])


def sample_data(seed=2026, days=730):
    """Original fictional counts; no real riders, stations, or weather records."""
    if not isinstance(days, int) or days < 1:
        raise ValueError('days must be a positive integer')
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=days)
    season = np.sin(2 * np.pi * dates.day_of_year.to_numpy() / 365.25)
    temp = np.clip(22 + 8 * season + rng.normal(0, 3, days), -5, 45)
    rain = np.minimum(rng.exponential(5, days) * (rng.random(days) < .3), 80)
    holiday = (rng.random(days) < .05).astype(int)
    weekend = (dates.day_of_week >= 5).astype(int)
    # Unobserved daily multiplicative noise makes intervals meaningful.
    log_mean = (4.7 + .025 * (temp - 22) - .0018 * (temp - 22)**2
                - .045 * rain + .24 * weekend - .2 * holiday + .14 * season)
    mean = np.exp(log_mean + rng.normal(0, .16, days))
    return pd.DataFrame(dict(date=dates.strftime('%Y-%m-%d'),
        temperature_c=np.round(temp, 3), rain_mm=np.round(rain, 3),
        holiday=holiday, rentals=rng.poisson(mean)))


def chronological_split(frame):
    df = validate(frame, target=True).sort_values('date').reset_index(drop=True)
    if len(df) < 100 or df.date.duplicated().any():
        raise ValueError('Need at least 100 unique daily dates.')
    if not (df.date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError('Expected consecutive daily observations.')
    a, b = int(len(df) * .6), int(len(df) * .8)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def residual_radius(residuals, coverage=.9):
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Residuals must be a nonempty finite nonnegative vector.')
    if not 0 < coverage < 1:
        raise ValueError('Coverage must lie between 0 and 1.')
    # Exact finite-sample order statistic, NOT NumPy's interpolated quantile.
    rank = math.ceil((len(values) + 1) * coverage)
    if rank > len(values):
        raise ValueError('Too few calibration rows for the requested coverage.')
    return float(np.sort(values)[rank - 1])


@dataclass
class DemandModel:
    estimator: object
    radius: float
    coverage: float

    def predict(self, frame):
        point = self.estimator.predict(features(frame))
        return pd.DataFrame({'expected': point,
            'lower': np.maximum(0, np.floor(point - self.radius)),
            'upper': np.ceil(point + self.radius)})


def fit_model(train, calibration, coverage=.9):
    train, calibration = validate(train, True), validate(calibration, True)
    if train.date.max() >= calibration.date.min():
        raise ValueError('Calibration must strictly follow training dates.')
    if train.rentals.sum() <= 0:
        raise ValueError('Training needs positive demand.')
    estimator = make_pipeline(StandardScaler(), PoissonRegressor(alpha=.05, max_iter=1000))
    estimator.fit(features(train), train.rentals)
    residuals = np.abs(calibration.rentals.to_numpy() - estimator.predict(features(calibration)))
    return DemandModel(estimator, residual_radius(residuals, coverage), coverage)


def benchmark(frame):
    train, calibration, test = chronological_split(frame)
    model = fit_model(train, calibration)
    pred = model.predict(test)
    actual = test.rentals.to_numpy()
    weekday_means = train.groupby(train.date.dt.day_of_week).rentals.mean()
    baseline = test.date.dt.day_of_week.map(weekday_means).to_numpy()
    details = pd.DataFrame({'date': test.date.dt.strftime('%Y-%m-%d'),
        'actual': actual, 'expected': pred.expected.to_numpy(),
        'lower': pred.lower.to_numpy(), 'upper': pred.upper.to_numpy(), 'baseline': baseline})
    metrics = {'train_rows': len(train), 'calibration_rows': len(calibration), 'test_rows': len(test),
        'mae': float(np.mean(np.abs(actual - pred.expected.to_numpy()))),
        'weekday_baseline_mae': float(np.mean(np.abs(actual - baseline))),
        'nominal_coverage': model.coverage,
        'test_coverage': float(np.mean((actual >= pred.lower) & (actual <= pred.upper))),
        'mean_interval_width': float(np.mean(pred.upper - pred.lower)),
        'calibration_radius': model.radius}
    return model, metrics, details

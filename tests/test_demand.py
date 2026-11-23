import numpy as np
import pandas as pd
import pytest
from demand import sample_data, chronological_split, fit_model, residual_radius, benchmark, features, validate


def test_original_data_reproducible_and_counts():
    a = sample_data()
    pd.testing.assert_frame_equal(a, sample_data())
    assert len(a) == 730 and (a.rentals >= 0).all()
    assert not a.equals(sample_data(seed=2))


def test_split_no_leakage_and_no_mutation():
    df = sample_data()
    original = df.copy()
    a, b, c = chronological_split(df.sample(frac=1, random_state=4))
    assert a.date.max() < b.date.min() < c.date.min()
    assert set(a.date).isdisjoint(c.date)
    assert len(a) + len(b) + len(c) == len(df)
    pd.testing.assert_frame_equal(df, original)


def test_finite_sample_rank():
    assert residual_radius(np.arange(1, 10), .8) == 8
    assert residual_radius(np.arange(1, 10), .9) == 9
    with pytest.raises(ValueError):
        residual_radius([1, 2], .99)


@pytest.mark.parametrize('values', [[], [-1], [np.nan], [[1, 2]]])
def test_invalid_residuals(values):
    with pytest.raises(ValueError):
        residual_radius(values)


@pytest.mark.parametrize('col,value', [('rain_mm', -1), ('temperature_c', 46), ('holiday', 2), ('rentals', .5), ('rentals', -1), ('rain_mm', np.inf), ('date', 'bad')])
def test_input_validation(col, value):
    df = sample_data(days=5)
    df[col] = df[col].astype(object)
    df.loc[0, col] = value
    with pytest.raises(ValueError):
        validate(df, True)


def test_reject_duplicate_and_missing_days():
    df = sample_data()
    with pytest.raises(ValueError):
        chronological_split(pd.concat([df, df.iloc[:1]]))
    with pytest.raises(ValueError):
        chronological_split(df.drop(index=5))


def test_target_is_not_a_feature():
    df = sample_data(days=20)
    changed = df.assign(rentals=99999)
    np.testing.assert_array_equal(features(df), features(changed))


def test_overlap_rejected():
    a, b, _ = chronological_split(sample_data())
    with pytest.raises(ValueError):
        fit_model(a, a)
    with pytest.raises(ValueError):
        fit_model(a.assign(rentals=0), b)


def test_real_fit_outperforms_baseline_and_interval_is_valid():
    model, metrics, details = benchmark(sample_data())
    assert metrics['mae'] < metrics['weekday_baseline_mae']
    assert np.isfinite(details.select_dtypes('number')).all().all()
    assert (details.lower >= 0).all()
    assert (details.lower <= details.expected).all()
    assert (details.upper >= details.expected).all()
    # Broad regression guard on this fixed simulation, not a general coverage guarantee.
    assert .7 < metrics['test_coverage'] <= 1


def test_test_labels_cannot_change_model_or_radius():
    df = sample_data()
    model, _, detail = benchmark(df)
    df.loc[int(.8 * len(df)):, 'rentals'] += 5000
    other, _, changed = benchmark(df)
    assert model.radius == other.radius
    np.testing.assert_allclose(detail.expected, changed.expected)

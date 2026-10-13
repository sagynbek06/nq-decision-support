"""
Tests for the MarketDataSource seam (src/data/source.py) and the synthetic source
(src/data/synthetic_intraday.py).

The synthetic source is checked against known answers: its contract, its determinism and prefix
consistency, the U-shape and Clark (1973) links it was built with, the exact planted-edge identity,
the event jump, and the option-chain snapping rule. Tolerances are standard errors computed from the
data itself, so each one states how many standard errors it allows.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import synthetic_intraday as synth
from src.data.schema import (
    validate_bars,
    validate_events,
    validate_option_chain,
    validate_quotes,
)
from src.data.source import MarketDataSource
from src.data.synthetic_intraday import (
    EVENT_JUMP,
    EVENT_VOL_LINK,
    SESSION_MINUTES,
    SyntheticIntradaySource,
    intraday_volatility_profile,
)

NY = "America/New_York"
N_SESSIONS = 120
SEED = 18
WIDE_START = pd.Timestamp("2024-01-01", tz=NY)
WIDE_END = pd.Timestamp("2030-01-01", tz=NY)
BAR_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume", "bid_volume", "ask_volume"}


def _bar_returns(bars):
    """Bar return log(close / open). Each bar opens at the previous bar's close (assumption 1), so this is the minute return."""
    return np.log(bars["close"].to_numpy() / bars["open"].to_numpy())


@pytest.fixture(scope="module")
def source():
    return SyntheticIntradaySource(n_sessions=N_SESSIONS, seed=SEED, planted_edge=0.0)


@pytest.fixture(scope="module")
def bars(source):
    return source.bars(WIDE_START, WIDE_END)


@pytest.fixture(scope="module")
def hidden(source):
    return source.hidden_state_for_validation()


@pytest.fixture(scope="module")
def planted_pair():
    null_source = SyntheticIntradaySource(n_sessions=N_SESSIONS, seed=SEED, planted_edge=0.0)
    planted = SyntheticIntradaySource(n_sessions=N_SESSIONS, seed=SEED, planted_edge=0.05)
    return null_source, planted


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------

def test_the_seam_is_abstract_and_the_synthetic_source_implements_it():
    assert {"bars", "depth", "option_chain", "events"} <= MarketDataSource.__abstractmethods__
    assert issubclass(SyntheticIntradaySource, MarketDataSource)
    with pytest.raises(TypeError):
        MarketDataSource()


def test_every_method_returns_a_table_that_passes_its_validator(source):
    for freq in ("1min", "5min", "15min"):
        assert validate_bars(source.bars(WIDE_START, WIDE_END, freq), require_aggressor=True) is not None
    assert validate_quotes(source.depth(WIDE_START, WIDE_END)) is not None
    assert validate_option_chain(source.option_chain(pd.Timestamp("2024-02-14 11:00", tz=NY))) is not None
    assert validate_events(source.events(WIDE_START, WIDE_END)) is not None


def test_bars_expose_only_the_documented_columns(bars):
    assert set(bars.columns) == BAR_COLUMNS  # hidden pressure and the true volatility never leak through


def test_windows_are_half_open_and_must_be_new_york_aware(source):
    with pytest.raises(ValueError, match="naive"):
        source.bars(pd.Timestamp("2024-01-02 09:30"), WIDE_END)
    with pytest.raises(ValueError, match="before end"):
        source.bars(WIDE_END, WIDE_START)
    end = pd.Timestamp("2024-01-02 09:35", tz=NY)
    window = source.bars(pd.Timestamp("2024-01-02 09:30", tz=NY), end)
    assert len(window) == 5 and window["timestamp"].max() < end  # the bar stamped 09:35 is excluded


def test_unsupported_frequency_is_rejected(source):
    with pytest.raises(ValueError, match="unsupported frequency"):
        source.bars(WIDE_START, WIDE_END, freq="30min")


# ---------------------------------------------------------------------------
# Structure and reproducibility
# ---------------------------------------------------------------------------

def test_bars_are_one_minute_cash_session_bars_in_new_york_time(bars):
    assert len(bars) == N_SESSIONS * SESSION_MINUTES
    assert str(bars["timestamp"].dt.tz) == NY
    per_day = bars.groupby(bars["timestamp"].dt.date)
    assert per_day.size().eq(SESSION_MINUTES).all()
    first = per_day["timestamp"].min().dt.strftime("%H:%M").unique()
    last = per_day["timestamp"].max().dt.strftime("%H:%M").unique()
    assert set(first) == {"09:30"} and set(last) == {"15:59"}
    steps = bars["timestamp"].diff().dropna()
    within_session = steps[steps < pd.Timedelta(hours=1)]
    assert (within_session == pd.Timedelta(minutes=1)).all()  # one-minute steps inside every session
    assert (steps >= pd.Timedelta(hours=1)).sum() == N_SESSIONS - 1  # one overnight gap between sessions


def test_same_seed_gives_identical_data_and_a_different_seed_does_not(source):
    twin = SyntheticIntradaySource(n_sessions=N_SESSIONS, seed=SEED, planted_edge=0.0)
    pd.testing.assert_frame_equal(source.bars(WIDE_START, WIDE_END), twin.bars(WIDE_START, WIDE_END))
    pd.testing.assert_frame_equal(source.depth(WIDE_START, WIDE_END), twin.depth(WIDE_START, WIDE_END))
    pd.testing.assert_frame_equal(source.events(WIDE_START, WIDE_END), twin.events(WIDE_START, WIDE_END))
    pd.testing.assert_frame_equal(source.hidden_state_for_validation(), twin.hidden_state_for_validation())
    other = SyntheticIntradaySource(n_sessions=N_SESSIONS, seed=SEED + 1, planted_edge=0.0)
    assert not source.bars(WIDE_START, WIDE_END)["close"].equals(other.bars(WIDE_START, WIDE_END)["close"])


def test_the_first_sessions_are_identical_whatever_the_horizon(source):
    longer = SyntheticIntradaySource(n_sessions=150, seed=SEED, planted_edge=0.0)
    end = pd.Timestamp(source.bars(WIDE_START, WIDE_END)["timestamp"].iloc[60 * SESSION_MINUTES - 1]) + pd.Timedelta(minutes=1)
    pd.testing.assert_frame_equal(source.bars(WIDE_START, end), longer.bars(WIDE_START, end))


def test_resampling_matches_a_hand_aggregation(source, bars):
    five = source.bars(WIDE_START, WIDE_END, freq="5min")
    minute = bars.copy()
    minute["bin"] = minute["timestamp"].dt.floor("5min")
    expected = minute.groupby("bin").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), bid_volume=("bid_volume", "sum"), ask_volume=("ask_volume", "sum"))
    got = five.set_index("timestamp")
    np.testing.assert_array_equal(got.index.to_numpy(), expected.index.to_numpy())
    for column in expected.columns:
        np.testing.assert_array_equal(got[column].to_numpy(), expected[column].to_numpy(), err_msg=column)


# ---------------------------------------------------------------------------
# Known structure: the U-shape, volatility clustering and the absence of return autocorrelation
# ---------------------------------------------------------------------------

def test_open_volatility_exceeds_midday_by_the_profile_ratio(bars):
    """
    Known answer: the expected ratio of per-minute mean squared returns, open window over midday window,
    is the ratio of the profile's mean f^2 over those windows (assumption 3). The statistic compares sums
    over windows of 30 and 60 minutes, so it is rescaled to per-minute means. Its standard error is
    cluster-robust across sessions (a delta-method ratio), which keeps the regime-level volatility from
    counting as noise. The tolerance is 3 standard errors.
    """
    profile = intraday_volatility_profile()
    returns = _bar_returns(bars).reshape(N_SESSIONS, SESSION_MINUTES)
    open_idx, mid_idx = np.arange(0, 30), np.arange(150, 210)
    a = (returns[:, open_idx] ** 2).sum(axis=1)
    b = (returns[:, mid_idx] ** 2).sum(axis=1)
    scale = len(mid_idx) / len(open_idx)
    ratio_of_sums = a.sum() / b.sum()
    observed = scale * ratio_of_sums
    se = scale * np.sqrt(np.sum((a - ratio_of_sums * b) ** 2)) / b.sum()
    expected = np.mean(profile[open_idx] ** 2) / np.mean(profile[mid_idx] ** 2)
    assert abs(observed - expected) <= 3 * se, f"observed {observed:.3f}, expected {expected:.3f}, se {se:.3f}"
    assert observed > 2.0  # the open is at least twice as volatile as midday per minute


def test_midday_volatility_is_lower_than_open_volatility(bars):
    returns = np.abs(_bar_returns(bars)).reshape(N_SESSIONS, SESSION_MINUTES)
    assert returns[:, :30].mean() > returns[:, 150:210].mean()


def test_lag1_autocorrelation_of_returns_is_near_zero_without_a_planted_edge(bars, hidden):
    """
    Tolerance: three standard errors of the lag-1 sample autocorrelation under the null of no autocorrelation.
    The standard error is 1/sqrt(N) inflated by the generator's own volatility structure, E[s_t^2 s_t+1^2] /
    E[s^2]^2, because heteroskedastic returns make the sample autocorrelation noisier than in the iid case.
    """
    r = _bar_returns(bars)
    rho1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    s = hidden["sigma"].to_numpy()
    inflation = np.mean((s[:-1] ** 2) * (s[1:] ** 2)) / np.mean(s ** 2) ** 2
    se = np.sqrt(inflation / (len(r) - 1))
    assert abs(rho1) <= 3 * se, f"rho1 {rho1:+.5f} exceeds 3 SE = {3 * se:.5f}"


def test_volatility_clustering_is_present_without_a_planted_edge(bars):
    """
    Under no clustering the lag-1 autocorrelation of |r| has standard deviation of about 1/sqrt(N), or 0.005
    here, so 0.02 is more than four of those. The U-shape and the regime volatility both create clustering,
    and this checks that they do.
    """
    absolute = np.abs(_bar_returns(bars) - _bar_returns(bars).mean())
    assert np.corrcoef(absolute[:-1], absolute[1:])[0, 1] > 0.02


# ---------------------------------------------------------------------------
# Known structure: Clark volume, the aggressor split and events
# ---------------------------------------------------------------------------

def test_volume_is_correlated_with_volatility_after_clark(bars, hidden):
    log_volume = np.log(bars["volume"].to_numpy().astype(float))
    log_sigma = np.log(hidden["sigma"].to_numpy())
    assert np.corrcoef(log_volume, log_sigma)[0, 1] > 0.5


def test_aggressor_split_follows_the_sign_of_the_return(bars, hidden):
    standardized = _bar_returns(bars) / hidden["sigma"].to_numpy()
    imbalance = (bars["ask_volume"] - bars["bid_volume"]).to_numpy() / bars["volume"].to_numpy()
    assert np.corrcoef(standardized, imbalance)[0, 1] > 0.3


def test_event_jump_and_volatility_multiplier_follow_the_surprise_exactly(source, bars, hidden):
    """Known answer: at a release minute the jump is EVENT_JUMP * surprise * sigma, and sigma is multiplied by (1 + EVENT_VOL_LINK * |surprise|)."""
    profile = intraday_volatility_profile()
    stamps = pd.DatetimeIndex(bars["timestamp"])
    events = source.events(WIDE_START, WIDE_END)
    assert len(events) > 0
    for ts, surprise in zip(events["timestamp"], events["surprise"]):
        index = stamps.get_loc(ts)
        minute = index % SESSION_MINUTES
        assert hidden["jump"].iloc[index] == pytest.approx(EVENT_JUMP * surprise * hidden["sigma"].iloc[index], abs=1e-15)
        ratio = hidden["sigma"].iloc[index] / (hidden["sigma"].iloc[index + 1] * profile[minute] / profile[minute + 1])
        assert ratio == pytest.approx(1.0 + EVENT_VOL_LINK * abs(surprise), rel=1e-12)


def test_events_follow_the_declared_schedule(source):
    dates = pd.bdate_range("2024-01-02", periods=N_SESSIONS)
    first_of_month = [k for k in range(N_SESSIONS) if k == 0 or dates[k].month != dates[k - 1].month]
    fomc_sessions = list(range(synth.FOMC_FIRST_SESSION, N_SESSIONS, synth.FOMC_EVERY_SESSIONS))
    events = source.events(WIDE_START, WIDE_END)
    ism = events[events["event_type"] == synth.ISM_EVENT_TYPE]
    fomc = events[events["event_type"] == synth.FOMC_EVENT_TYPE]
    assert ism["timestamp"].dt.strftime("%H:%M").eq("10:00").all()
    assert fomc["timestamp"].dt.strftime("%H:%M").eq("14:00").all()
    assert [pd.Timestamp(t).date() for t in ism["timestamp"]] == [dates[k].date() for k in first_of_month]
    assert [pd.Timestamp(t).date() for t in fomc["timestamp"]] == [dates[k].date() for k in fomc_sessions]


# ---------------------------------------------------------------------------
# Planted edge
# ---------------------------------------------------------------------------

def test_planted_edge_enters_the_normalised_return_exactly(planted_pair):
    null_source, planted = planted_pair
    h0 = null_source.hidden_state_for_validation()
    hp = planted.hidden_state_for_validation()
    z_lag = np.r_[0.0, h0["pressure"].to_numpy()[:-1]]
    difference = hp["normalized_residual"].to_numpy() - h0["normalized_residual"].to_numpy()
    np.testing.assert_allclose(difference, 0.05 * z_lag, rtol=0, atol=1e-12)


def test_planted_edge_slope_matches_its_parameter_within_three_standard_errors(planted_pair):
    """
    The regression of the normalised residual on the lagged hidden pressure has slope planted_edge. Its
    standard error is the residual standard deviation over sqrt(N) times the standard deviation of the
    regressor, both estimated from the data.
    """
    null_source, planted = planted_pair
    h0 = null_source.hidden_state_for_validation()
    hp = planted.hidden_state_for_validation()
    z_lag = np.r_[0.0, h0["pressure"].to_numpy()[:-1]]
    for hidden_frame, true_slope in ((h0, 0.0), (hp, 0.05)):
        x = hidden_frame["normalized_residual"].to_numpy()
        slope = np.cov(x, z_lag)[0, 1] / np.var(z_lag, ddof=1)
        resid_sd = np.std(x - slope * z_lag, ddof=1)
        se = resid_sd / (np.sqrt(len(z_lag)) * np.std(z_lag, ddof=1))
        assert abs(slope - true_slope) <= 3 * se, f"slope {slope:+.5f} vs {true_slope}, se {se:.5f}"


# ---------------------------------------------------------------------------
# Depth and the option chain
# ---------------------------------------------------------------------------

def test_depth_is_two_ticks_around_each_bar_close(source, bars):
    depth = source.depth(WIDE_START, WIDE_END)
    assert depth["timestamp"].equals(bars["timestamp"])
    np.testing.assert_allclose(depth["bid"].to_numpy(), bars["close"].to_numpy() - 0.25, rtol=0, atol=1e-9)
    np.testing.assert_allclose(depth["ask"].to_numpy(), bars["close"].to_numpy() + 0.25, rtol=0, atol=1e-9)
    assert (depth["bid_size"] >= 0).all() and (depth["ask_size"] >= 0).all()


def test_option_chain_snaps_to_the_last_bar_at_or_before_as_of(source, bars):
    as_of = pd.Timestamp("2024-02-14 11:00:30", tz=NY)
    chain = source.option_chain(as_of)
    snapped = pd.Timestamp("2024-02-14 11:00", tz=NY)
    assert (chain["as_of"] == snapped).all()
    spot = bars.loc[bars["timestamp"] == snapped, "close"].iloc[0]
    assert abs(chain["strike"].mean() - spot) < 1_000  # strikes centred on spot
    with pytest.raises(ValueError, match="before the first generated bar"):
        source.option_chain(pd.Timestamp("2024-01-01 08:00", tz=NY))


def test_option_chain_is_weekly_and_has_a_full_strike_ladder(source):
    chain = source.option_chain(pd.Timestamp("2024-02-14 11:00", tz=NY))
    expiries = sorted(chain["expiry"].unique())
    assert len(expiries) == synth.CHAIN_EXPIRIES
    assert all(pd.Timestamp(e).weekday() == 4 for e in expiries)  # Fridays
    assert all(pd.Timestamp(e) > pd.Timestamp("2024-02-14") for e in expiries)
    per_expiry = chain.groupby("expiry").size()
    assert per_expiry.eq(2 * synth.CHAIN_HALF_WIDTH + 1).all()
    assert (chain["strike"] % synth.CHAIN_STRIKE_STEP == 0).all()
    assert (chain["put_open_interest"] >= chain["call_open_interest"]).all()  # the fixed put tilt


# ---------------------------------------------------------------------------
# The seam must stay a seam
# ---------------------------------------------------------------------------

FORBIDDEN_CALLS = {"open", "read_csv", "read_parquet", "read_json", "read_excel", "read_sql", "read_pickle", "load"}
ALLOWED_IMPORTS = {"__future__", "math", "abc", "datetime", "typing", "numpy", "pandas", "src"}


def test_data_layer_imports_only_allowed_libraries_and_reads_no_files():
    """Downstream code depends on MarketDataSource only. The data layer itself must not import a vendor library or read a file."""
    data_dir = Path(synth.__file__).resolve().parent
    for path in sorted(data_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                modules = []
            for module in modules:
                assert module.split(".")[0] in ALLOWED_IMPORTS, f"{path.name} imports {module}"
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                assert name not in FORBIDDEN_CALLS, f"{path.name} calls {name}"

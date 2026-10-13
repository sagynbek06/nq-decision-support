"""
Tests for the intraday data contracts (src/data/schema.py).

Each validator is checked two ways: a clean table passes, and each named bad case is rejected with a
ContractError whose message names the rule and the first offending row. The good tables are built from
scratch here, so the tests do not depend on the synthetic generator.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    ContractError,
    cash_session,
    globex_open,
    to_new_york,
    validate_bars,
    validate_events,
    validate_option_chain,
    validate_quotes,
)

NY = "America/New_York"


def _good_bars(n=5, start="2024-01-02 09:30"):
    """A small table that satisfies every bar rule: Tuesday session minutes, on-grid prices, consistent OHLC."""
    stamps = pd.date_range(pd.Timestamp(start, tz=NY), periods=n, freq="min")
    volume = np.full(n, 100, dtype=np.int64)
    bid = np.full(n, 40, dtype=np.int64)
    return pd.DataFrame({
        "timestamp": stamps,
        "open": np.full(n, 18_000.0),
        "high": np.full(n, 18_000.5),
        "low": np.full(n, 17_999.75),
        "close": np.full(n, 18_000.25),
        "volume": volume,
        "bid_volume": bid,
        "ask_volume": volume - bid,
    })


def _with_timestamps(df, stamps):
    out = df.copy()
    out["timestamp"] = pd.DatetimeIndex(stamps)
    return out


# ---------------------------------------------------------------------------
# Bars: the clean table passes, and each named bad case is rejected
# ---------------------------------------------------------------------------

def test_clean_bars_pass_and_are_returned_unchanged():
    df = _good_bars()
    assert validate_bars(df, require_aggressor=True) is df


def test_aggressor_columns_are_optional_unless_required():
    df = _good_bars().drop(columns=["bid_volume", "ask_volume"])
    assert validate_bars(df) is df
    with pytest.raises(ContractError, match="missing required column"):
        validate_bars(df, require_aggressor=True)


def test_non_monotonic_timestamps_are_rejected():
    df = _good_bars()
    stamps = df["timestamp"].tolist()
    stamps[2], stamps[3] = stamps[3], stamps[2]
    with pytest.raises(ContractError, match="not monotonic increasing") as info:
        validate_bars(_with_timestamps(df, stamps))
    assert "row index: 3" in str(info.value)


def test_duplicate_timestamps_are_rejected():
    df = _good_bars()
    doubled = df.iloc[[0, 0, 1, 2, 3]].reset_index(drop=True)
    with pytest.raises(ContractError, match="duplicate timestamps") as info:
        validate_bars(doubled)
    assert "row index: 1" in str(info.value)


@pytest.mark.parametrize("start", [
    "2024-01-06 10:00",  # Saturday
    "2024-01-02 17:00",  # Tuesday daily halt, 17:00 to 18:00
    "2024-01-05 17:00",  # Friday after the 17:00 close
    "2024-01-07 17:00",  # Sunday before the 18:00 open
])
def test_bars_outside_the_globex_session_are_rejected(start):
    with pytest.raises(ContractError, match="outside the Globex session"):
        validate_bars(_good_bars(start=start))


def test_bars_inside_globex_but_outside_cash_are_accepted():
    """Pre-market (08:00) and overnight (19:00) are Globex hours, so they pass the session check."""
    validate_bars(_good_bars(start="2024-01-02 08:00"))
    validate_bars(_good_bars(start="2024-01-02 19:00"))


def test_prices_off_the_tick_grid_are_rejected():
    df = _good_bars()
    df.loc[2, "open"] = 18_000.10
    with pytest.raises(ContractError, match=r"open is off the 0\.25 tick grid") as info:
        validate_bars(df)
    assert "row index: 2" in str(info.value)


def test_high_below_low_is_rejected():
    df = _good_bars()
    df.loc[1, "high"] = 17_999.0
    with pytest.raises(ContractError, match="high < low") as info:
        validate_bars(df)
    assert "row index: 1" in str(info.value)


def test_high_below_close_is_rejected_as_inconsistent_ohlc():
    df = _good_bars()
    df.loc[0, "high"] = 18_000.0  # still above low, but below close (18000.25)
    with pytest.raises(ContractError, match="high is below open or close"):
        validate_bars(df)


def test_low_above_open_is_rejected_as_inconsistent_ohlc():
    df = _good_bars()
    df.loc[0, "low"] = 18_000.5  # still below high, but above open (18000.0)
    with pytest.raises(ContractError, match="low is above open or close"):
        validate_bars(df)


def test_negative_volume_is_rejected():
    df = _good_bars()
    df.loc[4, "volume"] = -5
    with pytest.raises(ContractError, match="negative volume") as info:
        validate_bars(df)
    assert "row index: 4" in str(info.value)


def test_missing_values_are_rejected():
    df = _good_bars()
    df.loc[0, "close"] = np.nan
    with pytest.raises(ContractError, match="missing values"):
        validate_bars(df)


def test_naive_timestamps_are_rejected():
    df = _good_bars()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ContractError, match="naive"):
        validate_bars(df)


def test_timestamps_in_another_zone_are_rejected():
    df = _good_bars()
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    with pytest.raises(ContractError, match="America/New_York"):
        validate_bars(df)


def test_non_integer_volume_is_rejected():
    df = _good_bars()
    df["volume"] = df["volume"].astype(float)
    with pytest.raises(ContractError, match="whole numbers"):
        validate_bars(df)


def test_aggressor_split_that_does_not_sum_to_volume_is_rejected():
    df = _good_bars()
    df.loc[3, "ask_volume"] += 1
    with pytest.raises(ContractError, match="does not equal volume") as info:
        validate_bars(df, require_aggressor=True)
    assert "row index: 3" in str(info.value)


def test_negative_aggressor_volume_is_rejected():
    df = _good_bars()
    df.loc[0, "bid_volume"] = -1
    df.loc[0, "ask_volume"] = df.loc[0, "volume"] + 1  # keeps the sum right, so only the sign is wrong
    with pytest.raises(ContractError, match="negative bid_volume"):
        validate_bars(df, require_aggressor=True)


# ---------------------------------------------------------------------------
# Session calendar: a known truth table for the Globex and cash definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stamp, expected", [
    ("2024-01-01 10:00", True),   # Monday, in session
    ("2024-01-01 17:30", False),  # Monday halt
    ("2024-01-01 18:00", True),   # Monday evening, reopen
    ("2024-01-04 17:59", False),  # Thursday halt
    ("2024-01-04 18:00", True),   # Thursday evening
    ("2024-01-05 16:59", True),   # Friday before the close
    ("2024-01-05 17:00", False),  # Friday after the close
    ("2024-01-06 12:00", False),  # Saturday
    ("2024-01-07 17:59", False),  # Sunday before the open
    ("2024-01-07 18:00", True),   # Sunday open
])
def test_globex_calendar_truth_table(stamp, expected):
    assert bool(globex_open([pd.Timestamp(stamp, tz=NY)])[0]) is expected


@pytest.mark.parametrize("stamp, expected", [
    ("2024-01-02 09:29", False),
    ("2024-01-02 09:30", True),
    ("2024-01-02 15:59", True),
    ("2024-01-02 16:00", False),
    ("2024-01-06 10:00", False),  # Saturday
])
def test_cash_session_boundaries(stamp, expected):
    assert bool(cash_session([pd.Timestamp(stamp, tz=NY)])[0]) is expected


def test_to_new_york_rejects_naive_and_foreign_zones():
    with pytest.raises(ValueError, match="naive"):
        to_new_york(pd.Timestamp("2024-01-02 09:30"))
    with pytest.raises(ValueError, match="America/New_York"):
        to_new_york(pd.Timestamp("2024-01-02 09:30", tz="UTC"))
    assert str(to_new_york(pd.Timestamp("2024-01-02 09:30", tz=NY)).tzinfo) == NY


def test_contract_error_is_a_value_error():
    assert issubclass(ContractError, ValueError)


# ---------------------------------------------------------------------------
# Quotes (depth snapshots)
# ---------------------------------------------------------------------------

def _good_quotes(n=4):
    stamps = pd.date_range(pd.Timestamp("2024-01-02 09:30", tz=NY), periods=n, freq="min")
    return pd.DataFrame({
        "timestamp": stamps,
        "bid": np.full(n, 18_000.0),
        "ask": np.full(n, 18_000.5),
        "bid_size": np.full(n, 150, dtype=np.int64),
        "ask_size": np.full(n, 120, dtype=np.int64),
    })


def test_clean_quotes_pass():
    df = _good_quotes()
    assert validate_quotes(df) is df


def test_locked_quote_is_allowed_but_crossed_is_not():
    locked = _good_quotes()
    locked["ask"] = locked["bid"]
    validate_quotes(locked)
    crossed = _good_quotes()
    crossed.loc[1, "bid"] = 18_001.0
    with pytest.raises(ContractError, match="crossed quote"):
        validate_quotes(crossed)


def test_decreasing_quote_timestamps_are_rejected():
    df = _good_quotes()
    df["timestamp"] = df["timestamp"].iloc[[0, 2, 1, 3]].to_numpy()
    with pytest.raises(ContractError, match="not non-decreasing"):
        validate_quotes(df)


def test_quote_off_the_grid_and_negative_size_are_rejected():
    off = _good_quotes()
    off.loc[0, "bid"] = 18_000.1
    with pytest.raises(ContractError, match="off the 0.25 tick grid"):
        validate_quotes(off)
    negative = _good_quotes()
    negative.loc[0, "bid_size"] = -1
    with pytest.raises(ContractError, match="negative bid_size"):
        validate_quotes(negative)


def test_quote_outside_globex_is_rejected():
    df = _good_quotes()
    df["timestamp"] = pd.date_range(pd.Timestamp("2024-01-06 10:00", tz=NY), periods=4, freq="min")
    with pytest.raises(ContractError, match="outside the Globex session"):
        validate_quotes(df)


# ---------------------------------------------------------------------------
# Option chain
# ---------------------------------------------------------------------------

def _good_chain():
    return pd.DataFrame({
        "as_of": pd.DatetimeIndex([pd.Timestamp("2024-01-02 10:00", tz=NY)] * 3),
        "expiry": pd.DatetimeIndex(["2024-01-05", "2024-01-05", "2024-01-05"]),
        "strike": np.array([17_900.0, 18_000.0, 18_100.0]),
        "call_open_interest": np.array([100, 200, 300], dtype=np.int64),
        "put_open_interest": np.array([150, 250, 350], dtype=np.int64),
        "implied_vol": np.array([0.2, 0.18, 0.19]),
    })


def test_clean_option_chain_passes():
    df = _good_chain()
    assert validate_option_chain(df) is df


def test_option_chain_rejects_expired_contracts():
    df = _good_chain()
    df["expiry"] = pd.DatetimeIndex(["2024-01-01"] * 3)
    with pytest.raises(ContractError, match="already expired"):
        validate_option_chain(df)


@pytest.mark.parametrize("iv", [0.0, -0.1, 5.5])
def test_option_chain_rejects_implied_vol_outside_its_range(iv):
    df = _good_chain()
    df.loc[1, "implied_vol"] = iv
    with pytest.raises(ContractError, match=r"implied_vol must be in \(0, 5.0\]"):
        validate_option_chain(df)


def test_option_chain_rejects_non_positive_strike():
    df = _good_chain()
    df.loc[0, "strike"] = 0.0
    with pytest.raises(ContractError, match="strike must be finite and positive"):
        validate_option_chain(df)


def test_option_chain_rejects_expiry_with_a_time_of_day():
    df = _good_chain()
    df["expiry"] = pd.DatetimeIndex(["2024-01-05 16:00"] * 3)
    with pytest.raises(ContractError, match="carries a time of day"):
        validate_option_chain(df)


def test_option_chain_rejects_timezone_aware_expiry():
    df = _good_chain()
    df["expiry"] = pd.DatetimeIndex(["2024-01-05"] * 3).tz_localize(NY)
    with pytest.raises(ContractError, match="naive calendar date"):
        validate_option_chain(df)


def test_option_chain_rejects_duplicate_strikes_and_negative_open_interest():
    dup = _good_chain()
    dup.loc[2, "strike"] = 18_000.0
    with pytest.raises(ContractError, match="duplicate"):
        validate_option_chain(dup)
    negative = _good_chain()
    negative.loc[0, "call_open_interest"] = -1
    with pytest.raises(ContractError, match="negative call_open_interest"):
        validate_option_chain(negative)


# ---------------------------------------------------------------------------
# Event calendar
# ---------------------------------------------------------------------------

def _good_events():
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex([
            pd.Timestamp("2024-01-02 10:00", tz=NY),
            pd.Timestamp("2024-01-02 14:00", tz=NY),
        ]),
        "event_type": np.array(["ISM_MFG", "FOMC"], dtype=object),
        "surprise": np.array([0.5, np.nan]),  # NaN marks an unreleased event (assumption 8)
    })


def test_clean_event_calendar_passes_with_a_nullable_surprise():
    df = _good_events()
    assert validate_events(df) is df


def test_event_calendar_rejects_duplicate_pairs_and_decreasing_times():
    dup = _good_events().iloc[[0, 0, 1]].reset_index(drop=True)  # repeats the first pair without breaking time order
    with pytest.raises(ContractError, match="duplicate"):
        validate_events(dup)
    decreasing = _good_events().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ContractError, match="not non-decreasing"):
        validate_events(decreasing)


def test_event_calendar_rejects_naive_timestamps_blank_types_and_non_float_surprise():
    naive = _good_events()
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(ContractError, match="naive"):
        validate_events(naive)
    blank = _good_events()
    blank.loc[0, "event_type"] = "  "
    with pytest.raises(ContractError, match="non-empty string"):
        validate_events(blank)
    as_text = _good_events()
    as_text["surprise"] = as_text["surprise"].astype(object)
    with pytest.raises(ContractError, match="must be float"):
        validate_events(as_text)

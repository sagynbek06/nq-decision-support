"""
Canonical data contracts for the intraday pipeline.

Four tables cross the seam between a market data source and everything
downstream of it: bars, top-of-book depth snapshots (quotes), option chains
and an event calendar. Each table has a validator here. A validator returns its
input unchanged, or raises ContractError naming the rule and the first
offending rows. It never repairs data: a bad row is a finding about the source,
and fixing it silently would hide that finding.

Assumptions (explicit, since a contract is only as good as these choices):

1. Timestamps are timezone-aware in America/New_York. A bar's timestamp is its
   start (left-labelled). Naive timestamps are rejected rather than assumed to
   be New York time.
2. Prices are index points on a 0.25 tick grid (NQ's minimum increment). A price
   is on the grid when it lies within 1e-6 ticks of one. That tolerance absorbs
   float representation error and nothing else.
3. Trading hours are CME Globex NQ. Monday to Thursday trade 18:00 to 17:00 the
   next day, with a daily halt from 17:00 to 18:00. Sunday trading starts at
   18:00. Friday trading ends at 17:00. Saturday is closed. The cash session
   (09:30 to 16:00 ET, Monday to Friday) is a subset of Globex.
4. Exchange holidays are NOT modelled, so a holiday bar inside Globex hours
   passes the session check. That is a known gap, not a claim that the holiday
   traded.
5. A bar must satisfy high >= max(open, close), low <= min(open, close) and
   high >= low. The first two go beyond the minimum, because a bar that breaks
   them is internally inconsistent.
6. Volume is a non-negative whole number of contracts. When the aggressor split
   is present, bid_volume + ask_volume must equal volume exactly. ask_volume is
   buyer-initiated (lifts offers) and bid_volume is seller-initiated (hits bids).
7. Option expiries are naive calendar dates (midnight, no timezone). Implied
   volatility is a fraction (0.15 means 15%) in (0, 5]. An expiry before the
   as_of date is rejected: that contract has already expired.
8. An unreleased event carries surprise NaN. The contract permits that, and does
   not require it.

What this module does NOT claim: that a validated table is true. The validators
check shape, ordering, tick grid, session and internal consistency. Whether the
values match the market is a question no validator here can answer.
"""

import numpy as np
import pandas as pd

TIMEZONE = "America/New_York"
TICK_SIZE = 0.25
TICK_TOLERANCE = 1e-6  # in ticks; see assumption 2
MAX_IMPLIED_VOL = 5.0  # see assumption 7
MAX_REPORTED_ROWS = 5  # offending row indices named in an error message

BAR_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
AGGRESSOR_COLUMNS = ("bid_volume", "ask_volume")
QUOTE_COLUMNS = ("timestamp", "bid", "ask", "bid_size", "ask_size")
OPTION_CHAIN_COLUMNS = ("as_of", "expiry", "strike", "call_open_interest", "put_open_interest", "implied_vol")
EVENT_COLUMNS = ("timestamp", "event_type", "surprise")


class ContractError(ValueError):
    """A table breaks its contract. The message names the table, the rule and the first offending rows."""


def timezone_name(tz):
    """Name of a tzinfo object, for pytz and zoneinfo alike."""
    return getattr(tz, "zone", None) or getattr(tz, "key", None) or str(tz)


def to_new_york(value, name="timestamp"):
    """Return `value` as a timezone-aware New York Timestamp. Raise ValueError if it is naive or in another zone."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware ({TIMEZONE}); got a naive timestamp")
    if timezone_name(ts.tzinfo) != TIMEZONE:
        raise ValueError(f"{name} must be in {TIMEZONE}; got {timezone_name(ts.tzinfo)}")
    return ts


def globex_open(timestamps):
    """Boolean array: True where a New York timestamp falls inside CME Globex NQ trading hours (assumption 3)."""
    ts = pd.DatetimeIndex(timestamps)
    weekday = ts.weekday  # Monday = 0 ... Sunday = 6
    minutes = ts.hour * 60 + ts.minute
    halt = (minutes >= 17 * 60) & (minutes < 18 * 60)
    open_now = (weekday <= 3) & ~halt  # Monday to Thursday, outside the daily halt
    open_now |= (weekday == 4) & (minutes < 17 * 60)  # Friday until 17:00
    open_now |= (weekday == 6) & (minutes >= 18 * 60)  # Sunday from 18:00
    return np.asarray(open_now)


def cash_session(timestamps):
    """Boolean array: True where a New York timestamp falls inside the cash session, 09:30 to 16:00 Monday to Friday."""
    ts = pd.DatetimeIndex(timestamps)
    minutes = ts.hour * 60 + ts.minute
    return np.asarray((ts.weekday <= 4) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60))


def _fail(table, rule, rows=()):
    rows = [int(r) for r in rows]
    message = f"{table}: {rule}"
    if rows:
        shown = ", ".join(str(r) for r in rows[:MAX_REPORTED_ROWS])
        extra = f" (+{len(rows) - MAX_REPORTED_ROWS} more)" if len(rows) > MAX_REPORTED_ROWS else ""
        message += f"; offending row index: {shown}{extra}"
    raise ContractError(message)


def _require_columns(df, columns, table):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ContractError(f"{table}: missing required column(s) {missing}")


def _require_new_york_timestamps(series, table, column):
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ContractError(f"{table}: column {column!r} must be a datetime column, got {series.dtype}")
    tz = series.dt.tz
    if tz is None:
        raise ContractError(f"{table}: timestamps are naive; localize them to {TIMEZONE} before validating")
    if timezone_name(tz) != TIMEZONE:
        raise ContractError(f"{table}: timestamps are in {timezone_name(tz)}; the contract requires {TIMEZONE}")


def _check_no_missing(df, columns, table):
    missing = df[list(columns)].isna().any(axis=1).to_numpy()
    if missing.any():
        _fail(table, "missing values", np.flatnonzero(missing))


def _check_float(series, table, column):
    if not pd.api.types.is_float_dtype(series):
        raise ContractError(f"{table}: column {column!r} must be float, got {series.dtype}")


def _check_integer(series, table, column):
    if not pd.api.types.is_integer_dtype(series):
        raise ContractError(f"{table}: column {column!r} must be whole numbers (integer dtype), got {series.dtype}")


def _check_strictly_increasing(timestamps, table):
    decreasing = (timestamps.diff() < pd.Timedelta(0)).to_numpy()
    if decreasing.any():
        _fail(table, "timestamps are not monotonic increasing", np.flatnonzero(decreasing))
    duplicated = timestamps.duplicated().to_numpy()
    if duplicated.any():
        _fail(table, "duplicate timestamps", np.flatnonzero(duplicated))


def _off_grid(values):
    ticks = np.asarray(values, dtype=float) / TICK_SIZE
    return np.abs(ticks - np.round(ticks)) > TICK_TOLERANCE


def validate_bars(df, *, require_aggressor=False):
    """
    Validate a bars table. Returns `df` unchanged, or raises ContractError.

    Rejects: missing columns, naive or non-New-York timestamps, missing values,
    non-monotonic or duplicate timestamps, bars outside the Globex session,
    prices off the 0.25 tick grid, high < low, high below open or close, low
    above open or close, negative volume, and an aggressor split that does not
    sum to volume. The aggressor columns are checked whenever they are present
    and are required when `require_aggressor` is True.
    """
    table = "bars"
    columns = BAR_COLUMNS + (AGGRESSOR_COLUMNS if require_aggressor else ())
    _require_columns(df, columns, table)
    _require_new_york_timestamps(df["timestamp"], table, "timestamp")
    _check_no_missing(df, BAR_COLUMNS, table)
    for column in PRICE_COLUMNS:
        _check_float(df[column], table, column)
    _check_integer(df["volume"], table, "volume")
    _check_strictly_increasing(df["timestamp"], table)

    closed = ~globex_open(df["timestamp"])
    if closed.any():
        _fail(table, "bars outside the Globex session (weekend, the 17:00-18:00 halt, "
                     "or after Friday 17:00 / before Sunday 18:00)", np.flatnonzero(closed))
    for column in PRICE_COLUMNS:
        off = _off_grid(df[column])
        if off.any():
            _fail(table, f"{column} is off the {TICK_SIZE} tick grid", np.flatnonzero(off))

    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    bad = high < low
    if bad.any():
        _fail(table, "high < low", np.flatnonzero(bad))
    bad = high < np.maximum(open_, close)
    if bad.any():
        _fail(table, "high is below open or close", np.flatnonzero(bad))
    bad = low > np.minimum(open_, close)
    if bad.any():
        _fail(table, "low is above open or close", np.flatnonzero(bad))

    negative = (df["volume"] < 0).to_numpy()
    if negative.any():
        _fail(table, "negative volume", np.flatnonzero(negative))
    if all(c in df.columns for c in AGGRESSOR_COLUMNS):
        _check_aggressor_split(df, table)
    return df


def _check_aggressor_split(df, table):
    for column in AGGRESSOR_COLUMNS:
        _check_integer(df[column], table, column)
        negative = (df[column] < 0).to_numpy()
        if negative.any():
            _fail(table, f"negative {column}", np.flatnonzero(negative))
    mismatch = (df["bid_volume"] + df["ask_volume"] != df["volume"]).to_numpy()
    if mismatch.any():
        _fail(table, "bid_volume + ask_volume does not equal volume", np.flatnonzero(mismatch))


def validate_quotes(df):
    """
    Validate a depth (top-of-book) snapshot table. Returns `df`, or raises ContractError.

    Rejects: missing columns, naive or non-New-York timestamps, missing values,
    decreasing timestamps (equal timestamps are allowed), snapshots outside the
    Globex session, bid or ask off the tick grid, a crossed quote (bid above
    ask; a locked quote with bid equal to ask is allowed), and sizes that are
    negative or not whole numbers.
    """
    table = "quotes"
    _require_columns(df, QUOTE_COLUMNS, table)
    _require_new_york_timestamps(df["timestamp"], table, "timestamp")
    _check_no_missing(df, QUOTE_COLUMNS, table)
    decreasing = (df["timestamp"].diff() < pd.Timedelta(0)).to_numpy()
    if decreasing.any():
        _fail(table, "timestamps are not non-decreasing", np.flatnonzero(decreasing))
    closed = ~globex_open(df["timestamp"])
    if closed.any():
        _fail(table, "snapshots outside the Globex session", np.flatnonzero(closed))
    for column in ("bid", "ask"):
        _check_float(df[column], table, column)
        off = _off_grid(df[column])
        if off.any():
            _fail(table, f"{column} is off the {TICK_SIZE} tick grid", np.flatnonzero(off))
    crossed = (df["bid"] > df["ask"]).to_numpy()
    if crossed.any():
        _fail(table, "crossed quote (bid above ask)", np.flatnonzero(crossed))
    for column in ("bid_size", "ask_size"):
        _check_integer(df[column], table, column)
        negative = (df[column] < 0).to_numpy()
        if negative.any():
            _fail(table, f"negative {column}", np.flatnonzero(negative))
    return df


def validate_option_chain(df):
    """
    Validate an option chain table. Returns `df`, or raises ContractError.

    Rejects: missing columns, naive or non-New-York as_of, missing values, an
    expiry that is timezone-aware or carries a time of day, non-positive or
    non-finite strikes, open interest that is negative or not whole, implied
    volatility outside (0, 5], duplicate (as_of, expiry, strike) rows, and
    expiries before the as_of date.
    """
    table = "option chain"
    _require_columns(df, OPTION_CHAIN_COLUMNS, table)
    _require_new_york_timestamps(df["as_of"], table, "as_of")
    _check_no_missing(df, OPTION_CHAIN_COLUMNS, table)

    expiry = df["expiry"]
    if not pd.api.types.is_datetime64_any_dtype(expiry) or expiry.dt.tz is not None:
        raise ContractError(f"{table}: expiry must be a naive calendar date (datetime64 without a timezone)")
    with_time = (expiry != expiry.dt.normalize()).to_numpy()
    if with_time.any():
        _fail(table, "expiry carries a time of day; expiries are calendar dates", np.flatnonzero(with_time))

    strike = df["strike"].to_numpy()
    _check_float(df["strike"], table, "strike")
    bad = ~(np.isfinite(strike) & (strike > 0))
    if bad.any():
        _fail(table, "strike must be finite and positive", np.flatnonzero(bad))
    for column in ("call_open_interest", "put_open_interest"):
        _check_integer(df[column], table, column)
        negative = (df[column] < 0).to_numpy()
        if negative.any():
            _fail(table, f"negative {column}", np.flatnonzero(negative))
    _check_float(df["implied_vol"], table, "implied_vol")
    iv = df["implied_vol"].to_numpy()
    bad = ~(np.isfinite(iv) & (iv > 0) & (iv <= MAX_IMPLIED_VOL))
    if bad.any():
        _fail(table, f"implied_vol must be in (0, {MAX_IMPLIED_VOL}]", np.flatnonzero(bad))

    duplicated = df.duplicated(subset=["as_of", "expiry", "strike"]).to_numpy()
    if duplicated.any():
        _fail(table, "duplicate (as_of, expiry, strike) rows", np.flatnonzero(duplicated))
    as_of_date = df["as_of"].dt.tz_localize(None).dt.normalize()
    expired = (expiry < as_of_date).to_numpy()
    if expired.any():
        _fail(table, "expiry before the as_of date (contract already expired)", np.flatnonzero(expired))
    return df


def validate_events(df):
    """
    Validate an event calendar. Returns `df`, or raises ContractError.

    Rejects: missing columns, naive or non-New-York timestamps, a missing
    timestamp, a blank event_type, a surprise column that is not float, timestamps
    that decrease, and duplicate (timestamp, event_type) pairs. A NaN surprise is
    allowed: it marks an event that has not been released yet.
    """
    table = "event calendar"
    _require_columns(df, EVENT_COLUMNS, table)
    _require_new_york_timestamps(df["timestamp"], table, "timestamp")
    missing_ts = df["timestamp"].isna().to_numpy()
    if missing_ts.any():
        _fail(table, "missing timestamp", np.flatnonzero(missing_ts))
    types = df["event_type"]
    blank = (types.isna() | (types.astype(str).str.strip() == "")).to_numpy()
    if blank.any():
        _fail(table, "event_type must be a non-empty string", np.flatnonzero(blank))
    _check_float(df["surprise"], table, "surprise")
    decreasing = (df["timestamp"].diff() < pd.Timedelta(0)).to_numpy()
    if decreasing.any():
        _fail(table, "timestamps are not non-decreasing", np.flatnonzero(decreasing))
    duplicated = df.duplicated(subset=["timestamp", "event_type"]).to_numpy()
    if duplicated.any():
        _fail(table, "duplicate (timestamp, event_type) pairs", np.flatnonzero(duplicated))
    return df

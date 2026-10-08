"""
Synthetic 1-minute NQ-like bars behind the MarketDataSource seam.

Every number here is a documented constant. None is calibrated to real NQ data.
The module exists to exercise plumbing and to give tests structure that was put
in on purpose. It does not produce alpha, and no result from it is evidence of a
market edge. See docs/writeups/07_intraday_data_layer.md for what it can and
cannot validate.

Assumptions (explicit, since none of this is measured):

1. Only cash-session bars are generated: 09:30 to 15:59 New York time, 390
   one-minute bars per session, Monday to Friday. There are no overnight or
   Globex-only bars and no overnight gap: each session opens at the previous
   session's last close.
2. Regimes come from src.synthetic_data.generate_regime_switching_returns. Its
   daily regime path, annual drift and annual volatility are reused and scaled
   to minutes. A minute's drift is the daily drift divided by 390. A minute's
   volatility is the daily volatility divided by sqrt(390), times the intraday
   profile in assumption 3.
3. Intraday volatility is U-shaped, high at the open and the close and low at
   midday. The profile is f(m) = 1 + OPEN_VOL_BUMP * exp(-m / DECAY) +
   CLOSE_VOL_BUMP * exp(-(390 - m) / DECAY), with m the minute index. It is
   normalised so the mean of f^2 over a session is 1, which keeps each day's
   total variance equal to its regime's daily variance.
4. Innovations are Student-t with 5 degrees of freedom, scaled to unit variance.
   That is the same tail assumption as the daily generator.
5. Volume is U-shaped as well, and correlated with volatility after Clark (1973).
   Expected volume scales with the minute's volatility relative to the session
   average (exponent VOLUME_VOL_LINK), and rises with the size of the
   contemporaneous normalised return (VOLUME_SHOCK_LINK). Volume is lognormally
   overdispersed around that mean and rounded to whole contracts.
6. The aggressor split is correlated with the sign and size of the bar's return:
   buy share = 0.5 + AGGRESSOR_RETURN_LINK * tanh(return / volatility). ask_volume
   is buyer-initiated and bid_volume is seller-initiated, and they sum to volume
   exactly. This is the same mechanical link the daily order-flow module uses
   (Kyle 1985). It is not a claim about any real day.
7. Scheduled events. ISM_MFG is released at 10:00 ET on the first session of each
   month. FOMC is released at 14:00 ET every FOMC_EVERY_SESSIONS sessions, starting
   at FOMC_FIRST_SESSION. Each surprise is standard normal. At the release minute,
   volatility is multiplied by (1 + EVENT_VOL_LINK * |surprise|), and the return
   gets a directional jump of EVENT_JUMP * surprise in units of that minute's
   volatility. Surprises are always populated: a synthetic release has already
   happened by the time anyone can query it.
8. Depth is one top-of-book snapshot per bar, with bid and ask SPREAD_TICKS / 2
   ticks each side of the bar close. Resting size is lognormal around
   DEPTH_MEAN_SIZE and shrinks when volatility is high (DEPTH_VOL_LINK), as the
   daily order-flow module assumes.
9. Planted edge. A hidden latent order pressure z is an AR(1) with unit variance
   and persistence PRESSURE_PERSISTENCE. The normalised return of minute t is
   eps_t + planted_edge * z_(t-1). So planted_edge is the slope of the next
   minute's normalised return on the current hidden pressure. With planted_edge =
   0.0 the pressure never reaches the data. z is hidden: it is exposed only
   through hidden_state_for_validation(), which only validation code may call.
   Every planted_edge uses the same draws, so two sources that differ only in
   planted_edge differ by exactly planted_edge * z in the normalised return. Through
   the contemporaneous return, a nonzero planted_edge also reaches volume and the
   aggressor split. That is intended: order flow reacts to returns.
10. Option chain. Weekly Friday expiries (the next CHAIN_EXPIRIES), with strikes
    every CHAIN_STRIKE_STEP points, CHAIN_HALF_WIDTH strikes each side of spot.
    Implied vol follows the quadratic skew in log-moneyness used in
    src/greeks_dashboard.py, with this module's own constants. Open interest is a
    deterministic smooth profile with no noise, and puts carry a fixed tilt. The
    chain has no term structure and is not priced from the bars.
11. All randomness comes from one seed. Draws happen session by session, in a
    fixed order, so the first K sessions are identical whatever the horizon. Tests
    pin both determinism and this prefix consistency.

What this source can validate: the plumbing (session logic, the contract,
windowing, resampling, causal option-chain snapshots), and whether an analysis
recovers structure that was put in on purpose (the U-shape, the planted edge,
volatility clustering). What it cannot validate: anything about real markets,
including volatility or volume levels, event response, microstructure, the size
or existence of a real edge, or the realism of the option chain.
"""

import math

import numpy as np
import pandas as pd

from src.data.schema import (
    TICK_SIZE,
    TIMEZONE,
    to_new_york,
    validate_bars,
    validate_events,
    validate_option_chain,
    validate_quotes,
)
from src.data.source import MarketDataSource, check_frequency, check_window
from src.synthetic_data import DEFAULT_REGIMES, TRADING_DAYS_PER_YEAR, generate_regime_switching_returns

# --- session geometry and starting point (assumption 1) ---
SESSION_MINUTES = 390
DEFAULT_START_DATE = "2024-01-02"
DEFAULT_N_SESSIONS = 60
DEFAULT_SEED = 18
START_PRICE = 18_000.0

# --- tails (assumption 4) ---
T_DOF = 5

# --- intraday profiles (assumptions 3 and 5) ---
PROFILE_DECAY_MINUTES = 40.0
OPEN_VOL_BUMP = 1.0
CLOSE_VOL_BUMP = 0.5
OPEN_VOLUME_BUMP = 0.8
CLOSE_VOLUME_BUMP = 0.4

# --- volume and the Clark (1973) link (assumption 5) ---
DAILY_VOLUME = 200_000
VOLUME_VOL_LINK = 1.0
VOLUME_SHOCK_LINK = 0.3
VOLUME_OVERDISPERSION = 0.3

# --- aggressor split (assumption 6) ---
AGGRESSOR_RETURN_LINK = 0.20

# --- intra-bar range, which sets high and low (documented here, not in the assumptions list) ---
RANGE_SCALE = 0.5  # extension beyond open/close, in units of the minute's volatility times the price

# --- scheduled events (assumption 7) ---
EVENT_VOL_LINK = 1.0
EVENT_JUMP = 1.0
ISM_EVENT_TYPE = "ISM_MFG"
ISM_MINUTE = 30  # 10:00 ET
FOMC_EVENT_TYPE = "FOMC"
FOMC_EVERY_SESSIONS = 31  # about eight a year
FOMC_FIRST_SESSION = 20
FOMC_MINUTE = 270  # 14:00 ET

# --- depth (assumption 8) ---
SPREAD_TICKS = 2
DEPTH_MEAN_SIZE = 150.0
DEPTH_SIZE_SIGMA = 0.4
DEPTH_VOL_LINK = 0.5

# --- planted edge (assumption 9) ---
PRESSURE_PERSISTENCE = 0.95

# --- option chain (assumption 10) ---
CHAIN_ATM_VOL = 0.18
CHAIN_SKEW_SLOPE = -0.10
CHAIN_SKEW_CURVATURE = 0.20
CHAIN_STRIKE_STEP = 100.0
CHAIN_HALF_WIDTH = 10
CHAIN_EXPIRIES = 4
CHAIN_OI_BASE = 20_000
CHAIN_OI_SCALE = 0.02
CHAIN_PUT_OI_TILT = 1.2
CHAIN_MIN_IV = 0.03
CHAIN_MAX_IV = 5.0


def intraday_volatility_profile():
    """Minute volatility multiplier over one session, normalised so that mean(profile**2) == 1 (assumption 3)."""
    minutes = np.arange(SESSION_MINUTES, dtype=float)
    raw = (1.0
           + OPEN_VOL_BUMP * np.exp(-minutes / PROFILE_DECAY_MINUTES)
           + CLOSE_VOL_BUMP * np.exp(-(SESSION_MINUTES - minutes) / PROFILE_DECAY_MINUTES))
    return raw / np.sqrt(np.mean(raw ** 2))


def intraday_volume_profile():
    """Expected minute volume multiplier over one session, normalised to mean 1 (assumption 5)."""
    minutes = np.arange(SESSION_MINUTES, dtype=float)
    raw = (1.0
           + OPEN_VOLUME_BUMP * np.exp(-minutes / PROFILE_DECAY_MINUTES)
           + CLOSE_VOLUME_BUMP * np.exp(-(SESSION_MINUTES - minutes) / PROFILE_DECAY_MINUTES))
    return raw / raw.mean()


def _round_to_tick(values):
    return np.round(values / TICK_SIZE) * TICK_SIZE


def _ceil_to_tick(values):
    return np.ceil(values / TICK_SIZE - 1e-9) * TICK_SIZE


def _floor_to_tick(values):
    return np.floor(values / TICK_SIZE + 1e-9) * TICK_SIZE


def _session_timestamps(session_dates):
    """All minute timestamps for the given session dates, each session starting at 09:30 New York time."""
    chunks = []
    for day in session_dates:
        open_ts = pd.Timestamp(f"{day:%Y-%m-%d} 09:30").tz_localize(TIMEZONE)
        chunks.append(pd.date_range(start=open_ts, periods=SESSION_MINUTES, freq="min"))
    return chunks[0].append(chunks[1:])


def _first_session_of_month(session_dates):
    months = session_dates.to_period("M")
    return np.r_[True, months[1:] != months[:-1]]


def _next_fridays(as_of, count):
    """The next `count` Fridays strictly after the as_of date, as naive calendar dates."""
    first_day = as_of.tz_localize(None).normalize() + pd.Timedelta(days=1)
    candidates = pd.date_range(start=first_day, periods=7 * count, freq="D")
    return candidates[candidates.weekday == 4][:count]


class SyntheticIntradaySource(MarketDataSource):
    """
    Synthetic 1-minute bars, depth, option chain and events, behind the MarketDataSource interface.

    Construction draws and builds the whole horizon at once. Every table then passes its validator
    before the constructor returns, so a generator bug fails at construction rather than downstream.
    The planted_edge parameter (default 0.0) is described in assumption 9.
    """

    def __init__(self, n_sessions=DEFAULT_N_SESSIONS, start_date=DEFAULT_START_DATE,
                 seed=DEFAULT_SEED, planted_edge=0.0):
        if int(n_sessions) < 1:
            raise ValueError("n_sessions must be at least 1")
        self.n_sessions = int(n_sessions)
        self.start_date = start_date
        self.seed = seed
        self.planted_edge = float(planted_edge)
        self._build()

    def _build(self):
        k_sessions = self.n_sessions
        regime_seq, minute_seq = np.random.SeedSequence(self.seed).spawn(2)
        regime_rng = np.random.default_rng(regime_seq)
        minute_rng = np.random.default_rng(minute_seq)

        dates = pd.bdate_range(start=self.start_date, periods=k_sessions)
        _, regime_path = generate_regime_switching_returns(n_days=k_sessions, rng=regime_rng)

        # Every random number is drawn here, session by session, in a fixed order (assumption 11).
        # Nothing below draws again, and no draw depends on planted_edge.
        shape = (k_sessions, SESSION_MINUTES)
        eps = np.empty(shape)
        upsilon = np.empty(shape)
        eta_volume = np.empty(shape)
        xi_split = np.empty(shape)
        xi_high = np.empty(shape)
        xi_low = np.empty(shape)
        eta_bid = np.empty(shape)
        eta_ask = np.empty(shape)
        surprise_ism = np.empty(k_sessions)
        surprise_fomc = np.empty(k_sessions)
        t_scale = math.sqrt((T_DOF - 2) / T_DOF)
        for k in range(k_sessions):
            eps[k] = minute_rng.standard_t(T_DOF, size=SESSION_MINUTES) * t_scale
            upsilon[k] = minute_rng.standard_normal(SESSION_MINUTES)
            eta_volume[k] = minute_rng.standard_normal(SESSION_MINUTES)
            xi_split[k] = minute_rng.standard_normal(SESSION_MINUTES)
            xi_high[k] = minute_rng.standard_normal(SESSION_MINUTES)
            xi_low[k] = minute_rng.standard_normal(SESSION_MINUTES)
            eta_bid[k] = minute_rng.standard_normal(SESSION_MINUTES)
            eta_ask[k] = minute_rng.standard_normal(SESSION_MINUTES)
            surprise_ism[k] = minute_rng.standard_normal()
            surprise_fomc[k] = minute_rng.standard_normal()

        # Regime-scaled drift and volatility (assumption 2), with the U-shape (assumption 3).
        params = [DEFAULT_REGIMES[name] for name in regime_path]
        annual_vol = np.array([p["annual_vol"] for p in params])
        annual_drift = np.array([p["annual_drift"] for p in params])
        session_sigma = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR) / np.sqrt(SESSION_MINUTES)
        session_drift = annual_drift / TRADING_DAYS_PER_YEAR / SESSION_MINUTES
        sigma = session_sigma[:, None] * intraday_volatility_profile()[None, :]
        drift = np.broadcast_to(session_drift[:, None], shape)

        # Scheduled events (assumption 7): volatility multiplier and surprise per release minute.
        event_mult = np.ones(shape)
        event_surprise = np.zeros(shape)
        event_rows = []  # (session index, minute, event type, surprise), in time order
        first_of_month = _first_session_of_month(dates)
        for k in range(k_sessions):
            if first_of_month[k]:
                event_rows.append((k, ISM_MINUTE, ISM_EVENT_TYPE, float(surprise_ism[k])))
            if k >= FOMC_FIRST_SESSION and (k - FOMC_FIRST_SESSION) % FOMC_EVERY_SESSIONS == 0:
                event_rows.append((k, FOMC_MINUTE, FOMC_EVENT_TYPE, float(surprise_fomc[k])))
        for k, minute, _, surprise in event_rows:
            event_mult[k, minute] = 1.0 + EVENT_VOL_LINK * abs(surprise)
            event_surprise[k, minute] = surprise

        sigma_eff = (sigma * event_mult).ravel()
        session_sigma_flat = np.repeat(session_sigma, SESSION_MINUTES)
        drift_flat = drift.ravel()
        jump = EVENT_JUMP * event_surprise.ravel() * sigma_eff

        # Hidden order pressure (assumption 9): AR(1) over the flattened minute sequence.
        n_minutes = k_sessions * SESSION_MINUTES
        upsilon_flat = upsilon.ravel()
        pressure = np.empty(n_minutes)
        pressure[0] = upsilon_flat[0]
        innovation_sd = math.sqrt(1.0 - PRESSURE_PERSISTENCE ** 2)
        for n in range(1, n_minutes):
            pressure[n] = PRESSURE_PERSISTENCE * pressure[n - 1] + innovation_sd * upsilon_flat[n]
        pressure_lag = np.r_[0.0, pressure[:-1]]

        # Normalised return and the return itself (assumptions 2, 4 and 9).
        normalized = eps.ravel() + self.planted_edge * pressure_lag
        ret = drift_flat + sigma_eff * normalized + jump

        # Prices on the tick grid. Each close rounds the unrounded path, so rounding does not compound.
        raw_close = START_PRICE * np.exp(np.cumsum(ret))
        close = _round_to_tick(raw_close)
        open_ = np.r_[START_PRICE, close[:-1]]
        range_up = RANGE_SCALE * sigma_eff * open_ * np.abs(xi_high.ravel())
        range_down = RANGE_SCALE * sigma_eff * open_ * np.abs(xi_low.ravel())
        high = _ceil_to_tick(np.maximum(open_, close) + range_up)
        low = _floor_to_tick(np.minimum(open_, close) - range_down)

        # Volume (assumption 5): U-shaped, Clark-linked to volatility, overdispersed, whole contracts.
        clark = sigma_eff / session_sigma_flat
        expected_volume = ((DAILY_VOLUME / SESSION_MINUTES)
                           * np.tile(intraday_volume_profile(), k_sessions)
                           * clark ** VOLUME_VOL_LINK
                           * (1.0 + VOLUME_SHOCK_LINK * np.abs(normalized)))
        volume_float = expected_volume * np.exp(
            VOLUME_OVERDISPERSION * eta_volume.ravel() - 0.5 * VOLUME_OVERDISPERSION ** 2)
        volume = np.round(volume_float).astype(np.int64)

        # Aggressor split (assumption 6): buy share follows the sign and size of the standardised return.
        standardized = ret / sigma_eff
        buy_share = 0.5 + AGGRESSOR_RETURN_LINK * np.tanh(standardized)
        ask_float = (volume * buy_share
                     + np.sqrt(volume * buy_share * (1.0 - buy_share)) * xi_split.ravel())
        ask_volume = np.clip(np.round(ask_float), 0, volume).astype(np.int64)
        bid_volume = volume - ask_volume

        # Depth (assumption 8): top of book two ticks around the close; size shrinks when volatility is high.
        half_spread = SPREAD_TICKS * TICK_SIZE / 2.0
        size_scale = (session_sigma_flat / sigma_eff) ** DEPTH_VOL_LINK
        size_noise_var = -0.5 * DEPTH_SIZE_SIGMA ** 2
        bid_size = np.round(DEPTH_MEAN_SIZE * size_scale
                            * np.exp(DEPTH_SIZE_SIGMA * eta_bid.ravel() + size_noise_var)).astype(np.int64)
        ask_size = np.round(DEPTH_MEAN_SIZE * size_scale
                            * np.exp(DEPTH_SIZE_SIGMA * eta_ask.ravel() + size_noise_var)).astype(np.int64)

        timestamps = _session_timestamps(dates)
        event_index = np.array([k * SESSION_MINUTES + minute for k, minute, _, _ in event_rows], dtype=np.int64)

        self._session_dates = dates
        self._timestamps = pd.DatetimeIndex(timestamps)
        self._bars = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
        })
        self._depth = pd.DataFrame({
            "timestamp": timestamps,
            "bid": close - half_spread,
            "ask": close + half_spread,
            "bid_size": bid_size,
            "ask_size": ask_size,
        })
        self._events = pd.DataFrame({
            "timestamp": timestamps[event_index],
            "event_type": np.array([row[2] for row in event_rows], dtype=object),
            "surprise": np.array([row[3] for row in event_rows], dtype=float),
        })
        self._hidden = pd.DataFrame({
            "timestamp": timestamps,
            "pressure": pressure,
            "normalized_residual": normalized,
            "sigma": sigma_eff,
            "drift": drift_flat,
            "jump": jump,
        })

        validate_bars(self._bars, require_aggressor=True)
        validate_quotes(self._depth)
        validate_events(self._events)

    # ------------------------------------------------------------------
    # MarketDataSource interface
    # ------------------------------------------------------------------

    def bars(self, start, end, freq="1min"):
        start, end = check_window(start, end)
        check_frequency(freq)
        stamps = self._bars["timestamp"]
        window = self._bars[(stamps >= start) & (stamps < end)]
        if freq != "1min" and not window.empty:
            window = (window.set_index("timestamp")
                      .resample(freq, label="left", closed="left")
                      .agg(open=("open", "first"),
                           high=("high", "max"),
                           low=("low", "min"),
                           close=("close", "last"),
                           volume=("volume", "sum"),
                           bid_volume=("bid_volume", "sum"),
                           ask_volume=("ask_volume", "sum"))
                      .dropna(subset=["open"])
                      .reset_index())
        return validate_bars(window.reset_index(drop=True), require_aggressor=True)

    def depth(self, start, end):
        start, end = check_window(start, end)
        stamps = self._depth["timestamp"]
        window = self._depth[(stamps >= start) & (stamps < end)]
        return validate_quotes(window.reset_index(drop=True))

    def option_chain(self, as_of):
        as_of = to_new_york(as_of, "as_of")
        position = int(self._timestamps.searchsorted(as_of, side="right")) - 1
        if position < 0:
            raise ValueError(f"as_of {as_of} is before the first generated bar")
        snapshot = self._timestamps[position]
        spot = float(self._bars["close"].iloc[position])

        expiries = _next_fridays(snapshot, CHAIN_EXPIRIES)
        center = round(spot / CHAIN_STRIKE_STEP) * CHAIN_STRIKE_STEP
        strikes = center + np.arange(-CHAIN_HALF_WIDTH, CHAIN_HALF_WIDTH + 1) * CHAIN_STRIKE_STEP
        strikes = strikes[strikes > 0].astype(float)
        log_moneyness = np.log(strikes / spot)
        iv = np.clip(CHAIN_ATM_VOL + CHAIN_SKEW_SLOPE * log_moneyness
                     + CHAIN_SKEW_CURVATURE * log_moneyness ** 2, CHAIN_MIN_IV, CHAIN_MAX_IV)
        shape = np.exp(-0.5 * (log_moneyness / CHAIN_OI_SCALE) ** 2)
        call_oi = np.round(CHAIN_OI_BASE * shape).astype(np.int64)
        put_oi = np.round(CHAIN_OI_BASE * shape * CHAIN_PUT_OI_TILT).astype(np.int64)

        n_expiries, n_strikes = len(expiries), len(strikes)
        chain = pd.DataFrame({
            "as_of": pd.DatetimeIndex([snapshot] * (n_expiries * n_strikes)),
            "expiry": pd.DatetimeIndex(expiries).repeat(n_strikes),
            "strike": np.tile(strikes, n_expiries),
            "call_open_interest": np.tile(call_oi, n_expiries),
            "put_open_interest": np.tile(put_oi, n_expiries),
            "implied_vol": np.tile(iv, n_expiries),
        })
        return validate_option_chain(chain)

    def events(self, start, end):
        start, end = check_window(start, end)
        stamps = self._events["timestamp"]
        window = self._events[(stamps >= start) & (stamps < end)]
        return validate_events(window.reset_index(drop=True))

    # ------------------------------------------------------------------
    # Validation only
    # ------------------------------------------------------------------

    def hidden_state_for_validation(self):
        """
        Hidden quantities behind the generated bars: the latent order pressure, the normalised residual,
        the effective minute volatility, the drift and the event jump, one row per bar.

        Validation and tests only. Pipeline code must not call this, because it reads information that
        a real source would not expose (assumption 9).
        """
        return self._hidden.copy()

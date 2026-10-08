"""
The only seam between market data and the rest of the system.

Everything downstream of this module (signals, consensus, backtests) receives
data through a MarketDataSource and through nothing else. Downstream code does
not import a vendor library, does not read a file, and does not know whether
the numbers are synthetic. Moving from a synthetic source to a real one is a
constructor change at the call site, not an edit to downstream code.

Assumptions (explicit):

1. Every method returns a table that passes its validator in schema.py. An
   implementation that cannot meet the contract must raise. It must not return
   a partial table, and it must not repair one.
2. Windows are half-open: start is inclusive and end is exclusive. Both bounds
   must be timezone-aware in America/New_York. A naive bound, or a start that is
   not before end, raises ValueError. That is a bad request, not a contract
   violation.
3. bars() supports the frequencies in SUPPORTED_FREQUENCIES. Other frequencies
   raise, rather than resampling to a grid that does not line up with the
   09:30 session open.
4. option_chain(as_of) returns the chain as it stood at the last bar at or before
   as_of. An implementation must never return anything dated after as_of.
5. events(start, end) returns releases whose timestamp falls in the window. A
   surprise may be NaN for a release that has not happened yet.

What this module does NOT claim: that any implementation is faithful to a
venue. A source is a contract. Whether its data matches the market is an
empirical question that each adapter has to answer for itself.
"""

from abc import ABC, abstractmethod

from src.data.schema import to_new_york

SUPPORTED_FREQUENCIES = ("1min", "5min", "15min")  # see assumption 3


def check_window(start, end):
    """Validate a half-open [start, end) window and return both bounds as New York Timestamps."""
    start_ts = to_new_york(start, "start")
    end_ts = to_new_york(end, "end")
    if start_ts >= end_ts:
        raise ValueError(f"start ({start_ts}) must be before end ({end_ts})")
    return start_ts, end_ts


def check_frequency(freq):
    """Raise ValueError unless `freq` is one of SUPPORTED_FREQUENCIES."""
    if freq not in SUPPORTED_FREQUENCIES:
        raise ValueError(f"unsupported frequency {freq!r}; supported: {SUPPORTED_FREQUENCIES}")


class MarketDataSource(ABC):
    """
    Abstract source of market data. See the module docstring for the rules every method follows.

    Downstream code must depend on this class only. A concrete source is the one
    place that may talk to a vendor, a database or a file.
    """

    @abstractmethod
    def bars(self, start, end, freq="1min"):
        """
        Bars with timestamp in [start, end), aggregated to `freq`.

        Returns a DataFrame that passes validate_bars(df, require_aggressor=True).
        """

    @abstractmethod
    def depth(self, start, end):
        """
        Top-of-book snapshots with timestamp in [start, end).

        Returns a DataFrame that passes validate_quotes.
        """

    @abstractmethod
    def option_chain(self, as_of):
        """
        The option chain as it stood at the last bar at or before `as_of`.

        Returns a DataFrame that passes validate_option_chain. Never returns data dated after as_of.
        """

    @abstractmethod
    def events(self, start, end):
        """
        Calendar releases with timestamp in [start, end).

        Returns a DataFrame that passes validate_events.
        """

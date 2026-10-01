# Order Flow Monitor: Volume Delta, OBI, and Marchenko-Pastur Filtering

**Program 2 of the NQ decision-support system.** This writeup covers the
Volume Delta / Order Book Imbalance approach, the synthetic order-flow data
this runs on until real data is integrated, how Marchenko-Pastur (MP)
eigenvalue filtering was adapted to a single instrument, and the
quantitative validation behind that adaptation.

Code: [`src/order_flow.py`](../../src/order_flow.py). Tests:
[`tests/test_order_flow.py`](../../tests/test_order_flow.py).

## 1. Volume Delta and Order Book Imbalance

Two standard order-flow signals, computed directly:

- **Volume Delta** (`compute_volume_delta`): buy-initiated volume minus
  sell-initiated volume for the day. Positive means net aggressive buying.
- **Order Book Imbalance** (`compute_order_book_imbalance`): `(bid - ask) /
  (bid + ask)`, bounded in `[-1, 1]`. Positive means more resting size on
  the bid than the ask.

These measure different things — Volume Delta is about *executed trades*
(who was aggressive), OBI is about *resting liquidity* (who's posted size,
not necessarily traded) — which is why both are computed rather than
picking one.

## 2. Synthetic data: a documented limitation until Phase 6

There is no real order book or trade-tape data yet (`ROADMAP.md` Phase 6:
Bloomberg/Rithmic integration). `generate_synthetic_order_flow` produces a
placeholder bid/ask/volume dataset correlated with the Phase 1 price
series, under four explicit assumptions:

1. **Total traded volume scales with realized volatility** (`|log_return|`
   as a same-day proxy) — the well-documented volume/volatility
   relationship (Clark 1973's "mixture of distributions hypothesis").
2. **The aggressor split (buy vs. sell volume) is correlated with the sign
   and magnitude of the day's return** — mirrors the mechanical link
   between net order flow and price change in real markets (Kyle 1985),
   not a claim about actual historical order flow.
3. **Resting bid/ask depth shrinks on higher-volatility days** — liquidity
   providers reduce quoted size when short-term risk is elevated.
4. **Depth *imbalance* (as opposed to depth level) is modeled as mostly
   independent noise** with only a mild return link, since there's no
   principled way to simulate genuine limit-order-book dynamics (queue
   position, cancellations, iceberg orders) without real data.

None of the four coefficients controlling these effects are calibrated to
real NQ order flow. The first tuning pass made Volume Delta correlate with
same-day return at **0.93–0.95** — found by explicitly testing for it, not
by inspection — which would have made it a near-relabeling of price rather
than a complementary signal. The root cause was structural, not just a
loose coefficient: total volume and the aggressor split both scale with
the same underlying return, so the two effects reinforce each other
multiplicatively, and a few extreme-return days (this data has fat tails)
dominate the resulting Pearson correlation regardless of how weakly either
individual coefficient is set. The parameters were reworked to `corr ≈
0.52` (`ORDER_FLOW_RETURN_SENSITIVITY = 0.08`), in the same range as OBI's
own correlation with return (`≈ 0.39`) — informative, not redundant.

Every number this module produces should be read as illustrative, not
tradable, until Phase 6 replaces these synthetic inputs.

## 3. Marchenko-Pastur filtering: adapted for one instrument, not textbook RMT

The classic RMT-filtering application (Laloux, Cizeau, Bouchaud & Potters
1999; Bouchaud & Potters' filtering of financial correlation matrices) is
**cross-sectional**: N columns are N genuinely distinct instruments (e.g.
different stocks) observed over the same T time periods. Under the null
hypothesis of "no real co-movement," those columns are independent random
variables, which is exactly the assumption the Marchenko-Pastur
distribution's derivation requires.

This system has order flow for exactly one instrument (NQ), so
`mp_filter_signal` is applied to a **self-referential panel** instead
(`compute_order_flow_features`): raw Volume Delta and OBI, each plus
rolling-mean smoothings at 3/5/10-day windows — 8 columns total, all
derived from the same one or two underlying series.

This is an explicit deviation from standard RMT, not an oversight, and it
has a real consequence: a 5-day and a 10-day rolling mean of the *same*
series share overlapping raw observations by construction, so they
correlate mechanically even under a genuine "no signal" null. The
Marchenko-Pastur upper bound computed here is a well-defined, correctly
implemented *threshold* — but treating "an eigenvalue exceeds it" as a
rigorous significance test is not justified the way it would be for a true
cross-sectional panel, because some of the "signal" this method finds may
simply reflect the smoothing-induced autocorrelation itself, not
information content in the order flow. This caveat is documented directly
in `mp_filter_signal`'s docstring, `compute_order_flow_features`'s
docstring, and the module header — it should stay visible to anyone
extending this code, not just live in this writeup.

The honest framing: this gets an MP-style denoising step working
end-to-end for a single instrument, at the cost of the textbook
independence assumption. A genuine cross-sectional version would need
order flow across multiple correlated instruments (e.g. related futures
contracts or tenors), which isn't available until real data is integrated.

## 4. Validation

Because the adaptation above is non-standard, the implementation was
checked against known random matrix theory before trusting it on real
data, rather than just trusting the formula (`tests/test_order_flow.py`):

**Pure-noise spectrum matches the theoretical MP band.** For a literal
i.i.d. Gaussian matrix (T=2000, N=8), the theoretical band is
`(0.878, 1.131)`; the empirical eigenvalue range came out to
`(0.879, 1.095)` — inside the band, as it should be for data with no
genuine structure at all.

**An injected common factor is correctly recovered.** A synthetic panel
was built from one true common factor plus independent noise at 2x the
factor's scale (deliberately noise-dominated per column) across 8 columns,
400 observations. `mp_filter_signal` correctly identified exactly one
eigenvalue above the MP bound (`n_signal_components = 1`, matching the one
factor actually injected: eigenvalues `[2.57, 0.97, 0.92, ...]` against a
bound of `1.35`), and the extracted composite signal correlated **0.84**
with the true underlying factor — recovered from data where no single
column was even close to that correlated with it individually.

**Pure noise correctly triggers the no-signal fallback.** With a fully
independent 400×8 random panel, no eigenvalue cleared the MP bound, and
the function fell back to returning `min_signal_components = 1` rather
than fabricating a false-positive "signal."

**On the actual synthetic order-flow panel** (Phase 1 price series, 1,260
days): 2 eigenvalues cleared the MP bound (`2.73` and `2.70`, against a
threshold of `1.17`) — consistent with there being two genuinely distinct
underlying series (Volume Delta and OBI) feeding the panel. The extracted
composite signal correlated 0.45 with same-day return and 0.41 with the
raw Volume Delta series, while reducing day-to-day "roughness" (mean
absolute standardized first difference) from 0.86 to 0.65 — a ~24%
reduction — confirming the filter denoises without discarding the
genuine shared signal.

## 5. Where this leaves things

Volume Delta and OBI are implemented and tested; MP filtering works and is
validated against known RMT results, but on a panel construction that
trades textbook rigor for feasibility with a single instrument — that
trade-off is documented at three levels (module docstring, function
docstrings, this writeup) so it can't be missed by whoever picks this up
next. The synthetic input data is a placeholder with explicit, individually
justified assumptions, none of them calibrated to real markets. Both
limitations resolve the same way: Phase 6's real data integration replaces
the synthetic bid/ask series with actual order flow, and ideally extends
to enough correlated instruments to make the MP filtering cross-sectional
in the textbook sense rather than self-referential.

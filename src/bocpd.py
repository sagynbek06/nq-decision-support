"""
Bayesian online changepoint detection (Adams & MacKay 2007), Student-t flavor.

Why this alongside the HMM. Program 1's HMM has to be fit on a window, decodes
a fixed menu of three states, and its Viterbi path can rewrite the past when
new data arrives. BOCPD is the other classical way to talk about regimes: it
keeps a posterior over the RUN LENGTH (bars since the last structural break),
updates in O(max_run) per bar with no refit, and never revises history. For a
system whose job is fast intraday calls, that is the property that matters.
Recent literature applies it directly to financial series and to order flow
(for example "Bayesian Online Changepoint Detection for Financial Time
Series", ACM 2026, and "Online Learning of Order Flow and Market Impact with
Bayesian Change-Point Detection Methods", arXiv 2307.02375), which is why it
is here as a complement to the HMM rather than a replacement.

Model. Within a segment, observations are Gaussian with unknown mean and
variance under a conjugate Normal-Inverse-Gamma prior, so the one-step
predictive is a Student-t. The t is heavy-tailed only while the segment is
young: its degrees of freedom are 2 * alpha_r, which grows with the run length,
so after a few hundred bars the predictive is effectively Gaussian. Measured
consequence (standard-normal noise, hazard 1/200, one spike at bar 250): a 3-4
sigma spike is absorbed with no reset, a 5 sigma spike is flagged and then
forgiven within ~50 bars, and 6-7 sigma spikes are read as a genuine break and
the run length restarts. So this detector is NOT robust to single large
outliers the way the Student-t HMM is; treat a one-bar jump in `change_prob`
after an extreme return with suspicion, and prefer to act on persistence (the
mass staying high for several bars, or `regime_age_confidence` staying low).

A property worth knowing before using the output. With a constant hazard rate
H, the posterior probability that the run length is exactly 0 equals H at
every step, by construction. It carries no information. The informative
quantity is the posterior MASS ON SHORT RUN LENGTHS: `recent_change_prob(k)`
is P(run length <= k), which rises sharply when the data stop looking like the
current segment and then decays as the new segment accumulates evidence. That
is what `update` reports as `change_prob`.

Limits, stated plainly. It detects changes in the mean and variance of
whatever series you feed it, nothing else. The prior scale matters on short
series. It reacts with a lag of a few bars (it needs evidence), so it is a
confirmation tool, not a predictor. Feed it standardized returns, not prices.

Causal by construction: `update(x)` uses only x and prior state.
"""

import numpy as np
from scipy.special import gammaln, logsumexp

DEFAULT_HAZARD = 1.0 / 100.0   # expected segment length of 100 bars
DEFAULT_MAX_RUN = 500
DEFAULT_PRIOR = {"mu0": 0.0, "kappa0": 1.0, "alpha0": 1.0, "beta0": 1.0}


def _student_t_logpdf(x, df, loc, scale2):
    """Log density of a location-scale Student-t, vectorized over run lengths."""
    z2 = (x - loc) ** 2 / scale2
    return (gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi * scale2)
            - (df + 1.0) / 2.0 * np.log1p(z2 / df))


class BOCPD:
    def __init__(self, hazard=DEFAULT_HAZARD, max_run=DEFAULT_MAX_RUN, prior=None):
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be in (0, 1)")
        p = dict(DEFAULT_PRIOR if prior is None else prior)
        self.hazard = hazard
        self.max_run = max_run
        self._prior = p
        # One entry per run length 0..len-1; start with a single run of length 0.
        self._logp = np.array([0.0])
        self._mu = np.array([p["mu0"]])
        self._kappa = np.array([p["kappa0"]])
        self._alpha = np.array([p["alpha0"]])
        self._beta = np.array([p["beta0"]])
        self.t = 0

    # -- posterior summaries ------------------------------------------------

    @property
    def run_length_posterior(self):
        """P(run length = r) for r = 0..len-1 (sums to 1)."""
        return np.exp(self._logp)

    def recent_change_prob(self, k=5):
        """P(run length <= k): posterior mass on 'a change happened within the last k bars'."""
        post = self.run_length_posterior
        return float(post[: k + 1].sum())

    def expected_run_length(self):
        post = self.run_length_posterior
        return float(np.dot(np.arange(len(post)), post))

    # -- update -------------------------------------------------------------

    def update(self, x, k=5):
        """Consume one observation; return a dict of posterior summaries."""
        x = float(x)
        df = 2.0 * self._alpha
        scale2 = self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa)
        log_pred = _student_t_logpdf(x, df, self._mu, scale2)

        log_h = np.log(self.hazard)
        log_1mh = np.log1p(-self.hazard)

        log_growth = self._logp + log_pred + log_1mh
        log_cp = logsumexp(self._logp + log_pred + log_h)

        new_logp = np.concatenate(([log_cp], log_growth))
        new_logp -= logsumexp(new_logp)

        p = self._prior
        mu_new = (self._kappa * self._mu + x) / (self._kappa + 1.0)
        kappa_new = self._kappa + 1.0
        alpha_new = self._alpha + 0.5
        beta_new = self._beta + self._kappa * (x - self._mu) ** 2 / (2.0 * (self._kappa + 1.0))

        self._mu = np.concatenate(([p["mu0"]], mu_new))
        self._kappa = np.concatenate(([p["kappa0"]], kappa_new))
        self._alpha = np.concatenate(([p["alpha0"]], alpha_new))
        self._beta = np.concatenate(([p["beta0"]], beta_new))
        self._logp = new_logp

        if len(self._logp) > self.max_run:
            # Truncate the longest runs and renormalize; their mass is negligible
            # for any hazard worth using and this keeps the update O(max_run).
            sl = slice(0, self.max_run)
            self._logp = self._logp[sl]
            self._logp -= logsumexp(self._logp)
            self._mu, self._kappa = self._mu[sl], self._kappa[sl]
            self._alpha, self._beta = self._alpha[sl], self._beta[sl]

        self.t += 1
        return {
            "change_prob": self.recent_change_prob(k),
            "expected_run_length": self.expected_run_length(),
            "map_run_length": int(np.argmax(self._logp)),
        }


def run_bocpd(series, hazard=DEFAULT_HAZARD, k=5, **kwargs):
    """Run BOCPD over a whole series causally. Returns arrays aligned to `series`."""
    series = np.asarray(series, dtype=float)
    model = BOCPD(hazard=hazard, **kwargs)
    change_prob = np.zeros(len(series))
    expected_run = np.zeros(len(series))
    map_run = np.zeros(len(series), dtype=int)
    for i, x in enumerate(series):
        out = model.update(x, k=k)
        change_prob[i] = out["change_prob"]
        expected_run[i] = out["expected_run_length"]
        map_run[i] = out["map_run_length"]
    return {"change_prob": change_prob, "expected_run_length": expected_run, "map_run_length": map_run}


def regime_age_confidence(map_run_length, saturation=30):
    """
    Map the MAP run length to a [0, 1] trust factor: a regime that is only a
    few bars old has not earned much confidence, one that has persisted for
    `saturation` bars or more earns full confidence. Intended as a multiplier on
    the regime vote (a young regime counts for less), not as a signal itself.
    """
    return float(min(max(map_run_length, 0) / float(saturation), 1.0))

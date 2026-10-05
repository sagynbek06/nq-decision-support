"""
Conformal abstention gate for the Consensus Engine.

The consensus score says which way the votes lean. It says nothing about how
much that lean is worth: a score of +0.4 on a quiet day and +0.4 in the middle
of a volatility shock look identical to `position_size`. This module adds the
missing piece, a distribution-free prediction interval for the NEXT bar's
return, and abstains whenever that interval still contains zero. That fits the
project's stated philosophy (fewer, higher-quality calls; no trade beats a bad
trade) with a mechanism that carries a statistical guarantee instead of a
hand-tuned threshold.

Method: adaptive conformal inference, in the spirit of Temporal Conformal
Prediction (arXiv 2507.05470, which updates its interval threshold online with
gamma_t = gamma_0 / (1 + lambda * t)^beta, beta in (0.5, 1)) and of Gibbs &
Candes' adaptive conformal inference.

Step-size choice, measured rather than assumed. A DECAYING step (lambda > 0,
TCP's schedule) gives the cleanest asymptotic guarantee when the data are
stationary, but it freezes the interval just when a regime change needs it to
move: on this repo's test series (volatility doubles mid-sample) lambda = 0.01
reached only 85.5% coverage against a 90% target, averaged over 8 seeds. A
CONSTANT step (lambda = 0, the Gibbs-Candes choice) reached 89.4-89.9%. Markets
shift regimes, so the default here is constant; `lam` stays a parameter for
anyone who wants the decaying schedule on a series they trust to be stable.
At each bar:

    1. The base forecast is the consensus score times `forecast_scale` (a
       scaling from score units to return units, fit on a warm-up window by
       least squares through the origin, then frozen -- see `fit_scale`).
    2. The interval is forecast +/- q_t, where q_t is a running quantile of
       absolute forecast errors that is nudged after every bar: widened if the
       last return fell outside the interval, narrowed if it fell inside.
    3. The long-run miss rate converges to `alpha` without any assumption that
       returns are exchangeable. That is the whole point: financial returns
       are not.

What the guarantee does and does not say. Long-run MARGINAL coverage tends to
1 - alpha. It is not conditional coverage (a given regime can be over- or
under-covered while the average is right), and it says nothing about whether
the forecast has any edge. A forecast with no skill just gets wide intervals
that always contain zero, so the gate abstains almost always. That is the
correct failure mode: it turns "no edge" into "no trades" rather than into
confident losing trades.

All computation is causal. `update` consumes the realized return of the bar
that just closed; `interval` only reads state built from earlier bars.
"""

import numpy as np

DEFAULT_ALPHA = 0.10
DEFAULT_GAMMA0 = 0.05
DEFAULT_LAMBDA = 0.0   # 0 = constant step size; see the module docstring for why
DEFAULT_BETA = 0.7
DEFAULT_WARMUP = 60


def fit_scale(scores, next_returns):
    """
    Least-squares slope (through the origin) mapping consensus scores to the
    next bar's return. Returns 0.0 for a degenerate (all-zero) score history,
    which makes the gate abstain until real signal arrives.
    """
    scores = np.asarray(scores, dtype=float)
    next_returns = np.asarray(next_returns, dtype=float)
    if len(scores) != len(next_returns):
        raise ValueError("fit_scale: scores and next_returns must be the same length")
    denom = float(np.dot(scores, scores))
    if denom == 0.0:
        return 0.0
    return float(np.dot(scores, next_returns) / denom)


class ConformalGate:
    """
    Online adaptive-conformal interval for next-bar returns, plus an
    abstention rule.

    Usage, one bar at a time:

        gate = ConformalGate(alpha=0.10, forecast_scale=scale)
        for t in range(n):
            decision = gate.decide(score[t])      # uses only the past
            ...                                    # act on decision
            gate.update(score[t], realized_return[t])   # after the bar closes
    """

    def __init__(self, alpha=DEFAULT_ALPHA, forecast_scale=1.0, gamma0=DEFAULT_GAMMA0,
                 lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA, initial_width=None, warmup=DEFAULT_WARMUP):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.5 < beta < 1.0:
            raise ValueError("beta must be in (0.5, 1) for the step sizes to be summable-squared but not summable")
        self.alpha = alpha
        self.forecast_scale = forecast_scale
        self.gamma0 = gamma0
        self.lam = lam
        self.beta = beta
        self.warmup = warmup
        self.t = 0
        self._errors = []
        self._width = initial_width  # None until the first error arrives

    # -- state ---------------------------------------------------------------

    @property
    def width(self):
        """Current half-width of the interval (None before any data)."""
        return self._width

    def forecast(self, score):
        return self.forecast_scale * float(score)

    def interval(self, score):
        """(lower, upper) for the next return, or (-inf, inf) before warm-up ends."""
        f = self.forecast(score)
        if self._width is None or self.t < self.warmup:
            return (-np.inf, np.inf)
        return (f - self._width, f + self._width)

    def decide(self, score):
        """
        Returns a dict with the interval and a decision:
        "long" if the whole interval sits above zero, "short" if it sits below,
        "abstain" otherwise (including during warm-up).
        """
        lo, hi = self.interval(score)
        if lo > 0.0:
            action = "long"
        elif hi < 0.0:
            action = "short"
        else:
            action = "abstain"
        return {"action": action, "lower": lo, "upper": hi, "forecast": self.forecast(score)}

    def update(self, score, realized_return):
        """Feed the bar that just closed. Call after `decide`, never before."""
        error = abs(float(realized_return) - self.forecast(score))
        self._errors.append(error)
        self.t += 1

        if self._width is None:
            # First observation seeds the width; adaptation takes over from here.
            self._width = error if error > 0 else 1e-6
            return

        gamma = self.gamma0 / (1.0 + self.lam * self.t) ** self.beta
        # Pinball-loss gradient step on the (1 - alpha) quantile of |error|:
        # a miss (error > width) pushes the width up by gamma * (1 - alpha),
        # a cover pulls it down by gamma * alpha, so the miss rate settles at alpha.
        miss = 1.0 if error > self._width else 0.0
        scale = max(np.mean(self._errors[-self.warmup:]), 1e-12)
        self._width = max(self._width + gamma * scale * (miss - self.alpha), 1e-12)


def run_gate(scores, next_returns, alpha=DEFAULT_ALPHA, warmup=DEFAULT_WARMUP, **kwargs):
    """
    Convenience: walk a whole series causally. The forecast scale is fit on the
    first `warmup` bars only and then frozen, so nothing later leaks backwards.

    Returns a dict of arrays: action, lower, upper, covered (bool, NaN-free
    after warm-up) and the empirical coverage over the post-warm-up segment.
    """
    scores = np.asarray(scores, dtype=float)
    next_returns = np.asarray(next_returns, dtype=float)
    n = len(scores)
    if n != len(next_returns):
        raise ValueError("run_gate: scores and next_returns must be the same length")
    if n <= warmup:
        raise ValueError("run_gate: need more bars than the warm-up window")

    scale = fit_scale(scores[:warmup], next_returns[:warmup])
    gate = ConformalGate(alpha=alpha, forecast_scale=scale, warmup=warmup, **kwargs)

    actions, lowers, uppers, covered = [], [], [], []
    for t in range(n):
        d = gate.decide(scores[t])
        actions.append(d["action"])
        lowers.append(d["lower"])
        uppers.append(d["upper"])
        covered.append(d["lower"] <= next_returns[t] <= d["upper"])
        gate.update(scores[t], next_returns[t])

    covered = np.array(covered, dtype=bool)
    post = slice(warmup, None)
    return {
        "action": np.array(actions),
        "lower": np.array(lowers),
        "upper": np.array(uppers),
        "covered": covered,
        "coverage": float(covered[post].mean()),
        "abstain_rate": float(np.mean(np.array(actions)[post] == "abstain")),
        "forecast_scale": scale,
    }

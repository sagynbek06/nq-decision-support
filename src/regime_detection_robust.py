"""
Program 1 (robust variant): HMM with Student-t emissions.

The Gaussian baseline in `regime_detection.py` assumes Gaussian emissions --
an assumption `notebooks/02_gaussian_assumption_check.ipynb` found to be
violated (returns within every regime reject normality via Jarque-Bera).
This module implements a 3-state HMM with univariate Student-t emissions
instead, with EM written from scratch (hmmlearn only supports Gaussian/GMM
emissions).

The Student-t is used via its Gaussian scale-mixture representation:
    x_t | u_t, state k  ~  Normal(mu_k, sigma2_k / u_t)
    u_t                 ~  Gamma(nu_k / 2, nu_k / 2)
which marginalizes to x_t | state k ~ Student-t(mu_k, sigma2_k, nu_k). This
lets the E-step compute, alongside the usual HMM state posteriors, a
per-observation "reliability weight" u_t: outlier returns get a small u_t
and are automatically downweighted in the M-step, instead of dragging a
whole state's mean/variance toward them the way a Gaussian HMM's emissions
would. See McLachlan & Peel, "Finite Mixture Models" (2000), Ch. 7, for the
underlying EM derivation (there for mixtures; here adapted to an HMM's
forward-backward state posteriors playing the role of mixture weights).
"""

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import brentq

N_REGIMES = 3
N_INIT = 15


def _student_t_logpdf(x, mu, sigma2, nu):
    """Log-density of a scalar Student-t(mu, sigma^2, nu) at each point in x."""
    z2 = (x - mu) ** 2 / sigma2
    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi * sigma2)
        - ((nu + 1) / 2) * np.log1p(z2 / nu)
    )


def _logsumexp_axis0(a):
    """logsumexp(a, axis=0) for a 2D array, without scipy's general-N-d overhead."""
    m = a.max(axis=0)
    return m + np.log(np.sum(np.exp(a - m), axis=0))


def _logsumexp_axis1(a):
    """logsumexp(a, axis=1) for a 2D array, without scipy's general-N-d overhead."""
    m = a.max(axis=1)
    return m + np.log(np.sum(np.exp(a - m[:, None]), axis=1))


def _logsumexp_1d(a):
    m = a.max()
    return m + np.log(np.sum(np.exp(a - m)))


def _forward_backward(log_pi, log_A, log_B):
    """
    Log-space forward-backward. log_B has shape (T, K) with
    log_B[t, k] = log p(x_t | state=k). Returns (log_alpha, log_beta, loglik).

    This is the hot loop of the whole module: it runs once per EM iteration,
    per random restart. It's implemented with hand-rolled logsumexp helpers
    rather than scipy.special.logsumexp -- profiling showed scipy's general
    N-dimensional implementation costs ~8x more per call than a version
    specialized to our fixed 2D shape, and this loop makes thousands of
    those calls per fit.
    """
    T, K = log_B.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = log_B[t] + _logsumexp_axis0(log_alpha[t - 1][:, None] + log_A)

    log_beta = np.empty((T, K))
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp_axis1(log_A + (log_B[t + 1] + log_beta[t + 1])[None, :])

    loglik = _logsumexp_1d(log_alpha[T - 1])
    return log_alpha, log_beta, loglik


def _viterbi(log_pi, log_A, log_B):
    """Most likely state path via the Viterbi algorithm (log-space)."""
    T, K = log_B.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A  # scores[j, k]
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_B[t]

    states = np.empty(T, dtype=int)
    states[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _solve_nu(mean_log_u_minus_u, min_nu, max_nu):
    """
    Root-find the EM M-step equation for a Student-t's degrees of freedom:
        1 + log(nu/2) - digamma(nu/2) + mean_log_u_minus_u = 0
    This function is strictly decreasing in nu, from +inf as nu -> 0 to
    (1 + mean_log_u_minus_u) <= 0 as nu -> inf (since log(u) <= u - 1 makes
    mean_log_u_minus_u <= -1 always), so a root always exists in (0, inf).
    In practice we solve on a fixed bracket and clip to it: heavy-tailed
    data pushes the root below min_nu (clip there), near-Gaussian data
    pushes it toward infinity (clip at max_nu).
    """
    def g(nu):
        return 1 + np.log(nu / 2) - digamma(nu / 2) + mean_log_u_minus_u

    g_lo, g_hi = g(min_nu), g(max_nu)
    if g_lo <= 0:
        return min_nu
    if g_hi >= 0:
        return max_nu
    return brentq(g, min_nu, max_nu)


class StudentTHMM:
    """A hidden Markov model with univariate Student-t emissions, fit via EM."""

    def __init__(
        self,
        n_states=N_REGIMES,
        nu_init=8.0,
        n_iter=100,
        tol=1e-4,
        n_init=N_INIT,
        random_state=42,
        update_nu=True,
        min_nu=2.05,
        max_nu=200.0,
        min_sigma2=1e-10,
    ):
        self.n_states = n_states
        self.nu_init = nu_init
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.update_nu = update_nu
        self.min_nu = min_nu
        self.max_nu = max_nu
        self.min_sigma2 = min_sigma2

    def fit(self, returns):
        x = np.asarray(returns, dtype=float)
        rng = np.random.default_rng(self.random_state)
        init_seeds = rng.integers(0, 2**32 - 1, size=self.n_init)

        best_params, best_score = None, -np.inf
        for seed in init_seeds:
            try:
                params, score = self._fit_single(x, int(seed))
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                continue
            if np.isfinite(score) and score > best_score:
                best_params, best_score = params, score

        if best_params is None:
            raise RuntimeError("StudentTHMM.fit: every random initialization failed")

        self.pi_, self.A_, self.mu_, self.sigma2_, self.nu_ = best_params
        self.score_ = best_score
        return self

    def _init_params(self, x, seed):
        rng = np.random.default_rng(seed)
        K = self.n_states
        idx = rng.choice(len(x), size=K, replace=False)
        mu = np.sort(x[idx])
        sigma2 = np.full(K, np.var(x))
        nu = np.full(K, self.nu_init)
        pi = np.full(K, 1.0 / K)
        A = np.full((K, K), 1.0 / K)
        return pi, A, mu, sigma2, nu

    def _log_emissions(self, x, mu, sigma2, nu):
        K = len(mu)
        return np.stack([_student_t_logpdf(x, mu[k], sigma2[k], nu[k]) for k in range(K)], axis=1)

    def _fit_single(self, x, seed, loglik_trace=None):
        """
        `loglik_trace`, if given a list, gets each iteration's log-likelihood
        appended to it -- used by tests to check the EM monotonicity
        invariant (log-likelihood must never decrease). Not used in normal
        fitting.
        """
        K = self.n_states
        pi, A, mu, sigma2, nu = self._init_params(x, seed)

        prev_loglik = -np.inf
        for iteration in range(self.n_iter):
            log_B = self._log_emissions(x, mu, sigma2, nu)
            log_pi = np.log(pi + 1e-300)
            log_A = np.log(A + 1e-300)

            log_alpha, log_beta, loglik = _forward_backward(log_pi, log_A, log_B)
            if loglik_trace is not None:
                loglik_trace.append(float(loglik))

            # log_gamma/log_xi are mathematically <= 0 (they're log-probabilities),
            # but can overshoot by a hair of floating-point noise; clip defensively
            # before exponentiating.
            gamma = np.exp(np.clip(log_alpha + log_beta - loglik, -700, 0))
            gamma /= gamma.sum(axis=1, keepdims=True)

            log_xi = (
                log_alpha[:-1, :, None]
                + log_A[None, :, :]
                + log_B[1:, None, :]
                + log_beta[1:, None, :]
                - loglik
            )
            xi_sum = np.exp(np.clip(log_xi, -700, 0)).sum(axis=0)

            # M-step: HMM structural parameters
            pi = gamma[0] / gamma[0].sum()
            A = xi_sum / xi_sum.sum(axis=1, keepdims=True)

            # E-step (latent scale weights) + M-step (emission params) per state
            new_mu = np.empty(K)
            new_sigma2 = np.empty(K)
            new_nu = np.empty(K)
            for k in range(K):
                delta_k = (x - mu[k]) ** 2 / sigma2[k]
                u_k = (nu[k] + 1) / (nu[k] + delta_k)
                w = gamma[:, k]
                n_k = w.sum()

                new_mu[k] = np.sum(w * u_k * x) / np.sum(w * u_k)
                new_sigma2[k] = max(
                    np.sum(w * u_k * (x - new_mu[k]) ** 2) / n_k, self.min_sigma2
                )

                if self.update_nu:
                    log_u_k = digamma((nu[k] + 1) / 2) - np.log((nu[k] + delta_k) / 2)
                    mean_term = np.sum(w * (log_u_k - u_k)) / n_k
                    new_nu[k] = _solve_nu(mean_term, self.min_nu, self.max_nu)
                else:
                    new_nu[k] = nu[k]

            mu, sigma2, nu = new_mu, new_sigma2, new_nu

            # Student-t EM has a well-documented slow linear-convergence
            # tail (worse for small nu), so an absolute log-likelihood
            # tolerance is impractical here -- it would rarely trigger
            # before n_iter. Use a relative one instead.
            if iteration > 0 and abs(loglik - prev_loglik) < self.tol * abs(prev_loglik):
                prev_loglik = loglik
                break
            prev_loglik = loglik

        log_B = self._log_emissions(x, mu, sigma2, nu)
        _, _, final_loglik = _forward_backward(np.log(pi + 1e-300), np.log(A + 1e-300), log_B)
        return (pi, A, mu, sigma2, nu), final_loglik

    def predict(self, returns):
        """Most likely hidden state path (Viterbi) for a returns series."""
        x = np.asarray(returns, dtype=float)
        log_B = self._log_emissions(x, self.mu_, self.sigma2_, self.nu_)
        return _viterbi(np.log(self.pi_ + 1e-300), np.log(self.A_ + 1e-300), log_B)

    def filter_states(self, returns):
        """
        Most likely state at each bar given only the returns up to that bar
        (the forward/filtering pass). Viterbi's path labels an early bar using
        later returns, so anything traded in real time must use this instead.
        """
        x = np.asarray(returns, dtype=float)
        log_B = self._log_emissions(x, self.mu_, self.sigma2_, self.nu_)
        log_alpha, _, _ = _forward_backward(np.log(self.pi_ + 1e-300), np.log(self.A_ + 1e-300), log_B)
        return np.argmax(log_alpha, axis=1)

    def score(self, returns):
        """Total log-likelihood of a returns series under the fitted model."""
        x = np.asarray(returns, dtype=float)
        log_B = self._log_emissions(x, self.mu_, self.sigma2_, self.nu_)
        _, _, loglik = _forward_backward(np.log(self.pi_ + 1e-300), np.log(self.A_ + 1e-300), log_B)
        return loglik


def fit_hmm(returns, n_states=N_REGIMES, **kwargs):
    """Fit a Student-t HMM to a 1D array of daily log returns."""
    return StudentTHMM(n_states=n_states, **kwargs).fit(returns)


def label_states(model):
    """
    Map each hidden state of a fitted 3-state Student-t HMM to a regime name.
    Same mean/volatility logic as the Gaussian baseline's label_states, but
    volatility here is the Student-t's implied standard deviation
    sqrt(sigma^2 * nu / (nu - 2)) rather than sqrt(sigma^2) directly, since
    the fitted scale parameter alone understates spread for small nu.
    """
    if model.n_states != N_REGIMES:
        raise ValueError(
            f"label_states expects a {N_REGIMES}-state model, got {model.n_states}"
        )

    means = model.mu_
    nu = model.nu_
    vols = np.sqrt(model.sigma2_ * nu / (nu - 2))

    bull_state = int(np.argmax(means))
    remaining = [s for s in range(N_REGIMES) if s != bull_state]
    bear_state, sideways_state = sorted(remaining, key=lambda s: vols[s], reverse=True)
    return {bull_state: "bull", bear_state: "bear", sideways_state: "sideways"}


def predict_regimes(model, returns, state_labels):
    """Decode the most likely hidden state path and map it to regime names."""
    hidden_states = model.predict(returns)
    return np.array([state_labels[state] for state in hidden_states])


def fit_and_label(returns, n_states=N_REGIMES, **kwargs):
    """Fit the Student-t HMM and return (model, state_labels, predicted regime labels)."""
    model = fit_hmm(returns, n_states=n_states, **kwargs)
    state_labels = label_states(model)
    predicted = predict_regimes(model, returns, state_labels)
    return model, state_labels, predicted

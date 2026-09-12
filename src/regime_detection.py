"""
Program 1: HMM-based market regime detection.

Fits a 3-state Gaussian-emission Hidden Markov Model (via Baum-Welch / EM,
through hmmlearn) to a series of daily log returns, then labels each hidden
state as "bull", "bear", or "sideways" from its fitted mean return and
volatility. This is the Gaussian baseline from ROADMAP.md Phase 1; a more
robust skew-t emission model is planned as a follow-up.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM

N_REGIMES = 3
N_INIT = 25


def fit_hmm(returns, n_states=N_REGIMES, n_iter=200, random_state=42, n_init=N_INIT):
    """
    Fit a Gaussian HMM to a 1D array of daily log returns via Baum-Welch.

    Baum-Welch is EM under the hood, so it only finds a local optimum of the
    likelihood and is sensitive to its starting point. We fit `n_init` times
    from different random initializations (derived from `random_state`, so
    the result is still deterministic) and keep the model with the highest
    training log-likelihood.
    """
    X = np.asarray(returns).reshape(-1, 1)
    init_seeds = np.random.default_rng(random_state).integers(0, 2**32 - 1, size=n_init)

    best_model, best_score = None, -np.inf
    for seed in init_seeds:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=int(seed),
        )
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def _state_mean_and_vol(model):
    means = model.means_.flatten()
    variances = np.asarray(model.covars_)[:, 0, 0]
    return means, np.sqrt(variances)


def label_states(model):
    """
    Map each hidden state of a fitted 3-state HMM to a regime name.

    Bull is the state with the highest mean return. Between the two
    remaining states, the higher-volatility one is labeled bear (drawdowns
    are noisier than chop) and the lower-volatility one is sideways.
    """
    if model.n_components != N_REGIMES:
        raise ValueError(
            f"label_states expects a {N_REGIMES}-state model, got {model.n_components}"
        )

    means, vols = _state_mean_and_vol(model)
    bull_state = int(np.argmax(means))
    remaining = [s for s in range(N_REGIMES) if s != bull_state]
    bear_state, sideways_state = sorted(remaining, key=lambda s: vols[s], reverse=True)
    return {bull_state: "bull", bear_state: "bear", sideways_state: "sideways"}


def predict_regimes(model, returns, state_labels):
    """Decode the most likely hidden state path (Viterbi) and map it to regime names."""
    X = np.asarray(returns).reshape(-1, 1)
    hidden_states = model.predict(X)
    return np.array([state_labels[state] for state in hidden_states])


def fit_and_label(returns, n_states=N_REGIMES, n_iter=200, random_state=42, n_init=N_INIT):
    """Fit the HMM and return (model, state_labels, predicted regime labels)."""
    model = fit_hmm(returns, n_states=n_states, n_iter=n_iter, random_state=random_state, n_init=n_init)
    state_labels = label_states(model)
    predicted = predict_regimes(model, returns, state_labels)
    return model, state_labels, predicted

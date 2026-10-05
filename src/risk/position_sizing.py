"""
Position sizing on top of the Consensus Engine.

    size = sign(score) * max_size * magnitude * agreement * throttle

- magnitude: how far |score| sits above the neutral band, rescaled to (0, 1].
  It starts at 0 at the band edge, so size is continuous through the threshold.
- agreement: the share of consensus votes whose sign matches the score's. An
  abstaining vote (exactly 0) counts as not agreeing, consistent with how
  src/dynamic_weights.py treats abstentions.
- throttle: 1 - CDaR_current / CDaR_limit, clipped to [0, 1]. Full size while
  the trailing drawdown tail is clean, reduced linearly as it approaches the
  limit, and zero at or past it.

A score within the neutral band of zero -- the same test compute_consensus
uses to call "neutral" -- gives size exactly 0.0.
"""

import numpy as np

import src.consensus_engine as consensus_engine


def throttle_from_cdar(cdar_current, cdar_limit):
    if cdar_limit <= 0:
        raise ValueError(f"cdar_limit must be positive, got {cdar_limit}")
    return float(np.clip(1.0 - cdar_current / cdar_limit, 0.0, 1.0))


def position_size(consensus, cdar_current, cdar_limit, neutral_band=None, max_size=1.0):
    band = consensus_engine.NEUTRAL_BAND if neutral_band is None else neutral_band
    if not 0 <= band < 1:
        raise ValueError(f"neutral_band must be in [0, 1), got {band}")

    score = consensus["score"]
    if abs(score) <= band:
        return 0.0

    direction = np.sign(score)
    magnitude = (abs(score) - band) / (1.0 - band)
    votes = list(consensus["votes"].values())
    agreement = sum(np.sign(v) == direction for v in votes) / len(votes)
    throttle = throttle_from_cdar(cdar_current, cdar_limit)
    return float(direction * max_size * magnitude * agreement * throttle)

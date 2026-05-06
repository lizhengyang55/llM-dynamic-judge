from __future__ import annotations

import math


def exploration_decay(
    database_size: int,
    sampled_questions: int,
    alpha: float = 0.005,
    beta: float = 0.001,
) -> float:
    """Compute m = 1 - exp(-(alpha * N + beta * S))."""
    return 1.0 - math.exp(-(alpha * database_size + beta * sampled_questions))


def dynamic_bias_weight(
    base_weight: float,
    h_top_k: float,
    triggered: bool,
    lambda_: float = 1.5,
    alpha_bias: float = 2.0,
    epsilon: float = 0.1,
) -> float:
    """Apply w' = w + lambda * H_topK^alpha_bias / (w + epsilon) when triggered."""
    if not triggered:
        return base_weight
    return base_weight + lambda_ * ((h_top_k**alpha_bias) / (base_weight + epsilon))


def sharpened_softmax_probability(
    score_a: float,
    score_b: float,
    tau: float = 2.0,
) -> float:
    """Compute exp(tau*S_A) / (exp(tau*S_A) + exp(tau*S_B)) stably."""
    scaled_a = tau * score_a
    scaled_b = tau * score_b
    max_scaled = max(scaled_a, scaled_b)
    exp_a = math.exp(scaled_a - max_scaled)
    exp_b = math.exp(scaled_b - max_scaled)
    return exp_a / (exp_a + exp_b)


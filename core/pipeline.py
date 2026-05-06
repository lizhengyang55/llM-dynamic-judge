from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from core.database import DatabaseConfig, MetricDatabase, MetricItem, MetricReason
from core.math_utils import (
    dynamic_bias_weight,
    exploration_decay,
    sharpened_softmax_probability,
)
from llm_api.agent_mock import DualAgentMock


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluatorConfig:
    exploration_alpha: float = 0.005
    exploration_beta: float = 0.001
    reasons_per_side: int = 5
    bias_lambda: float = 1.5
    bias_alpha: float = 2.0
    bias_epsilon: float = 0.1
    softmax_tau: float = 2.0
    hit_reward: float = 0.18
    miss_penalty: float = 0.10
    ambiguous_penalty: float = 0.04
    min_weight: float = 0.05
    max_weight: float = 8.0
    topic_top_k: int = 3
    purge_rounds: int = 20
    purge_top_x: int = 5
    purge_threshold: float = 0.45
    random_seed: int = 42

    def to_database_config(self) -> DatabaseConfig:
        return DatabaseConfig(
            hit_reward=self.hit_reward,
            miss_penalty=self.miss_penalty,
            ambiguous_penalty=self.ambiguous_penalty,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            topic_top_k=self.topic_top_k,
        )


class EvaluatorPipeline:
    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()
        self.rng = random.Random(self.config.random_seed)
        self.db = MetricDatabase(config=self.config.to_database_config())
        self.agent = DualAgentMock(reasons_per_side=self.config.reasons_per_side, rng=self.rng)
        self.sampled_question_count = 0

    def bootstrap_mock_data(self, topics: Sequence[str], per_topic: int = 4) -> None:
        self.agent.seed_database(db=self.db, topics=topics, per_topic=per_topic)
        logger.info("Bootstrapped Metric DB with %d metric(s).", len(self.db))

    def compute_m_value(self) -> float:
        return exploration_decay(
            database_size=len(self.db),
            sampled_questions=self.sampled_question_count,
            alpha=self.config.exploration_alpha,
            beta=self.config.exploration_beta,
        )

    def train_step(self, topic: str, ans_chosen: str, ans_rejected: str) -> None:
        m_value = self.compute_m_value()
        logger.info(
            "Train step topic=%s | N=%d | S=%d | m=%.4f",
            topic,
            len(self.db),
            self.sampled_question_count,
            m_value,
        )

        reasons_chosen, reasons_rejected = self.agent.evaluate_pair(
            topic=topic,
            ans_a=ans_chosen,
            ans_b=ans_rejected,
            db=self.db,
            m_value=m_value,
        )
        self.db.update_metric_weights(
            topic=topic,
            chosen_reasons=reasons_chosen,
            rejected_reasons=reasons_rejected,
        )
        self.sampled_question_count += 1

    def inference_step(self, topic: str, ans_a: str, ans_b: str) -> Dict[str, object]:
        m_value = self.compute_m_value()
        logger.info(
            "Inference topic=%s | N=%d | S=%d | m=%.4f",
            topic,
            len(self.db),
            self.sampled_question_count,
            m_value,
        )

        reasons_a, reasons_b = self.agent.evaluate_pair(
            topic=topic,
            ans_a=ans_a,
            ans_b=ans_b,
            db=self.db,
            m_value=m_value,
        )
        score_a, details_a = self._score_reasons_with_dynamic_bias(topic, reasons_a)
        score_b, details_b = self._score_reasons_with_dynamic_bias(topic, reasons_b)
        prob_a = sharpened_softmax_probability(
            score_a=score_a,
            score_b=score_b,
            tau=self.config.softmax_tau,
        )
        winner = "A" if prob_a >= 0.5 else "B"

        logger.info("S_A=%.4f | S_B=%.4f", score_a, score_b)
        logger.info("P(A beats B)=%.4f | winner=%s", prob_a, winner)

        return {
            "topic": topic,
            "m_value": m_value,
            "S_A": score_a,
            "S_B": score_b,
            "prob_A": prob_a,
            "winner": winner,
            "reasons_A": reasons_a,
            "reasons_B": reasons_b,
            "details_A": details_a,
            "details_B": details_b,
        }

    def purge(self) -> List[MetricItem]:
        return self.db.purge_metrics(
            n_rounds=self.config.purge_rounds,
            top_x=self.config.purge_top_x,
            threshold=self.config.purge_threshold,
        )

    def _score_reasons_with_dynamic_bias(
        self,
        topic: str,
        reasons: Sequence[MetricReason],
    ) -> Tuple[float, List[Dict[str, object]]]:
        total_score = 0.0
        details: List[Dict[str, object]] = []

        for reason in reasons:
            base_weight = reason.base_weight
            top_topics = self.db.top_k_topics_for_metric(reason.metric, self.config.topic_top_k)
            triggered = topic in top_topics
            h_top_k = reason.metric.hit_rate_for_topic(topic)
            final_weight = dynamic_bias_weight(
                base_weight=base_weight,
                h_top_k=h_top_k,
                triggered=triggered,
                lambda_=self.config.bias_lambda,
                alpha_bias=self.config.bias_alpha,
                epsilon=self.config.bias_epsilon,
            )
            total_score += final_weight

            detail = {
                "metric": reason.metric.name,
                "polarity": reason.polarity,
                "source": reason.source,
                "base_weight": base_weight,
                "final_weight": final_weight,
                "bias_triggered": triggered,
                "H_topK": h_top_k,
                "top_topics": top_topics,
            }
            details.append(detail)
            logger.info(
                "Bias %-7s | metric=%s | base=%.3f | H=%.3f | final=%.3f | topK=%s",
                "trigger" if triggered else "skip",
                reason.metric.name,
                base_weight,
                h_top_k,
                final_weight,
                top_topics,
            )

        return total_score, details


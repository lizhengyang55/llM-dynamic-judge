from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.database import MetricDatabase, MetricItem, MetricReason


logger = logging.getLogger(__name__)


class DualAgentMock:
    """Local mock for dual LLM agents that extract or generate pairwise metrics."""

    def __init__(self, reasons_per_side: int, rng: random.Random) -> None:
        self.reasons_per_side = reasons_per_side
        self.rng = rng
        self._new_metric_idx = 0
        self._topic_templates: Dict[str, List[str]] = {
            "math": [
                "logical_derivation",
                "symbol_consistency",
                "boundary_coverage",
                "calculation_verifiability",
                "conclusion_alignment",
            ],
            "coding": [
                "algorithmic_complexity",
                "edge_case_handling",
                "maintainability",
                "test_coverage",
                "interface_contract",
            ],
            "writing": [
                "argument_focus",
                "natural_expression",
                "evidence_density",
                "structural_clarity",
                "audience_fit",
            ],
            "reasoning": [
                "evidence_chain",
                "counterexample_sensitivity",
                "assumption_explicitness",
                "multi_step_stability",
                "confidence_calibration",
            ],
            "general": [
                "instruction_following",
                "factual_grounding",
                "completeness",
                "clarity",
                "helpfulness",
            ],
        }

    def seed_database(self, db: MetricDatabase, topics: Sequence[str], per_topic: int = 4) -> None:
        for topic in topics:
            templates = self._topic_templates.get(topic, self._topic_templates["general"])
            for idx in range(per_topic):
                name = f"{topic}:{templates[idx % len(templates)]}"
                db.add_metric(MetricItem(name=name, topic=topic))

    def evaluate_pair(
        self,
        topic: str,
        ans_a: str,
        ans_b: str,
        db: MetricDatabase,
        m_value: float,
        user_prompt: Optional[str] = None,
    ) -> Tuple[List[MetricReason], List[MetricReason]]:
        del ans_a, ans_b, user_prompt
        reasons_a = self._build_side_reasons(topic=topic, db=db, m_value=m_value, side="A")
        reasons_b = self._build_side_reasons(topic=topic, db=db, m_value=m_value, side="B")
        logger.info("Agent A reasons: %s", [reason.label() for reason in reasons_a])
        logger.info("Agent B reasons: %s", [reason.label() for reason in reasons_b])
        return reasons_a, reasons_b

    def evaluate_pairs_batch(
        self,
        batch_pairs: List[Any],
        db: MetricDatabase,
        m_value: float,
    ) -> List[Tuple[List[MetricReason], List[MetricReason]]]:
        return [
            self.evaluate_pair(
                topic=sample.topic,
                ans_a=sample.answer_a,
                ans_b=sample.answer_b,
                db=db,
                m_value=m_value,
                user_prompt=sample.user_prompt,
            )
            for sample in batch_pairs
        ]

    def _build_side_reasons(
        self,
        topic: str,
        db: MetricDatabase,
        m_value: float,
        side: str,
    ) -> List[MetricReason]:
        db_count = round(self.reasons_per_side * m_value)
        db_count = min(self.reasons_per_side, max(0, db_count))
        generated_count = self.reasons_per_side - db_count

        used_names: set[str] = set()
        picked_metrics = db.sample_relevant_metrics(
            topic=topic,
            count=db_count,
            rng=self.rng,
            exclude_names=used_names,
        )
        used_names.update(metric.name for metric in picked_metrics)

        generated_metrics = [
            self._generate_metric(topic=topic, db=db, side=side, exclude_names=used_names)
            for _ in range(generated_count)
        ]

        reasons = [
            MetricReason(metric=metric, polarity=self._sample_polarity(), source="db")
            for metric in picked_metrics
        ]
        reasons.extend(
            MetricReason(metric=metric, polarity=self._sample_polarity(), source="generated")
            for metric in generated_metrics
        )
        self.rng.shuffle(reasons)

        logger.info(
            "Side %s extraction ratio: db=%d, generated=%d, total=%d",
            side,
            len(picked_metrics),
            len(generated_metrics),
            len(reasons),
        )
        return reasons

    def _generate_metric(
        self,
        topic: str,
        db: MetricDatabase,
        side: str,
        exclude_names: set[str],
    ) -> MetricItem:
        templates = self._topic_templates.get(topic, self._topic_templates["general"])
        phrase = self.rng.choice(templates)

        for _ in range(20):
            self._new_metric_idx += 1
            name = f"{topic}:generated_metric_{phrase}_{side}_{self._new_metric_idx:04d}"
            if name not in exclude_names:
                exclude_names.add(name)
                return db.add_metric(
                    MetricItem(
                        name=name,
                        topic=topic,
                        weight_adv=self.rng.uniform(0.85, 1.15),
                        weight_dis=self.rng.uniform(0.85, 1.15),
                    )
                )

        fallback_name = f"{topic}:generated_metric_{side}_{self.rng.random():.8f}"
        return db.add_metric(MetricItem(name=fallback_name, topic=topic))

    def _sample_polarity(self) -> str:
        return "adv" if self.rng.random() >= 0.35 else "dis"

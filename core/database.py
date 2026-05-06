from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


@dataclass
class TopicStat:
    hit_count: int = 0
    miss_count: int = 0

    @property
    def total(self) -> int:
        return self.hit_count + self.miss_count

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hit_count / self.total


@dataclass
class MetricItem:
    name: str
    topic: str
    weight_adv: float = 1.0
    weight_dis: float = 1.0
    hit_count: int = 0
    miss_count: int = 0
    ambiguous_count: int = 0
    topic_stats: Dict[str, TopicStat] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return self.hit_count + self.miss_count

    @property
    def global_hit_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count

    @property
    def ambiguous_ratio(self) -> float:
        return self.ambiguous_count / max(1, self.total_count + self.ambiguous_count)

    def record_observation(self, topic: str, is_hit: bool, is_ambiguous: bool = False) -> None:
        if is_hit:
            self.hit_count += 1
        else:
            self.miss_count += 1

        if is_ambiguous:
            self.ambiguous_count += 1

        topic_stat = self.topic_stats.setdefault(topic, TopicStat())
        if is_hit:
            topic_stat.hit_count += 1
        else:
            topic_stat.miss_count += 1

    def hit_rate_for_topic(self, topic: str) -> float:
        return self.topic_stats.get(topic, TopicStat()).hit_rate

    def best_topic_hit_rate(self) -> float:
        if not self.topic_stats:
            return 0.0
        return max(stat.hit_rate for stat in self.topic_stats.values())


@dataclass(frozen=True)
class MetricReason:
    metric: MetricItem
    polarity: str
    source: str

    @property
    def base_weight(self) -> float:
        if self.polarity == "adv":
            return self.metric.weight_adv
        return self.metric.weight_dis

    def label(self) -> str:
        return f"{self.metric.name}[{self.polarity}/{self.source}]"


@dataclass(frozen=True)
class DatabaseConfig:
    hit_reward: float = 0.18
    miss_penalty: float = 0.10
    ambiguous_penalty: float = 0.04
    min_weight: float = 0.05
    max_weight: float = 8.0
    topic_top_k: int = 3


class MetricDatabase:
    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self.metrics: List[MetricItem] = []
        self._metric_by_name: Dict[str, MetricItem] = {}

    def __len__(self) -> int:
        return len(self.metrics)

    def add_metric(self, metric: MetricItem) -> MetricItem:
        existing = self._metric_by_name.get(metric.name)
        if existing is not None:
            return existing

        self.metrics.append(metric)
        self._metric_by_name[metric.name] = metric
        return metric

    def extend(self, metrics: Iterable[MetricItem]) -> None:
        for metric in metrics:
            self.add_metric(metric)

    def sample_relevant_metrics(
        self,
        topic: str,
        count: int,
        rng: random.Random,
        exclude_names: Optional[set[str]] = None,
    ) -> List[MetricItem]:
        if count <= 0 or not self.metrics:
            return []

        excluded = exclude_names or set()
        candidates = [metric for metric in self.metrics if metric.name not in excluded]
        if not candidates:
            return []

        scored: List[Tuple[float, MetricItem]] = []
        for metric in candidates:
            topic_bonus = 1.2 if metric.topic == topic else 0.0
            history_bonus = 2.5 * metric.hit_rate_for_topic(topic)
            global_bonus = 0.8 * metric.global_hit_rate
            weight_bonus = 0.1 * (metric.weight_adv + metric.weight_dis)
            jitter = rng.uniform(0.0, 0.05)
            score = topic_bonus + history_bonus + global_bonus + weight_bonus + jitter
            scored.append((score, metric))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [metric for _, metric in scored[:count]]

    def top_k_topics_for_metric(self, metric: MetricItem, k: Optional[int] = None) -> List[str]:
        topic_count = self.config.topic_top_k if k is None else k
        ranked = sorted(
            metric.topic_stats.items(),
            key=lambda item: (item[1].hit_rate, item[1].total),
            reverse=True,
        )
        return [topic for topic, _ in ranked[:topic_count]]

    def update_metric_weights(
        self,
        topic: str,
        chosen_reasons: Sequence[MetricReason],
        rejected_reasons: Sequence[MetricReason],
    ) -> None:
        for reason in chosen_reasons:
            self._apply_single_update(reason, topic=topic, is_hit=True)

        for reason in rejected_reasons:
            self._apply_single_update(reason, topic=topic, is_hit=False)

        chosen_names = {reason.metric.name for reason in chosen_reasons}
        rejected_names = {reason.metric.name for reason in rejected_reasons}
        ambiguous_names = chosen_names & rejected_names
        if ambiguous_names:
            logger.info("Ambiguous metrics in same pair: %s", sorted(ambiguous_names))

        for reason in list(chosen_reasons) + list(rejected_reasons):
            if reason.metric.name in ambiguous_names:
                reason.metric.ambiguous_count += 1
                self._nudge_weight(reason.metric, reason.polarity, -self.config.ambiguous_penalty)

    def purge_metrics(self, n_rounds: int, top_x: int, threshold: float) -> List[MetricItem]:
        if n_rounds <= 0 or not self.metrics:
            return []

        ranked = sorted(self.metrics, key=lambda metric: metric.ambiguous_ratio, reverse=True)
        suspects = ranked[: min(top_x, len(ranked))]
        removed = [
            metric
            for metric in suspects
            if metric.total_count >= n_rounds and metric.best_topic_hit_rate() < threshold
        ]

        if not removed:
            logger.info("Purge: no metric removed.")
            return []

        removed_names = {metric.name for metric in removed}
        self.metrics = [metric for metric in self.metrics if metric.name not in removed_names]
        for name in removed_names:
            self._metric_by_name.pop(name, None)

        logger.info("Purge removed %d metric(s): %s", len(removed), sorted(removed_names))
        return removed

    def _apply_single_update(self, reason: MetricReason, topic: str, is_hit: bool) -> None:
        reason.metric.record_observation(topic=topic, is_hit=is_hit)
        delta = self.config.hit_reward if is_hit else -self.config.miss_penalty
        self._nudge_weight(reason.metric, reason.polarity, delta)

    def _nudge_weight(self, metric: MetricItem, polarity: str, delta: float) -> None:
        if polarity == "adv":
            metric.weight_adv = self._clip_weight(metric.weight_adv + delta)
        else:
            metric.weight_dis = self._clip_weight(metric.weight_dis + delta)

    def _clip_weight(self, value: float) -> float:
        return min(self.config.max_weight, max(self.config.min_weight, value))


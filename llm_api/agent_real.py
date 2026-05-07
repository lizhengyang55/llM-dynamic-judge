from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tiktoken
from openai import OpenAI

from core.database import MetricDatabase, MetricItem, MetricReason
from core.data_loader import PairwiseSample


logger = logging.getLogger(__name__)

total_tokens = 0


@dataclass(frozen=True)
class _BatchPairContext:
    pair_id: int
    sample: PairwiseSample
    candidate_metrics: List[MetricItem]
    db_count: int
    generated_count: int
    adv_count: int
    dis_count: int


class RealDualAgent:
    """OpenAI-compatible API adapter for pairwise metric extraction."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://yeysai.com/v1",
        model: str = "deepseek-v4-flash",
        reasons_per_side: int = 5,
        rng: Optional[random.Random] = None,
        max_retries: int = 1,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.reasons_per_side = reasons_per_side
        self.rng = rng or random.Random(42)
        self.max_retries = max_retries
        self._new_metric_idx = 0
        self._fallback_encoding = tiktoken.get_encoding("cl100k_base")
        self.api_attempts = 0
        self.api_successes = 0
        self.fallback_side_calls = 0
        self.fallback_pair_sides = 0

    def evaluate_pair(
        self,
        topic: str,
        ans_a: str,
        ans_b: str,
        db: MetricDatabase,
        m_value: float,
        user_prompt: Optional[str] = None,
    ) -> Tuple[List[MetricReason], List[MetricReason]]:
        sample = PairwiseSample(
            user_prompt=user_prompt or topic,
            topic=topic,
            answer_a=ans_a,
            answer_b=ans_b,
            label="A",
        )
        return self.evaluate_pairs_batch([sample], db=db, m_value=m_value)[0]

    def evaluate_pairs_batch(
        self,
        batch_pairs: List[PairwiseSample],
        db: MetricDatabase,
        m_value: float,
    ) -> List[Tuple[List[MetricReason], List[MetricReason]]]:
        if not batch_pairs:
            return []

        contexts = self._build_batch_contexts(batch_pairs=batch_pairs, db=db, m_value=m_value)
        reasons_a = self._evaluate_batch_side(side="A", contexts=contexts, db=db)
        reasons_b = self._evaluate_batch_side(side="B", contexts=contexts, db=db)

        merged = list(zip(reasons_a, reasons_b))
        for idx, (side_a, side_b) in enumerate(merged, start=1):
            logger.info(
                "Batch pair %d reasons_A=%s reasons_B=%s",
                idx,
                [reason.label() for reason in side_a],
                [reason.label() for reason in side_b],
            )
        return merged

    def _build_batch_contexts(
        self,
        batch_pairs: List[PairwiseSample],
        db: MetricDatabase,
        m_value: float,
    ) -> List[_BatchPairContext]:
        raw_db_count = min(self.reasons_per_side, max(0, round(self.reasons_per_side * m_value)))
        contexts: List[_BatchPairContext] = []

        for pair_id, sample in enumerate(batch_pairs, start=1):
            candidate_metrics = db.sample_relevant_metrics(
                topic=sample.topic,
                count=max(raw_db_count, 12),
                rng=self.rng,
            )
            db_count = min(raw_db_count, len(candidate_metrics))
            generated_count = self.reasons_per_side - db_count
            adv_count = self.rng.randint(2, 4)
            dis_count = self.reasons_per_side - adv_count
            contexts.append(
                _BatchPairContext(
                    pair_id=pair_id,
                    sample=sample,
                    candidate_metrics=candidate_metrics,
                    db_count=db_count,
                    generated_count=generated_count,
                    adv_count=adv_count,
                    dis_count=dis_count,
                )
            )
        return contexts

    def _evaluate_batch_side(
        self,
        side: str,
        contexts: List[_BatchPairContext],
        db: MetricDatabase,
    ) -> List[List[MetricReason]]:
        messages = self._build_batch_messages(side=side, contexts=contexts)

        for attempt in range(self.max_retries + 1):
            try:
                raw_content = self._chat_json(messages=messages, max_tokens=3000)
                parsed = self._parse_batch_json_array(raw_content)
                return self._materialize_batch_reasons(
                    side=side,
                    parsed_payload=parsed,
                    contexts=contexts,
                    db=db,
                )
            except Exception as exc:
                logger.warning(
                    "Batch JSON failed for side %s attempt %d/%d: %s",
                    side,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

        logger.warning("Using batch fallback metrics for side %s.", side)
        self.fallback_side_calls += 1
        self.fallback_pair_sides += len(contexts)
        return [
            self._fallback_reasons(
                topic=context.sample.topic,
                db=db,
                candidate_metrics=context.candidate_metrics,
                db_count=context.db_count,
                generated_count=context.generated_count,
                adv_count=context.adv_count,
            )
            for context in contexts
        ]

    def _build_batch_messages(
        self,
        side: str,
        contexts: List[_BatchPairContext],
    ) -> List[Dict[str, str]]:
        pair_blocks = []
        for context in contexts:
            metric_block = self._format_metric_block(context.candidate_metrics)
            sample = context.sample
            if side == "A":
                adv_symbol = "i1"
                dis_symbol = "j2"
                target_answer = "Answer A"
                target_meaning = "A over B advantages"
                dis_meaning = "B versus A disadvantages"
            else:
                adv_symbol = "j1"
                dis_symbol = "i2"
                target_answer = "Answer B"
                target_meaning = "B over A advantages"
                dis_meaning = "A versus B disadvantages"

            pair_blocks.append(
                f"""
[Pair {context.pair_id}]
Topic: {sample.topic}
User Prompt:
{sample.user_prompt}

Answer A:
{sample.answer_a}

Answer B:
{sample.answer_b}

Available Metric List for Pair {context.pair_id}:
{metric_block}

Counting constraints for Pair {context.pair_id}:
- Return exactly {self.reasons_per_side} reasons for {target_answer}.
- Return exactly {context.adv_count} reasons with polarity="adv" ({adv_symbol}: {target_meaning}).
- Return exactly {context.dis_count} reasons with polarity="dis" ({dis_symbol}: {dis_meaning}).
- {adv_symbol} + {dis_symbol} = {self.reasons_per_side}.
- Return exactly {context.db_count} reasons with source="db".
- Return exactly {context.generated_count} reasons with source="generated".
""".strip()
            )

        user_content = f"""
You are a strict batch pairwise LLM evaluation metric designer.
You are processing {len(contexts)} answer pairs in one API call from the perspective of side {side}.

{chr(10).join(pair_blocks)}

Output rules:
1. Output strict JSON only. No markdown, no comments, no trailing text.
2. Output one JSON array containing exactly {len(contexts)} objects.
3. Each object must have integer "pair_id" and array "reasons".
4. pair_id values must be exactly {[context.pair_id for context in contexts]}.
5. Every reasons item must have exactly these keys:
   - "name": string metric name
   - "polarity": either "adv" or "dis"
   - "source": either "db" or "generated"
6. For source="db", name must exactly match that pair's Available Metric List.
7. For source="generated", name must be a concise reusable evaluation metric.

Required JSON shape:
[
  {{
    "pair_id": 1,
    "reasons": [
      {{"name": "...", "polarity": "adv", "source": "db"}}
    ]
  }}
]
""".strip()

        return [
            {
                "role": "system",
                "content": "You output valid JSON only and obey all per-pair counting constraints.",
            },
            {"role": "user", "content": user_content},
        ]

    def _format_metric_block(self, metrics: Sequence[MetricItem]) -> str:
        if not metrics:
            return "(empty metric list)"
        return "\n".join(f"{idx + 1}. {metric.name}" for idx, metric in enumerate(metrics))

    def _chat_json(self, messages: Sequence[Dict[str, str]], max_tokens: int = 3000) -> str:
        self.api_attempts += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=0.2,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty response content")
        self.api_successes += 1
        self._accumulate_usage(response=response, messages=messages, completion=content)
        return content

    def _accumulate_usage(
        self,
        response: Any,
        messages: Sequence[Dict[str, str]],
        completion: str = "",
    ) -> None:
        global total_tokens

        usage = getattr(response, "usage", None)
        if usage is None:
            total_tokens += self._estimate_tokens(messages, completion=completion)
            return

        prompt_tokens = self._usage_int(usage, "prompt_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens")
        total_response_tokens = self._usage_int(usage, "total_tokens")
        if prompt_tokens or completion_tokens:
            total_tokens += prompt_tokens + completion_tokens
        elif total_response_tokens:
            total_tokens += total_response_tokens
        else:
            total_tokens += self._estimate_tokens(messages, completion=completion)

    def _usage_int(self, usage: Any, key: str) -> int:
        if isinstance(usage, dict):
            value = usage.get(key, 0)
        else:
            value = getattr(usage, key, 0)
        return int(value or 0)

    def _estimate_tokens(self, messages: Sequence[Dict[str, str]], completion: str = "") -> int:
        joined = "\n".join(message.get("content", "") for message in messages)
        return len(self._fallback_encoding.encode(joined)) + len(self._fallback_encoding.encode(completion))

    def _parse_json_array(self, raw_content: str) -> List[Dict[str, str]]:
        cleaned = self._strip_json_fence(raw_content)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("response is not a JSON array")
        return [self._validate_reason_item(item) for item in parsed]

    def _parse_batch_json_array(self, raw_content: str) -> List[Dict[str, object]]:
        cleaned = self._strip_json_fence(raw_content)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("batch response is not a JSON array")

        validated: List[Dict[str, object]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("batch item is not an object")
            pair_id = int(item.get("pair_id", 0))
            reasons = item.get("reasons")
            if not isinstance(reasons, list):
                raise ValueError(f"pair {pair_id} reasons is not an array")
            validated.append(
                {
                    "pair_id": pair_id,
                    "reasons": [self._validate_reason_item(reason) for reason in reasons],
                }
            )
        return validated

    def _strip_json_fence(self, raw_content: str) -> str:
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return cleaned

    def _validate_reason_item(self, item: Any) -> Dict[str, str]:
        if not isinstance(item, dict):
            raise ValueError("JSON reason item is not an object")

        name = str(item.get("name", "")).strip()
        polarity = str(item.get("polarity", "")).strip().lower()
        source = str(item.get("source", "")).strip().lower()
        if not name:
            raise ValueError("metric name is empty")
        if polarity not in {"adv", "dis"}:
            raise ValueError(f"invalid polarity: {polarity}")
        if source not in {"db", "generated"}:
            raise ValueError(f"invalid source: {source}")
        return {"name": name, "polarity": polarity, "source": source}

    def _materialize_batch_reasons(
        self,
        side: str,
        parsed_payload: Sequence[Dict[str, object]],
        contexts: Sequence[_BatchPairContext],
        db: MetricDatabase,
    ) -> List[List[MetricReason]]:
        del side
        payload_by_pair_id = {int(item["pair_id"]): item for item in parsed_payload}
        expected_pair_ids = {context.pair_id for context in contexts}
        if set(payload_by_pair_id) != expected_pair_ids:
            raise ValueError(
                f"pair_id mismatch: expected={sorted(expected_pair_ids)}, "
                f"actual={sorted(payload_by_pair_id)}"
            )

        batch_reasons: List[List[MetricReason]] = []
        for context in contexts:
            payload = payload_by_pair_id[context.pair_id]
            parsed_reasons = payload["reasons"]
            if not isinstance(parsed_reasons, list):
                raise ValueError(f"pair {context.pair_id} has invalid reasons payload")
            batch_reasons.append(
                self._materialize_reasons(
                    topic=context.sample.topic,
                    parsed_items=parsed_reasons,
                    db=db,
                    candidate_metrics=context.candidate_metrics,
                    db_count=context.db_count,
                    generated_count=context.generated_count,
                    adv_count=context.adv_count,
                    dis_count=context.dis_count,
                )
            )
        return batch_reasons

    def _materialize_reasons(
        self,
        topic: str,
        parsed_items: Sequence[Dict[str, str]],
        db: MetricDatabase,
        candidate_metrics: Sequence[MetricItem],
        db_count: int,
        generated_count: int,
        adv_count: int,
        dis_count: int,
    ) -> List[MetricReason]:
        candidate_by_name = {metric.name: metric for metric in candidate_metrics}
        normalized = self._normalize_counts(
            items=list(parsed_items),
            candidate_names=set(candidate_by_name),
            db_count=db_count,
            generated_count=generated_count,
            adv_count=adv_count,
            dis_count=dis_count,
        )

        reasons: List[MetricReason] = []
        for item in normalized:
            if item["source"] == "db":
                metric = candidate_by_name[item["name"]]
            else:
                metric = self._add_generated_metric(topic=topic, db=db, name=item["name"])
            reasons.append(
                MetricReason(
                    metric=metric,
                    polarity=item["polarity"],
                    source=item["source"],
                )
            )
        return reasons

    def _normalize_counts(
        self,
        items: List[Dict[str, str]],
        candidate_names: set[str],
        db_count: int,
        generated_count: int,
        adv_count: int,
        dis_count: int,
    ) -> List[Dict[str, str]]:
        unique: List[Dict[str, str]] = []
        seen_keys: set[Tuple[str, str]] = set()
        for item in items:
            name = item["name"]
            key = (name, item["source"])
            if key in seen_keys:
                continue
            if item["source"] == "db" and name not in candidate_names:
                continue
            unique.append(item)
            seen_keys.add(key)

        db_items = [item for item in unique if item["source"] == "db"][:db_count]
        generated_items = [item for item in unique if item["source"] == "generated"][:generated_count]
        selected = db_items + generated_items

        if len(db_items) != db_count or len(generated_items) != generated_count:
            raise ValueError("source count constraint not satisfied")
        if len(selected) != self.reasons_per_side:
            raise ValueError("reason count constraint not satisfied")

        selected = self._repair_polarity_counts(selected, adv_count=adv_count, dis_count=dis_count)
        return selected

    def _repair_polarity_counts(
        self,
        items: List[Dict[str, str]],
        adv_count: int,
        dis_count: int,
    ) -> List[Dict[str, str]]:
        if adv_count + dis_count != self.reasons_per_side:
            raise ValueError("adv/dis count must sum to reasons_per_side")

        repaired = [dict(item) for item in items]
        current_adv = sum(1 for item in repaired if item["polarity"] == "adv")
        idx = 0
        while current_adv > adv_count and idx < len(repaired):
            if repaired[idx]["polarity"] == "adv":
                repaired[idx]["polarity"] = "dis"
                current_adv -= 1
            idx += 1

        idx = 0
        while current_adv < adv_count and idx < len(repaired):
            if repaired[idx]["polarity"] == "dis":
                repaired[idx]["polarity"] = "adv"
                current_adv += 1
            idx += 1

        final_adv = sum(1 for item in repaired if item["polarity"] == "adv")
        final_dis = sum(1 for item in repaired if item["polarity"] == "dis")
        if final_adv != adv_count or final_dis != dis_count:
            raise ValueError("polarity count repair failed")
        return repaired

    def _fallback_reasons(
        self,
        topic: str,
        db: MetricDatabase,
        candidate_metrics: Sequence[MetricItem],
        db_count: int,
        generated_count: int,
        adv_count: int,
    ) -> List[MetricReason]:
        picked = list(candidate_metrics[:db_count])
        reasons: List[MetricReason] = []
        for idx, metric in enumerate(picked):
            polarity = "adv" if idx < adv_count else "dis"
            reasons.append(MetricReason(metric=metric, polarity=polarity, source="db"))

        for idx in range(generated_count):
            metric = self._add_generated_metric(
                topic=topic,
                db=db,
                name=f"{topic}:fallback_metric_{self._new_metric_idx + idx + 1}",
            )
            polarity = "adv" if len(reasons) < adv_count else "dis"
            reasons.append(MetricReason(metric=metric, polarity=polarity, source="generated"))

        return reasons[: self.reasons_per_side]

    def _add_generated_metric(self, topic: str, db: MetricDatabase, name: str) -> MetricItem:
        self._new_metric_idx += 1
        clean_name = name[:120]
        if ":" not in clean_name:
            clean_name = f"{topic}:{clean_name}"
        return db.add_metric(
            MetricItem(
                name=clean_name,
                topic=topic,
                weight_adv=self.rng.uniform(0.85, 1.15),
                weight_dis=self.rng.uniform(0.85, 1.15),
            )
        )

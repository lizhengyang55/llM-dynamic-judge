from __future__ import annotations

import random
from collections.abc import Sequence as RuntimeSequence
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


@dataclass(frozen=True)
class PairwiseSample:
    user_prompt: str
    topic: str
    answer_a: str
    answer_b: str
    label: str


def load_ultrafeedback_samples(
    train_size: int = 70,
    test_size: int = 200,
    seed: int = 42,
) -> Tuple[List[PairwiseSample], List[PairwiseSample]]:
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    total_size = train_size + test_size
    shuffled = dataset.shuffle(seed=seed).select(range(total_size))
    rng = random.Random(seed)
    samples = [_record_to_sample(record, rng) for record in shuffled]
    return samples[:train_size], samples[train_size:]


def _record_to_sample(record: Dict[str, Any], rng: random.Random) -> PairwiseSample:
    prompt = _extract_prompt(record)
    chosen = _textify(record.get("chosen", ""))
    rejected = _textify(record.get("rejected", ""))
    topic = _infer_topic(prompt)

    chosen_is_a = rng.random() >= 0.5
    if chosen_is_a:
        answer_a = chosen
        answer_b = rejected
        label = "A"
    else:
        answer_a = rejected
        answer_b = chosen
        label = "B"

    return PairwiseSample(
        user_prompt=prompt,
        topic=topic,
        answer_a=answer_a,
        answer_b=answer_b,
        label=label,
    )


def _extract_prompt(record: Dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if prompt:
        return _textify(prompt)

    chosen = record.get("chosen", "")
    if isinstance(chosen, RuntimeSequence) and not isinstance(chosen, (str, bytes)):
        for message in chosen:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", "")).strip()

    return ""


def _textify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        content = value.get("content")
        if content is not None:
            return str(content).strip()
        return "\n".join(f"{key}: {val}" for key, val in value.items())

    if isinstance(value, RuntimeSequence) and not isinstance(value, (bytes, bytearray)):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}" if role else content)
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(value).strip()


def _infer_topic(prompt: str) -> str:
    lowered = prompt.lower()
    coding_keywords = ["code", "python", "java", "function", "bug", "algorithm", "program"]
    math_keywords = ["math", "equation", "calculate", "proof", "probability", "integral"]
    writing_keywords = ["write", "essay", "story", "email", "summarize", "rewrite"]
    reasoning_keywords = ["reason", "explain", "why", "compare", "analyze", "decide"]

    if any(keyword in lowered for keyword in coding_keywords):
        return "coding"
    if any(keyword in lowered for keyword in math_keywords):
        return "math"
    if any(keyword in lowered for keyword in writing_keywords):
        return "writing"
    if any(keyword in lowered for keyword in reasoning_keywords):
        return "reasoning"
    return "general"

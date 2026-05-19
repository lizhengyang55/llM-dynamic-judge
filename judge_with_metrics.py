"""
Judge one Chatbot Arena sample using:
task -> five-dimensional classification -> metric retrieval -> A/B/tie judgment.
"""

import json
import re
import time
from pathlib import Path

from config import (
    DATASET_PATH,
    METRICS_OUTPUT_DIR,
    MODEL_NAME,
    create_openai_client,
    ensure_directories,
)


METRICS_DIR = METRICS_OUTPUT_DIR
ensure_directories()
client = create_openai_client()


def call_llm(
    prompt,
    system_msg="You are a precise AI assistant.",
    max_tokens=2048,
    temperature=0.1,
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            time.sleep(1)
        except Exception as exc:
            print(f"  [retry {attempt + 1}/{max_retries}] {exc}")
            time.sleep(2**attempt)
    return None


DIMENSION_LABELS = {
    "task_type": [
        "creative_writing",
        "knowledge_qa",
        "reasoning_math",
        "code_programming",
        "summarization_rewriting",
        "conversation_roleplay",
        "advice_planning",
        "extraction_analysis",
        "translation_language",
        "other",
    ],
    "difficulty": ["easy", "medium", "hard", "very_hard"],
    "domain": [
        "science_tech",
        "humanities_social",
        "daily_life",
        "business_finance",
        "education_academic",
        "entertainment_culture",
        "law_politics",
        "creative_arts",
        "general",
    ],
    "response_format": [
        "short_answer",
        "explanation",
        "list_steps",
        "long_form",
        "code_structured",
        "open_ended",
    ],
    "capability": [
        "factual_accuracy",
        "logical_reasoning",
        "creativity_imagination",
        "instruction_following",
        "communication_clarity",
        "domain_expertise",
        "empathy_tone",
        "multilingual",
    ],
}


def classify_task(task_text):
    """Classify a task along the five dimensions in one LLM call."""
    query = task_text[:1000] if len(task_text) > 1000 else task_text

    prompt = f"""Classify this user query along 5 dimensions. Output ONLY one label per line.

Query: {query}

1) task_type: {', '.join(DIMENSION_LABELS['task_type'])}
2) difficulty: {', '.join(DIMENSION_LABELS['difficulty'])}
3) domain: {', '.join(DIMENSION_LABELS['domain'])}
4) response_format: {', '.join(DIMENSION_LABELS['response_format'])}
5) capability: {', '.join(DIMENSION_LABELS['capability'])}

Format (exactly 5 lines):
task_type: <label>
difficulty: <label>
domain: <label>
response_format: <label>
capability: <label>"""

    response = call_llm(prompt, temperature=0.0, max_tokens=200)
    if not response:
        return {dimension: labels[-1] for dimension, labels in DIMENSION_LABELS.items()}

    return _parse_labels(response)


def _parse_labels(text):
    """Parse the five-dimensional labels."""
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        for dim_id, valid in DIMENSION_LABELS.items():
            if dim_id in line.lower():
                raw = line.split(":", 1)[1].strip().lower().strip('"\'`*. ')
                matched = next((label for label in valid if label == raw), None)
                if not matched:
                    matched = next((label for label in valid if label in raw), valid[-1])
                result[dim_id] = matched
                break

    for dim_id, valid in DIMENSION_LABELS.items():
        if dim_id not in result:
            result[dim_id] = valid[-1]
    return result


_metrics_cache = None


def load_metrics_index():
    """Load metric files and build a (dimension_id, group_label) -> metrics index."""
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    index = {}
    for metric_file in Path(METRICS_DIR).glob("metrics_*.json"):
        if "summary" in metric_file.name:
            continue
        try:
            data = json.loads(metric_file.read_text(encoding="utf-8"))
            key = (data["dimension_id"], data["group_label"])
            index[key] = [
                {
                    "metric_name": metric["metric_name"],
                    "metric_description": metric["metric_description"],
                    "accuracy": metric.get("accuracy", 0),
                }
                for metric in data.get("verified_metrics", [])
            ]
        except Exception:
            pass

    _metrics_cache = index
    return index


def retrieve_metrics(labels_dict):
    """Retrieve all metrics matching the five-dimensional labels."""
    index = load_metrics_index()
    matched = []
    for dim_id, label in labels_dict.items():
        key = (dim_id, label)
        if key in index:
            for metric in index[key]:
                matched.append({**metric, "dimension_id": dim_id, "group_label": label})

    matched.sort(key=lambda item: item["accuracy"], reverse=True)
    return matched


def _extract_reply(conversation):
    parts = []
    for turn in conversation:
        if turn["role"] == "assistant":
            content = turn["content"]
            if len(content) > 1500:
                content = content[:1500] + "...[truncated]"
            parts.append(content)
    return "\n".join(parts) if parts else "[empty]"


def _extract_query(conversation):
    for turn in conversation:
        if turn["role"] == "user":
            content = turn["content"]
            return content[:1500] if len(content) > 1500 else content
    return ""


def _format_metrics_block(metrics_list):
    if not metrics_list:
        return "No specific metrics available. Use your best general judgment."
    lines = []
    for i, metric in enumerate(metrics_list, 1):
        lines.append(
            f"Metric {i} (from {metric['dimension_id']}/{metric['group_label']}, "
            f"accuracy={metric['accuracy']:.0%}):\n"
            f"  {metric['metric_name']}: {metric['metric_description']}"
        )
    return "\n\n".join(lines)


def judge(sample, labels_dict, metrics_list):
    """
    Use retrieved metrics as criteria and ask the LLM to choose A, B, or tie.
    """
    query = _extract_query(sample["conversation_a"])
    reply_a = _extract_reply(sample["conversation_a"])
    reply_b = _extract_reply(sample["conversation_b"])
    metrics_block = _format_metrics_block(metrics_list)

    prompt = f"""You are an AI response evaluator. Judge which assistant gave a better response.

## Query Classification
- task_type: {labels_dict.get('task_type')}
- difficulty: {labels_dict.get('difficulty')}
- domain: {labels_dict.get('domain')}
- response_format: {labels_dict.get('response_format')}
- capability: {labels_dict.get('capability')}

## Evaluation Metrics (apply ALL of these)
{metrics_block}

## Responses

[User Query]:
{query}

[Assistant A]:
{reply_a}

[Assistant B]:
{reply_b}

## Instructions
Evaluate both responses against each metric above, then give a final verdict.
If very close in quality, say tie.

Output exactly:
REASONING: <your analysis in 2-3 sentences>
VERDICT: <A or B or tie>"""

    response = call_llm(
        prompt,
        system_msg="You are a fair and rigorous evaluator. Be objective.",
        max_tokens=600,
        temperature=0.1,
    )

    if not response:
        return "tie", "API failed"

    return _parse_verdict(response)


def _parse_verdict(text):
    reasoning = ""
    verdict = "tie"

    reasoning_match = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nVERDICT|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    verdict_match = re.search(r"VERDICT\s*:\s*(A|B|tie)", text, re.IGNORECASE)
    if verdict_match:
        parsed = verdict_match.group(1).strip().upper()
        verdict = parsed if parsed in ("A", "B") else "tie"

    return verdict, reasoning


def judge_sample(sample):
    """
    Full flow: task -> classification -> metric retrieval -> judgment.

    Returns:
      dict: {labels, metrics_used, verdict, reasoning}
    """
    task = _extract_query(sample["conversation_a"])

    labels = classify_task(task)
    metrics = retrieve_metrics(labels)
    verdict, reasoning = judge(sample, labels, metrics)

    return {
        "labels": labels,
        "metrics_used": [
            {
                "dimension": metric["dimension_id"],
                "group": metric["group_label"],
                "name": metric["metric_name"],
                "accuracy": metric["accuracy"],
            }
            for metric in metrics
        ],
        "verdict": verdict,
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    from datasets import load_from_disk

    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    row = next(
        item for item in dataset["train"]
        if "model_a" in item["winner"] or "model_b" in item["winner"]
    )
    sample = {
        "conversation_a": row["conversation_a"],
        "conversation_b": row["conversation_b"],
        "winner": row["winner"],
    }

    print("Judging...\n")
    result = judge_sample(sample)

    print(f"Labels:  {result['labels']}")
    print(f"Metrics: {len(result['metrics_used'])}")
    for metric in result["metrics_used"]:
        print(
            f"  - [{metric['dimension']}/{metric['group']}] "
            f"{metric['name']} ({metric['accuracy']:.0%})"
        )
    print(f"Verdict: {result['verdict']}")
    print(f"Reason:  {result['reasoning']}")

    ground_truth = row["winner"]
    if "model_a" in ground_truth:
        ground_truth = "A"
    elif "model_b" in ground_truth:
        ground_truth = "B"
    else:
        ground_truth = "tie"
    print(f"GT:      {ground_truth}")
    print(f"Correct: {result['verdict'] == ground_truth}")
